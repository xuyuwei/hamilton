import argparse
import torch
import os
import networkx as nx
import re
import numpy as np
from main import Classifier
from util import S2VGraph, cmd_args
import time


# Regular expression for the model files
MODEL_FILE_REGEX = r'epoch-best([0-9]+-[0-9]+)\w+'

# using buckets for models
model_buckets = [[] for i in range(35)]

# basically same function as the one in util.py but without 10-fold
def load_data(data_file):
    print('loading data')

    g_list = []
    feat_dict = {}

    with open(data_file, 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            #print(row)
            n, l = [int(w) for w in row]
            g = nx.Graph()
            node_tags = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                row = [int(w) for w in row]
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
            assert len(g.edges()) * 2 == n_edges
            assert len(g) == n
            g_list.append(S2VGraph(g, node_tags, l))
    cmd_args.num_class = 2
    cmd_args.feat_dim = len(feat_dict)
    return g_list


def import_models(model_files):
    # Load models
    for f in model_files:
        model = torch.load(f, map_location=lambda storage, loc: storage)
        re_match = re.search(MODEL_FILE_REGEX, os.path.basename(f))
        lower_p, upper_p = list(map(int, re_match.group(1).split('-')))
        for i in range(lower_p, upper_p):
            model_buckets[i].append([f, model])


# get the model prediction of whether the graph contains a hamilton cycle
def contains_hamilton(graph):
    sparsity = int(graph.get_sparsity() * 100)
    if sparsity == 0:
        return False
    if sparsity > len(model_buckets):
        raise Exception('We need a model for this sparsity level %d' % sparsity)
    # hack for now: only one model needs to return true
    if len(model_buckets[sparsity]) == 0:
        raise Exception('We need a model for this sparsity level %d' % sparsity)
    count = 0

    for model_name, model in model_buckets[sparsity]:
        output, loss, x = model([graph])
        if int(output.max(1)[-1]) == 1:
            count += 1
    if float(count)*len(model_buckets[sparsity]) >= 0.9:
        return True
    return False


def edges_to_matrix(num_nodes, edges):
    matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
    for x,y in edges:
        matrix[x][y] = 1
        matrix[y][x] = 1
    return matrix


# reduce graph into a simpler graph to solve for concorde
# return bool (if hamilton found), nxn matrix (reduced graph)
# matrix is empty list if no hamilton cycle exists
def reduce_graph(graph):
    # if graph has no hamilton cycle
    if graph.num_nodes > graph.num_edges or not contains_hamilton(graph):
        return False, [], graph.num_edges

    if graph.num_edges == graph.num_nodes:
        return True, edges_to_matrix(graph.num_nodes, graph.edges), graph.num_edges

    graph_edge_indices = set([i for i in range(graph.num_edges)])
    not_cycle = set()    # set of edges not in cycle

    ################# PARAMETERS TO MODIFY ##################
    b_size = int((graph.num_edges - graph.num_nodes) / 16)
    num_batches = graph.num_edges / 256
    #######################################################3

    batch = max(1, b_size)
    for i in range(num_batches):
        # choose a random edge that has not been used yet

        # if we find a cycle
        if graph.num_edges - len(not_cycle) == graph.num_nodes:
            cycle_edge_indices = graph_edge_indices
            for j in not_cycle:
                cycle_edge_indices.remove(j)
            return True, edges_to_matrix(graph.num_nodes,
                graph.get_edges(list(cycle_edge_indices))), graph.num_edges

        edges_to_remove = []
        for j in xrange(batch):
            edge_index = graph.choose_random_edge(not_cycle)
            edges_to_remove.append(edge_index)
            not_cycle.add(edge_index)

        # remove edges from graph
        graph.remove_edges(list(not_cycle))

        # if there does not exist a hamilton cycle, that edge mmight be in the cycle
        if not contains_hamilton(graph):
            for e_index in edges_to_remove:
                not_cycle.remove(e_index)

        # reset the graphs edges to show all of them
        graph.reset()

    print ('%d node graph reduced from %d edges to %d edges' %
        (graph.num_nodes, graph.num_edges, graph.num_edges - len(not_cycle)))

    # return reduced graph
    num_removed = (graph.num_edges - len(not_cycle))
    reduced_graph_edges = graph_edge_indices
    for j in not_cycle:
        reduced_graph_edges.remove(j)
    return False, edges_to_matrix(graph.num_nodes,
        graph.get_edges(list(reduced_graph_edges))), num_removed


def count_nodes(nodes):
    count = {}
    for n in nodes:
        if n not in count:
            count[n] = 1
        else:
            count[n] += 1
    return count


# helper to verify hamilton cycles
def is_hamilton_cycle(edges):
    count = {}
    for v in np.array(edges).flatten():
        if v not in count:
            count[v] = 0
        count[v] += 1
    for c in count:
        if count[c] != 2:
            return False
    return True


if __name__ == '__main__':
    
    # Keep track of correct responses
    cor = 0         # Correct output from concorde
    call_con = 0    # Total calls to concorde
    result = 0      # Total correct results
    cor_class = 0   # Total correct classification

    model_files = []
    for f in os.listdir(cmd_args.models_dir):
        if re.search(MODEL_FILE_REGEX, f):
            model_files.append(os.path.join(cmd_args.models_dir, f))
    import_models(model_files)
    
    # TEST SET DIRS
    #   50-100Node.txt - 50-100 Node graphs
    #   100-200Node.txt - 100-200 Node graphs
    #   500-1000Node.txt - 500-1000 Node graphs
    #   fhcpcs200.txt - Flinders challenge set <= 200 Nodes
    #   02-05_50-100.txt
    #   05-10_50-100.txt
    #   10-15_50-100.txt
    #   15-20_50-100.txt
    #   01-05_100-200.txt
    #   10-15_100-200.txt
    #   15-20_100-200.txt
    #   20-25_50-100.txt
    #
    # All Testing Dataset 
    #graphs = load_data('data/TESTSET/50-100Node.txt')
    #graphs = load_data('data/TESTSET/100-200Node.txt')
    #graphs = load_data('data/TESTSET/500-1000Node.txt')
    #graphs = load_data('data/TESTSET/fhcpcs200.txt')
    #graphs = load_data('data/TESTSET/02-05_50-100.txt')
    #graphs = load_data('data/TESTSET/05-10_50-100.txt')
    #graphs = load_data('data/TESTSET/10-15_50-100.txt')
    #graphs = load_data('data/TESTSET/15-20_50-100.txt')
    #graphs = load_data('data/TESTSET/20-25_50-100.txt')
    #graphs = load_data('data/TESTSET/01-05_100-200.txt')
    #graphs = load_data('data/TESTSET/05-10_100-200.txt')
    graphs = load_data('data/TESTSET/10-15_100-200.txt')
    #graphs = load_data('data/TESTSET/15-20_100-200.txt')    
    #graphs = load_data('data/TESTSET/20-25_100-200.txt')

    # Get time
    start = time.time()
    ratio = 0
    for g in graphs:
    
        bef = g.num_edges

        found_ham, matrix, aft = reduce_graph(g)

        if found_ham:
            print 'model found hamilton cycle'
            predict = 1
            if g.label == 1:
                result +=1

        elif len(matrix) == 0:
            if g.label == 0:
                print 'no hamilton cycle'
                result += 1
                cor_class += 1
            else:
                print 'wrong inital classification'

        else:

            # Initial classification is correct
            if g.label == 1:
                cor_class += 1
                call_con += 1

            k = float(bef - aft)/bef
            ratio += k 
            dim = len(matrix)

            pre_filename = str(0)
            filename = str(0) + 'input.tsp'
            writeval = ['NAME: ' + pre_filename + '\n', \
                'TYPE: TSP (M.~Hofmeister)' + '\n', \
                'DIMENSION: ' + str(dim) + '\n', \
                'EDGE_WEIGHT_TYPE: EXPLICIT' + '\n', \
                'EDGE_WEIGHT_FORMAT: FULL_MATRIX' + '\n', \
                'DISPLAY_DATA_TYPE: NO_DISPLAY' + '\n', \
                'EDGE_WEIGHT_SECTION' + '\n ']

            # Add matrix
            MAX_ROW_SIZE = 16
            cur_row_size = 0
            for i in range(0,dim):
                for j in range(0,dim):
                    if cur_row_size < MAX_ROW_SIZE:
                        writeval.append(str(1-(int)(matrix[i][j])) + ' ')
                        cur_row_size += 1
                    else:
                        writeval.append(str(1-(int)(matrix[i][j])) + '\n ')
                        cur_row_size = 0

            # Write to file
            writeval.append('\n EOF')
            f = open(filename, 'w')
            f.writelines(writeval)
            f.close()

            # Run concorde on file
            concorde_out = os.popen('concorde -x ' + filename + ' 2>/dev/null ').read()
            # Check output to find ham cycle
            if 'Optimal Solution: 0.00' in concorde_out:
                
                if g.label == 1:
                    cor += 1
                    result += 1
                    print 'cycle found by concorde'
               
            else:
                if g.label == 0:
                    #cor += 1
                    result += 1
                    print 'cycle not found by concorde, which is correct'
                else :
                    print 'cycle not found, but there should be'

    # Print Results                
    end = time.time()
    ratio = ratio/call_con
    print 'Correct initial classification %d/%d'%((cor_class),len(graphs))
    print 'Fraction of edges removed: %f'%(ratio)
    print 'Correct calls to concorde: %d/%d'%(cor,call_con)
    print 'Correct total: %d/%d'%(result, len(graphs))
    print 'Time using our method: %f'%(end-start)

    # COMPARE WITH CONCORDE
    start_con = time.time()

    iteration = 0
    for g in graphs:
        
        # Write to matrix format
        matrix = edges_to_matrix(g.num_nodes, g.edges)
        dim = len(matrix)

        pre_filename = str(0)
        filename = str(0) + 'input.tsp'
        writeval = ['NAME: ' + pre_filename + '\n', \
            'TYPE: TSP (M.~Hofmeister)' + '\n', \
            'DIMENSION: ' + str(dim) + '\n', \
            'EDGE_WEIGHT_TYPE: EXPLICIT' + '\n', \
            'EDGE_WEIGHT_FORMAT: FULL_MATRIX' + '\n', \
            'DISPLAY_DATA_TYPE: NO_DISPLAY' + '\n', \
            'EDGE_WEIGHT_SECTION' + '\n ']

        # Add matrix
        MAX_ROW_SIZE = 16
        cur_row_size = 0
        for i in range(0,dim):
            for j in range(0,dim):
                if cur_row_size < MAX_ROW_SIZE:
                    writeval.append(str(1-(int)(matrix[i][j])) + ' ')
                    cur_row_size += 1
                else:
                    writeval.append(str(1-(int)(matrix[i][j])) + '\n ')
                    cur_row_size = 0

        # Write to file
        writeval.append('\n EOF')
        f = open(filename, 'w')
        f.writelines(writeval)
        f.close()

        # Run concorde on file
        concorde_out = os.popen('concorde -x ' + filename + ' 2>/dev/null ').read()

    # Running time for concorde
    end_con = time.time()
    print 'Time concorde: %f'%(end_con-start_con)
    
