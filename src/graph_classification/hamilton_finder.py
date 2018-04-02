import argparse
import torch
import os
import networkx as nx
import re
import numpy as np
from main import Classifier
from util import S2VGraph, cmd_args

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
        # get the edge % the model has been trained on
        re_match = re.search(MODEL_FILE_REGEX, os.path.basename(f))
        lower_p, upper_p = list(map(int, re_match.group(1).split('-')))
        print f, lower_p, upper_p
        for i in range(lower_p, upper_p):
            model_buckets[i].append(model)
    print map(len, model_buckets)


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

    for model in model_buckets[sparsity]:
        output, loss, x = model([graph])
        if int(output.max(1)[-1]) == 1:
            count += 1
    if float(count)*len(model_buckets[sparsity]) >= 0.5:
        return True
    return False


# get hamilton cycle from a graph according to our algorithm
# returns list of edges if cycle is found, otherwise returns empty list
def get_hamilton(graph):
    # if graph has no hamilton cycle
    if graph.num_edges < graph.num_nodes or not contains_hamilton(graph):
        print 'no cycle'
        return []

    if graph.num_edges == graph.num_nodes:
        print 'found hamilton cycle'
        return graph.edges

    graph_edge_indices = set([i for i in range(graph.num_edges)])
    not_cycle = set()    # set of edges not in cycle
    for i in range (1):

        # if we go through all the edges, we must have not found the cycle
        if graph.num_edges == len(not_cycle):
            print 'could not find cycle'
            return []

        perm = np.random.permutation(graph.num_edges)
        # choose a random edge that has not been used yet
        for edge_index in perm:
            # if we find a cycle
            if graph.num_edges - len(not_cycle) == graph.num_nodes:
                print 'found hamilton cycle'
                cycle_edge_indices = graph_edge_indices
                for j in not_cycle:
                    cycle_edge_indices.remove(j)
                return graph.get_edges(list(cycle_edge_indices))
            if edge_index in not_cycle:
                continue

            # remove edge from graph
            not_cycle.add(edge_index)
            graph.remove_edges(list(not_cycle))

            # if there does not exist a hamilton cycle, that edge mmight be in the cycle
            if not contains_hamilton(graph):
                not_cycle.remove(edge_index)

            # reset the graphs edges to show all of them
            graph.reset()

    print ('%d node graph reduced from %d edges to %d edges' %
        (graph.num_nodes, graph.num_edges, graph.num_edges - len(not_cycle)))
    return []


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
    model_files = []
    for f in os.listdir(cmd_args.models_dir):
        if re.search(MODEL_FILE_REGEX, f):
            model_files.append(os.path.join(cmd_args.models_dir, f))
    import_models(model_files)
    graphs = load_data('data/test_data/test.txt')[0:20]
    score = 0
    for g in graphs:
        # print g.num_nodes, g.label, g.num_edges
        edges = get_hamilton(g)
        predict = 0
        if len(edges) > 0:
            if is_hamilton_cycle(edges):
                print "confirmed cycle"
            else:
                print 'not a cycle tho'

        if contains_hamilton(g):
            predict = 1
        if predict == g.label:
            score += 1
        else:
            print graphs
    print (float(score) / len(graphs))
