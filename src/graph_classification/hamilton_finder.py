import argparse
import torch
import os
import networkx as nx
import re
from main import Classifier
from util import S2VGraph, cmd_args

# Regular expression for the model files
MODEL_FILE_REGEX = r'epoch-best([0-9]+-[0-9]+)\w+'
MODELS = []


# basically same function as the one in util.py but without 10-fold
def load_data(data_file):
    print('loading data')

    g_list = []
    label_dict = {}
    feat_dict = {}

    with open(data_file, 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
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
    for g in g_list:
        g.label = label_dict[g.label]
    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim = len(feat_dict)
    return g_list
    

def import_models(model_files):
    # Load models
    for f in model_files:
        model = torch.load(f)
        # get the edge % the model has been trained on
        re_match = re.search(MODEL_FILE_REGEX, os.path.basename(f))
        lower_p, upper_p = re_match.group(1).split('-')
        MODELS.append([lower_p, upper_p, model])
    MODELS.sort()


# get the model prediction of whether the graph contains a hamilton cycle
def contains_hamilton(graph):
    # placeholder to just choose first model
    res, _, _ = MODELS[0][-1].forward([graph])
    return int(res.max(1)[-1]) # get the index with the larger element


# get hamilton cycle from a graph according to our algorithm
# returns list of edges if cycle is found, otherwise returns empty list
def get_hamilton(graph):
    # if graph has no hamilton cycle
    if not contains_hamilton(graph):
        print 'no cycle'
        return []
    
    cycle = set()
    B = set()  # set B from the algorithm
    while True:
        # if we find a cycle
        if len(cycle) == graph.num_nodes-1:
            print 'found hamilton cycle:'
            return graph.get_edges(list(cycle))

        # if we go through all the edges, we must have not found the cycle
        if len(cycle) + len(B) == graph.num_edges:
            print "we found %d/%d of the edges in the hamilton cycle" % (len(cycle), graph.num_nodes-1)
            return []

        # choose a random edge that has not been used yet
        edge_index = graph.choose_random_edge(cycle.union(B))

        # add this edge to B, remove B from graph
        B.add(edge_index)
        graph.remove_edges(list(B))

        # if there exists no hamilton cycle, that edge must be in the cycle
        if not contains_hamilton(graph):
            cycle.add(edge_index)
            B.remove(edge_index)

        # reset the graphs edges to show all of them
        graph.reset()


if __name__ == '__main__':
    import_models(['best-model/test-model/epoch-best01-03_test.model'])
    graphs = load_data('data/test_data/simple.txt')
    for g in graphs:
        print get_hamilton(g)
