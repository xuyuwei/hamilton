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
    if float(count)/len(model_buckets[sparsity]) >= .5:
        return True
    return False


# get hamilton cycle from a graph according to our algorithm
# returns list of edges if cycle is found, otherwise returns empty list
def get_hamilton(graph):
    # if graph has no hamilton cycle
    if graph.num_nodes > graph.num_edges or not contains_hamilton(graph):
        print 'no cycle'
        return []

    if graph.num_edges == graph.num_nodes:
        print 'found hamilton cycle'
        return graph.edges

    cycle = set()
    B = set()       # set B from the algorithm
    used = set()    # set of edges already chosen
    while True:
        # if we find a cycle
        if len(cycle) == graph.num_nodes:
            print 'found hamilton cycle'
            return graph.get_edges(list(cycle))

        # if we go through all the edges, we must have not found the cycle
        if len(used) == graph.num_edges:
            print "we found %d/%d of the edges in the hamilton cycle" % (len(cycle), graph.num_nodes)
            return graph.get_edges(list(cycle))

        # choose a random edge that has not been used yet
        edge_index = graph.choose_random_edge(used)
        edge = graph.get_edges(edge_index)

        used.add(edge_index)

        # add this edge to B, remove B from graph
        B.add(edge_index)
        graph.remove_edges(list(B))
        # if there exists no hamilton cycle, that edge must be in the cycle
        if not contains_hamilton(graph):
            cycle.add(edge_index)
            B.remove(edge_index)

        # reset the graphs edges to show all of them
        graph.reset()


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
    graphs = load_data('data/test_data/test.txt')[0:30]
    score = 0
    for g in graphs:
        # print g.num_nodes, g.label, g.num_edges
        contains = contains_hamilton(g)
        sparsity = int(g.get_sparsity() * 100)
        edges = get_hamilton(g)
        predict = 0
        if len(edges) > 0:
            if is_hamilton_cycle(edges):
                print "confirmed cycle"
            else:
                print 'not a cycle tho'
            predict = 1
        if predict == g.label:
            score += 1
        else:
            print g.num_nodes, g.label, g.get_sparsity()
    print (float(score) / len(graphs))
