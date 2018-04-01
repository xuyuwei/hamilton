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
    if not contains_hamilton(graph):
        print 'no cycle'
        return []

    start = []
    cycle = set()
    used_edges = set()    # set of edges already chosen

    # get first edge in hamilton cycle
    while True:
        if len(start) >= 1:
            break

        if len(used_edges) == graph.num_edges:
            print 'Could not find a single edge in hamilton cycle'
            return []

        # choose a random edge that has not been used yet
        edge_index = graph.choose_random_edge(used_edges)
        edge = graph.get_edges(edge_index)

        used_edges.add(edge_index)

        graph.remove_edges(list(used_edges))
        # if there exists no hamilton cycle, that edge must be in the cycle
        if not contains_hamilton(graph):
            start = list(graph.get_edges([edge_index]).flatten())
            used_edges.remove(edge_index)

        # reset the graphs edges to show all of them
        graph.reset()

    return find_hamilton(graph, start, used_edges)


def find_hamilton(graph, start, used_edges):
    visited = [False for i in range(graph.num_nodes)]
    for s in start:
        visited[s] = True
    return dfs(graph, start, visited, used_edges)


def dfs(graph, cur, visited, used_edges):
    first_vertex = cur[0]
    last_vertex = cur[-1]

    first_edge_indices, first_edges = graph.get_vertex_edges(first_vertex)
    last_edge_indices, last_edges = graph.get_vertex_edges(last_vertex)
    edge_indices = first_edge_indices + last_edge_indices
    all_possible_edges = first_edges + last_edges

    possible_next_edge = [] # possible next vertices
    edges_tried = []

    # print cur
    # print "edge_indices", edge_indices
    # print "all_possible_edges:", all_possible_edges
    # print 'used_edges:', used_edges

    for i, edge in enumerate(all_possible_edges):
        start, end = edge
        # if the edge connects the last vertex and the first one,
        # we've found the cycle
        if len(cur) == graph.num_nodes:
            if sorted(edge) == sorted([first_vertex, last_vertex]):
                print "FOUND THE CYCLE"
                return cur
            continue

        if visited[end]:
            if start == first_vertex and end != cur[1]:
                used_edges.add(edge_indices[i])
            elif end == last_vertex and end != cur[-2]:
                used_edges.add(edge_indices[i])
            continue

        used_edges.add(edge_indices[i])
        edges_tried.append(edge_indices[i])
        graph.remove_edges(list(used_edges))
        if not contains_hamilton(graph):
            possible_next_edge.append(edge)
            used_edges.remove(edge_indices[i])
        graph.reset()

    for edge in possible_next_edge:
        # dfs with recursive backtracking
        start, end = edge
        cur_copy = cur[:]
        if start == first_vertex:
            cur = [end] + cur
        else:
            cur.append(end)
        visited[end] = True

        # dfs
        ans = dfs(graph, cur, visited, used_edges)
        if ans: # if we get a legit cycle, return it
            return ans

        # backtrack
        cur = cur_copy
        visited[end] = False

    for e in edges_tried:
        if e in used_edges:
            used_edges.remove(e)
    return None


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
    print count
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
    graphs = load_data('data/test_data/test.txt')[1:3]
    score = 0
    for g in graphs:
        print g.num_edges
        contains = contains_hamilton(g)
        sparsity = int(g.get_sparsity() * 100)
        edges = get_hamilton(g)
        predict = 0
        if edges:
            print edges
            predict = 1
        if predict == g.label:
            score += 1
    print (float(score) / len(graphs))
