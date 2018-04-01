from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os
import cPickle as cp
import networkx as nx

import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='mean_field', help='mean_field/loopy_bp')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=128, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=1, help='dimension of node feature')
cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-num_epochs', type=int, default=2000, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=int, default=64, help='dimension of latent layers')
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='s2v output size')
cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of regression')
cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-optim', default='Adam', help='optimizer')
cmd_opt.add_argument('-momentum', type=float, default=0., help='momentum')
cmd_opt.add_argument('-lr_decay', type=float, default=0., help='set decay of learning rate')
cmd_opt.add_argument('-save_dir', type=str, default='.', help='dir to save model in')
cmd_opt.add_argument('-models_dir', type=str, default='.', help='dir containing trained models')

cmd_args, _ = cmd_opt.parse_known_args()
print(cmd_args)


class S2VGraph(object):
    def __init__(self, g, node_tags, label):
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label

        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.edges = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edges[:, 0] = x
        self.edges[:, 1] = y
        self.edge_pairs = self.edges.flatten()

    # helpful for printing graphs
    def __repr__(self):
        return_str = "Graph {\n" \
                   + "\tnum_nodes : " + str(self.num_nodes) + "\n" \
                   + "\tnode_tags : " + str(self.node_tags) + "\n" \
                   + "\tlabel : " + str(self.label) + "\n" \
                   + "\tnum_edges : " + str(self.num_edges) + "\n" \
                   + "\tedge_pairs : " + str(self.edge_pairs) + "\n" \
                   + "}"
        return return_str

    # helper for getting some of the graph info
    def get_info(self):
        return ({
            'num_nodes': self.num_nodes,
            'node_tags': self.node_tags,
            'label': self.label,
            'num_edges': self.num_edges,
        })

    # return index of random edge from self.edges that is not in *chosen*
    def choose_random_edge(self, chosen):
        index = np.random.randint(0, len(self.edges)-len(chosen))
        count = 0
        for i in range(len(self.edges)):
            if i in chosen:
                continue
            if count == index:
                return i
            count += 1
        raise Exception("WTF is going on, there are no more edges to choose from")

    # get sparsity of graph
    def get_sparsity(self):
        return 2 * float(self.num_edges) / (self.num_nodes * (self.num_nodes-1))

    # given indices, get corresponding edges from graph
    def get_edges(self, indices):
        return np.take(self.edges, indices, 0)

    # given indices, remove edges in graph
    def remove_edges(self, indices):
        edges = np.delete(np.copy(self.edges), indices, 0)
        self.num_edges = len(edges)
        self.edge_pairs = edges.flatten()

    def reset(self):
        self.num_edges = len(self.edges)
        self.edge_pairs = self.edges.flatten()


def load_data(dataset):
    print('loading data')

    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('./data/%s/%s/%s.txt' % (cmd_args.data, dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())  # number of graphs in dataset
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]  # number of vertices, label
            # label_dict basically creates a dictionary that maps arbitrary labels
            # to integer labels (0, 1, ..) not really used in our case
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
    print('# classes: %d' % cmd_args.num_class)
    print('# node features: %d' % cmd_args.feat_dim)

    train_idxes = np.loadtxt('./data/%s/%s/10fold_idx/train_idx-%d.txt' % (cmd_args.data, dataset, cmd_args.fold), dtype=np.int32).tolist()
    test_idxes = np.loadtxt('./data/%s/%s/10fold_idx/test_idx-%d.txt' % (cmd_args.data, dataset, cmd_args.fold), dtype=np.int32).tolist()

    return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes]
