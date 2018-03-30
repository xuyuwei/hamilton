import argparse
import torch
import os
import networkx as nx
from util import S2VGraph

# Regular expression for the model files
MODEL_FILE_REGEX = r'epoch-best([0-9]+-[0-9]+)\w+'
MODELS = []

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
    

def import_models(model_files):
    # Load models
    for file in model_file:
        model = torch.load(file)
        # get the edge % the model has been trained on
        re_match = re.search(MODEL_FILE_REGEX, os.path.basename(file))
        lower_p, upper_p = re_match.group(1).split('-')
        MODELS.append([lower_p, upper_p, model])
    MODELS.sort()

if __name__=='__main__':
    load_data('data/ACTUAL_DATA/01-03_50-100_50/01-03_50-100_50.txt')

        