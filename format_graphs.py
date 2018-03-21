import matgraph
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

counter = 0

no_hc_graphs = []
no_hc_path = 'tspgraphs/no-hc'
for filename in tqdm(listdir(no_hc_path)):
    filename_full = join(no_hc_path, filename)
    mg = matgraph.read_to_matgraph(filename_full)
    mg.set_hc(0)
    if mg.adj_mat.sum() > 0:
        no_hc_graphs.append(mg)
        counter += 1

hc_graphs = []
hc_path = 'tspgraphs/hc'
for filename in tqdm(listdir(hc_path)):
    filename_full = join(hc_path, filename)
    mg = matgraph.read_to_matgraph(filename_full)
    mg.set_hc(1)
    hc_graphs.append(mg)
    counter += 1

f = file('graphdata.txt', 'w')
f.write(str(counter) + '\n')
for g in tqdm(no_hc_graphs):
    f.write(repr(g))
for g in tqdm(hc_graphs):
    f.write(repr(g))
f.close()
