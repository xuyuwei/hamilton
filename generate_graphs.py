import matgraph
from tqdm import tqdm

for i in range(1000):
    graphsize = matgraph.random.randint(5,100)
    matgraph.MatGraph(graphsize).writeTSPFormat()
