
import Graph as Graph
import time
import random
import os
import numpy as np
import DataSet as DataSet



d = DataSet.DataSet(10000,0.05,0.25)
d.DeleteCurrentFolder()
d.DeleteDataFolder()
d.Generate(low=50, high=150, num_threads=4)
d.ToDirectory()
d.ToDataFolder()

print('Done')