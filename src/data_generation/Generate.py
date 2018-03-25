import Graph as Graph
import time
import random
import os
import numpy as np
import DataSet as DataSet


d = DataSet.DataSet(100,0.1)
d.DeleteCurrentFolder()
d.DeleteDataFolder()
d.Generate(num_threads=0)
d.ToDirectory()
d.ToDataFolder()
