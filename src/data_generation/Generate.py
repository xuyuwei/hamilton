
import Graph as Graph
import time
import random
import os
import numpy as np
import DataSet as DataSet


d = DataSet.DataSet(points = 1000, 
	edge_range = [0.07,0.15], 
	node_range = [20, 100], 
	prob_hc = 0.20)
d.Generate(num_threads=4)
d.ToDirectory()
d.ToDataFolder()



'''
d = DataSet.DataSet(points = 18000, 
	edge_range = [0.0,0.09], 
	node_range = [20, 100], 
	prob_hc = 0.35)
d.Generate(num_threads=4)
d.ToDirectory()
d.ToDataFolder()

d = DataSet.DataSet(points = 15000, 
	edge_range = [0.09,0.10], 
	node_range = [20, 100], 
	prob_hc = 0.30)
d.Generate(num_threads=4)
d.ToDirectory()
d.ToDataFolder()


d = DataSet.DataSet(points = 15000, 
	edge_range = [0.10,0.11], 
	node_range = [20, 100], 
	prob_hc = 0.25)
d.Generate(num_threads=4)
d.ToDirectory()
d.ToDataFolder()


d = DataSet.DataSet(points = 22000, 
	edge_range = [0.16,0.20], 
	node_range = [20, 60], 
	prob_hc = 0)
d.Generate(num_threads=4)
d.ToDirectory()
d.ToDataFolder()


d = DataSet.DataSet(points = 10000, 
	edge_range = [0.08,0.10], 
	node_range = [50, 100], 
	prob_hc = 0)
d.Generate(num_threads=4)
d.ToDirectory()
d.ToDataFolder()


d = DataSet.DataSet(points = 10000, 
	edge_range = [0.10,0.12], 
	node_range = [50, 100], 
	prob_hc = 0)
d.Generate(num_threads=4)
d.ToDirectory()
d.ToDataFolder()


d = DataSet.DataSet(points = 8000, 
	edge_range = [0.12,0.15], 
	node_range = [50, 100], 
	prob_hc = 0)
d.Generate(num_threads=4)
d.ToDirectory()
d.ToDataFolder()



d = DataSet.DataSet(points = 5000, 
	edge_range = [0.16,0.20], 
	node_range = [50, 100], 
	prob_hc = 0)
d.Generate(num_threads=4)
d.ToDirectory()
d.ToDataFolder()
'''
print('Done')
