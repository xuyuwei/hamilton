import os
import numpy as np
import string
import random
import hashlib


class Graph:

    # dim - Dimension of the adjaceny matrix
    # ratio - Approximate number of edges to include
    # hc - specify whether the graph is hamiltonian
    def __init__(self, dim = 100, ratio = 0.5, hc = -1):
        
        self.G = np.random.uniform(size=[dim, dim])
        self.dim = dim
        self.s2v = ''
        self.hc = 0


        # Adjust ratio to include edges
        if hc == 1:
            ratio -= (2*dim)/(dim*dim)
        
        # Set diagonal to 0
        for i in range(0,self.dim):
            self.G[i,i] = 0

        # Set edges based on threshold
        for i in range(0,self.dim):
            for j in range(i+1,self.dim):
                if self.G[i,j] >= 1 - ratio:
                    self.G[i,j] = 1
                    self.G[j,i] = 1
                else:
                    self.G[i,j] = 0
                    self.G[j,i] = 0

        # Add random hamiltonian cycle if specified
        if hc == 1:
            perm = np.random.permutation(dim)
            for i in range(0,dim-1):
                self.G[perm[i], perm[i+1]] = 1
                self.G[perm[i+1], perm[i]] = 1
            self.G[perm[0],perm[-1]] = 1
            self.G[perm[-1],perm[0]] = 1
        
    
    # Determine if G has a cycle using concorde
    # Concorde executable must be in PATH
    def HasCycle(self, thread_num = 0):

        # Header for file
        pre_filename = str(thread_num)
        filename = str(thread_num) + 'input.tsp'
        writeval = ['NAME: ' + pre_filename + '\n', \
            'TYPE: TSP (M.~Hofmeister)' + '\n', \
            'DIMENSION: ' + str(self.dim) + '\n', \
            'EDGE_WEIGHT_TYPE: EXPLICIT' + '\n', \
            'EDGE_WEIGHT_FORMAT: FULL_MATRIX' + '\n', \
            'DISPLAY_DATA_TYPE: NO_DISPLAY' + '\n', \
            'EDGE_WEIGHT_SECTION' + '\n ']

        # Add matrix
        MAX_ROW_SIZE = 16
        cur_row_size = 0
        for i in range(0,self.dim):
            for j in range(0,self.dim):
                if cur_row_size < MAX_ROW_SIZE:
                    writeval.append(str(1-(int)(self.G[i,j])) + ' ')
                    cur_row_size += 1
                else:
                    writeval.append(str(1-(int)(self.G[i,j])) + '\n ')
                    cur_row_size = 0

        # Write to file
        writeval.append('\n EOF')
        f = open(filename, 'w')
        f.writelines(writeval)
        f.close()

        # Run concorde on file
        concorde_out = os.popen('concorde ' + filename + ' 2>/dev/null ').read()
        
        # Remove concorde created files
        #os.system('rm input.* Oinput.*')
        
        # Check output to find ham cycle
        if 'Optimal Solution: 0.00' in concorde_out:
            self.hc = 1
        else:
            self.hc = 0



    
    # Print Graph
    def printGraph(self):
        for i in range(0,self.dim):
            for j in range(0,self.dim):
                print(self.G[i][j]+' ')
            print("\n")


    # Format graph for struct to vec input 
    def adj_to_s2v(self):

        # Header
        self.s2v = str(self.dim) + ' ' + str(self.hc) + '\n'

        # Adjacency
        for i in range(0,self.dim):
            x = np.where(self.G[i,:] == 1.)[0]
            self.s2v += '0 ' + str(x.shape[0]) + ' '
            for i in x:
                self.s2v += str((int)(i)) + ' '
            self.s2v += '\n'