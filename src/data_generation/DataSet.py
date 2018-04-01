import Graph as Graph
import time
import random
import os
import multiprocessing
import numpy as np

class DataSet:

    def __init__(self, points = 11000, edge_range = [0.02,0.2], node_range = [20, 100], prob_hc = 0):

        self.points = points
        self.edge_range = edge_range
        self.node_range = node_range
        self.prob_hc = prob_hc

        low_range = str(self.edge_range[0])[2:]
        high_range = str(self.edge_range[1])[2:]
        hc_val = str(self.prob_hc)[2:]
        if (len(low_range) == 1):
            low_range += '0'
        if (len(high_range) == 1):
            high_range += '0'
        if (len(hc_val) == 1):
            hc_val += '0'

        # Get filename
        self.filename = low_range + '-'
        self.filename += high_range + '_'
        self.filename += str(self.node_range[0]) + '-'
        self.filename += str(self.node_range[1]) + '_'
        self.filename += hc_val +  '.txt'
        self.dirname = self.filename[0:-4]
        self.num_hc = 0

        # Start data
        self.data = str(self.points) + '\n'

        # Create Vectors for 10fold_idx
        self.fold_size = (int)(points/10)
        self.folds = np.arange(self.points).reshape((10,self.fold_size))
        np.random.permutation(self.folds)

    # Call for concurrent generation
    def ConcurrentGen(self, thread_num, iters, out_q):
        # Generate data
        output = ''
        for i in range(0,iters):

            if (i+1)%500 == 0:
                print('Iter: ', i+1, '   Cycle: ', self.num_hc, '  No Cycle: ' ,i-self.num_hc, 'Thread Num: ', thread_num)

            if i == iters-1:
                print('Thread ', thread_num, ' is done')

            n = random.randint(self.node_range[0],self.node_range[1])
            edges = np.random.uniform(self.edge_range[0],self.edge_range[1])
            prob = np.random.uniform()
            # Add cycle with probabiolity
            if prob < self.prob_hc:
                g = Graph.Graph(dim = n, ratio = edges, hc = 1)
            else:
                g = Graph.Graph(dim = n, ratio = edges)

            g.HasCycle(thread_num = thread_num)
            g.adj_to_s2v()
            if g.hc:
                self.num_hc += 1
            output += g.s2v
        out_q.put(output)


    def Generate(self, num_threads = 1):

        # Print dataset info
        print('Generating ', self.points, ' instances')
        print('  Number of nodes: ', self.node_range[0], '-', self.node_range[1])
        print('  Approximate number of edges as a ratio: ', self.edge_range[0], '-', self.edge_range[1])
        print('  Prob. of adding a cycle: ', self.prob_hc)

        # Time
        start_time = time.time()

        # Genereate data in parallel
        processes = []
        iters = (int)(self.points/num_threads)
        out_q = multiprocessing.Queue()

        # Create processes
        for i in range(num_threads):
            args = {'thread_num':i,'iters':iters,'out_q':out_q}
            p = multiprocessing.Process(target=self.ConcurrentGen, kwargs=args)
            processes.append(p)

        # Start each thread
        [p.start() for p in processes]

        # Add result to data
        for i in range(num_threads):
            self.data += out_q.get()

        # Wait on all to finish
        [p.join() for p in processes]

        # Remove concorde created files
        os.system('rm *.mas *.pul *.tsp *.sol *.sav *.res')

        # Total Time
        elapsed_time = time.time() - start_time
        print(elapsed_time)


    def ToDirectory(self):

        # Create directory
        mk_dir = 'mkdir ' + self.dirname
        os.system(mk_dir)

        # Write dataset file and move to directory
        f = open(self.filename, 'w')
        f.writelines(self.data)
        f.close()
        mv_file = 'mv ' + self.filename + ' ' + self.dirname
        os.system(mv_file)

        # Create 10fold_idx folder
        os.system('mkdir 10fold_idx')
        mv_dir = 'mv 10fold_idx/ ' + self.dirname + '/'
        os.system(mv_dir)

        # Create 10 fold CV set
        train_ind = []
        test_ind = []
        for j in range(1,11):
            train_data = ''
            test_data = ''
            for i in range(0,self.points):
                if i < j*self.fold_size or i >= (j+1)*self.fold_size:
                    train_data += str(i) + '\n'
                else:
                    test_data += str(i) + '\n'
            train_ind.append(train_data)
            test_ind.append(test_data)

        # Create 10fold_idx files
        for i in range(1,11):
            train_name = 'train_idx-' + str(i) + '.txt'
            test_name = 'test_idx-' + str(i) + '.txt'

            # Write and move train
            f = open(train_name, 'w')
            f.writelines(train_ind[i-1])
            f.close()
            mv_file = 'mv ' + train_name + ' ' + self.dirname + '/10fold_idx'
            os.system(mv_file)

            # Write and move test
            f = open(test_name, 'w')
            f.writelines(test_ind[i-1])
            f.close()
            mv_file = 'mv ' + test_name + ' ' + self.dirname + '/10fold_idx'
            os.system(mv_file)


    def ToDataFolder(self):
        mv = 'mv ' + self.dirname + '/'
        mv += ' ../graph_classification/data/'
        os.system(mv)

    def DeleteCurrentFolder(self):
        rm = 'rm -r ' + self.dirname
        os.system(rm)

    def DeleteDataFolder(self):
        rm = 'rm -r ../graph_classification/data/' + self.dirname
        os.system(rm)
