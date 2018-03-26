import Graph as Graph
import time
import random
import os
import multiprocessing
import numpy as np

class DataSet:

    def __init__(self, points = 1000, ratio = 0.1):
        
        self.points = points
        self.ratio = ratio
        self.filename = str(self.ratio).replace('.', '-')
        self.filename += '_' + str(self.points) + '.txt'
        self.dirname = self.filename[0:-4]
        self.num_hc = 0

        # Start data
        self.data = str(self.points) + '\n'

        # Create Vectors for 10fold_idx
        self.fold_size = (int)(points/10)
        self.folds = np.arange(self.points).reshape((10,self.fold_size))
        np.random.permutation(self.folds)

    # Call for concurrent generation
    def ConcurrentGen(self, low, high, thread_num, iters, out_q):
        # Generate data
        output = ''
        for i in range(0,iters):
            if i%20 == 0:
                print('Iter: ', i, '   Cycle: ', self.num_hc, '  No Cycle: ' ,i-self.num_hc, 'Thread Num: ', thread_num)
            n = random.randint(low,high)
            g = Graph.Graph(dim=n, ratio = self.ratio)
            g.HasCycle(thread_num = thread_num)
            g.adj_to_s2v()
            if g.hc:
                self.num_hc += 1
            output += g.s2v
        out_q.put(output)


    def Generate(self, low = 20, high = 80, num_threads = 1):

        # Time 
        start_time = time.time()

        # Genereate data in parallel
        processes = []
        iters = (int)(self.points/num_threads)
        out_q = multiprocessing.Queue()

        # Create processes
        for i in range(num_threads):
            args = {'low':low,'high':high,'thread_num':i,'iters':iters,'out_q':out_q}
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

        # Use all data in each fold for now
        # Note that train and test are also the same
        train_data = ''
        for i in range(0,self.points):
            train_data += str(i) + '\n'


        # Create 10fold_idx files
        for i in range(1,11):
            train_name = 'train_idx-' + str(i) + '.txt'
            test_name = 'test_idx-' + str(i) + '.txt'

            # Write and move train
            f = open(train_name, 'w')
            f.writelines(train_data)
            f.close()
            mv_file = 'mv ' + train_name + ' ' + self.dirname + '/10fold_idx'
            os.system(mv_file)

            # Write and move test
            f = open(test_name, 'w')
            f.writelines(train_data)
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


