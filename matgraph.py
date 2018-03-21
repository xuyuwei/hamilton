import numpy as np
import string
import random

class MatGraph:
    
    def __init__(self, size, adj_mat = None):
        self.size = size
        self.hc = -1
        if adj_mat is None:
            mat = np.random.randint(0,2,(size, size))
            adj_mat = (mat + mat.T)/2
            adj_mat[np.diag_indices(self.size)] = 0
            self.adj_mat = adj_mat
        else:
            self.adj_mat = adj_mat

    def __repr__(self):
        printval = str(self.size) + ' ' + str(self.hc) + '\n'
        for i in range(self.size):
            row_i = self.adj_mat[i]
            printval = printval + '0 ' + str(int(row_i.sum() - row_i[i]))
            for j in range(self.size):
                if row_i[j] and i != j:
                    printval = printval + ' ' + str(j)
            printval = printval + '\n'
        return printval

    def set_hc(self, hc):
        self.hc = hc

    def writeTSPFormat(self, pre_filename = 0):
        if isinstance(pre_filename, int):
            pre_filename = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
        filename = 'tspgraphs/' + pre_filename + '.tsp'

        writeval = ['NAME: ' + pre_filename + '\n', \
                    'TYPE: TSP (M.~Hofmeister)' + '\n', \
                    'DIMENSION: ' + str(self.size) + '\n', \
                    'EDGE_WEIGHT_TYPE: EXPLICIT' + '\n', \
                    'EDGE_WEIGHT_FORMAT: UPPER_DIAG_ROW' + '\n', \
                    'DISPLAY_DATA_TYPE: NO_DISPLAY' + '\n', \
                    'EDGE_WEIGHT_SECTION' + '\n']

        tsp_adj_mat = self.adj_mat
        tsp_adj_mat[tsp_adj_mat == 0] = 2
        upper_vals = tsp_adj_mat[np.triu_indices(self.size)]
        num_cur_row_vals = 0
        cur_row_string = ''
        for i in upper_vals:
            if num_cur_row_vals < 16:
                cur_row_string = cur_row_string + ' ' + str(i)
                num_cur_row_vals += 1
            else:
                cur_row_string = cur_row_string + '\n'
                writeval.append(cur_row_string)
                cur_row_string = ' ' + str(i)
                num_cur_row_vals = 1
        cur_row_string = cur_row_string + '\n'
        writeval.append(cur_row_string)

        writeval.append('EOF')
        
        f = file(filename, 'w')
        f.writelines(writeval)
        print(pre_filename)

def read_to_matgraph(filename):
    f = file(filename, 'r')
    lines = f.readlines()
    values = lines[7:-1]
    size = int(lines[2].split()[1])

    num_list = []
    for line in values:
        for num in line.split():
            num_list.append(int(num))

    col_start_index = 0
    num_nonzero = size
    cur_ind = 0
    adj_mat = np.zeros((size, size))
    for i in range(size):
        cur_weights = num_list[cur_ind:(cur_ind+num_nonzero)]
        adj_mat[i][col_start_index:] = cur_weights
        cur_ind = cur_ind + num_nonzero
        num_nonzero -= 1
        col_start_index += 1
    adj_mat[adj_mat == 2] = 0
    adj_mat[np.tril_indices(size)] = adj_mat.T[np.tril_indices(size)]

    mg = MatGraph(size, adj_mat)
    return mg
