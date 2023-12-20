import matplotlib.pyplot as plt
import numpy as np
import torch
from os.path import join

def read_sampling_pattern_from_file(fname):
    im = plt.imread(fname)
    im = np.expand_dims(im, 0 )
    im = np.expand_dims(im, 0 )
    return torch.tensor(im > 0)

def read_count(count_path, count_name="COUNT.txt"):
    """ Read and updates the runner count. 
    
    To keep track of all the different runs of the algorithm, one store the 
    run number in the file 'COUNT.txt' at ``count_path``. It is assumed that 
    the file 'COUNT.txt' is a text file containing one line with a single 
    integer, representing number of runs so far. 

    This function reads the current number in this file and increases the 
    number by 1. 
    
    :return: Current run number (int).
    """
    fname = join(count_path, count_name);
    infile = open(fname);
    data = infile.read();
    count = int(eval(data));
    infile.close();

    outfile = open(fname, 'w');
    outfile.write('%d ' % (count+1));
    outfile.close();
    return count;

def cut_to_01(im):
    idx_neg = im < 0
    idx_pos = im > 1
    im[idx_neg] = 0
    im[idx_pos] = 1
    return im


def nullspace_proj(x, opA):
    z = x - opA.adj(opA(x))
    return z

if __name__ == "__main__":

    fname = 'samp_patt/spf2_DAS_N_2048_db1.png'
    samp_patt = read_sampling_pattern_from_file(fname)




