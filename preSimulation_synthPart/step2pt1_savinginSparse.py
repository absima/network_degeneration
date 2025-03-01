import numpy as np

import time, itertools
from multiprocessing import Pool, cpu_count

from scipy.sparse import coo_matrix, load_npz, save_npz

from orderedPruningFunc import *

numNets = 3
numIters = 10

N = 3611
## reindexing the nodes to avoid any order bias 
## this extra step is because my docker did not handle saving npz properly. ... till that's fixed.
ifold = '/home/sima/project/nester/synth3611/synthNetData'
sfold = '/mnt/sdb/sima/data/nester/synthNets/permsANDnets'
for inet, net in enumerate(['er', 'sw', 'sf']):
    edg = np.loadtxt('%s/%s_%d_ordEdges.txt'%(ifold, name, ireal)).astype(int)
    src, tgt = edg.T
    cmtx = coo_matrix((np.ones(len(src)), (tgt, src)), shape=(N,N))
    save_npz('%s/%s_%d_sparseNet.npz'%(sfold, name, ireal), cmtx)
    



