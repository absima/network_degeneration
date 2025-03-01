import numpy as np

import time, itertools
from multiprocessing import Pool, cpu_count

from scipy.sparse import coo_matrix, load_npz, save_npz

from orderedPruningFunc import *

numNets = 3
numIters = 10

## reindexing the nodes to avoid any order bias 

def orderEdges(iparam):
    iname, ireal = iparam
    name = ['er', 'sw', 'sf'][iname]
    pmtx = load_npz('%s/%s_%d_doubleRelabeled.npz'%(netdir, name, ireal))
    ## ordering the links using maximum-matching algorithm for ordered pruning
    edgez = np.column_stack((pmtx.nonzero()))[:,::-1]
    edgez = edgez[edgez[:,0].argsort()]

    g0 = zenGraph(edgez)
    ordEdges, nMM = orderMMEdge(g0)	
    
    np.savetxt('/home/synthNetData/%s_%d_ordEdges.txt'%(name, ireal), ordEdges, fmt='%d')
    np.savetxt('/home/synthNetData/%s_%d_numMM.txt'%(name, ireal), nMM, fmt='%d')
    return 
    
    


paramList = list(itertools.product(np.arange(numNets), np.arange(numIters)))

if __name__ == '__main__':
    __spec__ = None
    start_time = time.strftime("%H:%M:%S", time.localtime())

    ppl = Pool(processes=cpu_count()-1)
    ppl.map(orderEdges, paramList)

    finish_time = formatted_time = time.strftime("%H:%M:%S", time.localtime())
    
    print('started at: ', start_time)
    print('stopped at: ', finish_time)
    # print("finished in {} seconds".format(finish_time-start_time))




