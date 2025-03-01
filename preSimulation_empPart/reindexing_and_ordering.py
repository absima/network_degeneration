import numpy as np
from scipy.sparse import coo_matrix

from funcDegeneration import *   
from orderedPruningFunc import *

from parameters import *



## reindexing the nodes to avoid any order bias 

def reindexAndOrder(ipermut):
    '''
    smtx is a sparse coo matrix 
    ipermut is the index of node permuter variable 
    
    output: ordered edges as indices for the new coo matrix saved.
            number of iterative MaxMatchs saved
    '''
    smtx = origordsparse.copy()
    permx = permuters[ipermut]
    pmtx = relabelingNeurons(smtx, perm=permx)
 

    ## ordering the links using maximum-matching algorithm for ordered pruning
    edgez = np.column_stack((pmtx.nonzero()))[:,::-1]
    edgez = edgez[edgez[:,0].argsort()]

    g0 = zenGraph(edgez)
    ordEdges, nMM = orderMMEdge(g0)	
    
    
    # ## recreate permuted matrix, but by keeping the order of indices as in ordEdges
    # colind, rowind = ordEdges.T
    #
    # ndta = pmtx.data[np.argsort(np.lexsort((rowind, colind)))]
    # opmtx = coo_matrix((ndta, (rowind, colind)), shape=pmtx.shape)
    #
    # save_npz('/data/conMat_perm%d.npz'%ipermut, pmtx)
    # save_npz('/data/ordConMat_perm%d.npz'%ipermut, opmtx)
    # save_npz('/data/nMM_perm%d.npz'%ipermut, nMM)
    
    np.savetxt('data/ordEdges_%d.txt'%ipermut, ordEdges, fmt='%d')
    np.savetxt('data/numMM_%d.txt'%ipermut, nMM, fmt='%d')
    
    

import time, itertools
from multiprocessing import Pool, cpu_count


if __name__ == '__main__':
    __spec__ = None
    start_time = time.strftime("%H:%M:%S", time.localtime())

    ppl = Pool(processes=cpu_count()-1)
    ppl.map(reindexAndOrder, np.arange(10))

    finish_time = formatted_time = time.strftime("%H:%M:%S", time.localtime())
    
    print('started at: ', start_time)
    print('stopped at: ', finish_time)
    # print("finished in {} seconds".format(finish_time-start_time))



# ipermut = 0
# smtx = origordsparse.copy()
# permx = permuters[ipermut]
# pmtx = relabelingNeurons(smtx, perm=permx)
#
#
# ## ordering the links using maximum-matching algorithm for ordered pruning
# edgez = np.column_stack((pmtx.nonzero()))[:,::-1]
#
# g0 = zenGraph(edgez)
# MM = np.array(zen.maximum_matching(g0))
#
#
#
# ipermut = 1
# smtx = origordsparse.copy()
# permx = permuters[ipermut]
# pmtx = relabelingNeurons(smtx, perm=permx)
#
#
# ## ordering the links using maximum-matching algorithm for ordered pruning
# edgez = np.column_stack((pmtx.nonzero()))[:,::-1]
# edgez = edgez[edgez[:,0].argsort()]
#
# g1 = zenGraph(edgez)
# MM = np.array(zen.maximum_matching(g1))



