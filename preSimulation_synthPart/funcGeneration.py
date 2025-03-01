import numpy as np
import networkx as nx

from scipy.sparse import coo_matrix, load_npz, save_npz


    
def erNet(N,nE):
    p = nE/N/(N-1)
    nedg = 0
    while nedg < nE:
        mtx = np.random.binomial(1, p, (N,N))
        np.fill_diagonal(mtx,0)
        nedg = np.sum(mtx)
    mtx = coo_matrix(mtx)    
    edg = np.column_stack((mtx.col,mtx.row))
    edg = np.random.permutation(edg)
    print('pre ' , len(edg))
    edg = edg[:nE]
    mtx = coo_matrix((np.ones(len(edg)), (edg[:,0], edg[:,1])), shape=(N,N))
    return mtx

def swNet(N,nE):
    k = int(np.ceil(nE/N))    
    prand = 0.02
    ### preferred based on time cost for large networks
    ## ring lattice:
    mtx = np.zeros((N, N), dtype=int) 
    rows = np.arange(N)   
    cols = ((rows[:, None] - np.arange(1, k + 1)) % N).astype(int)  
    mtx[rows[:, None], cols] = 1
    
    mtx = coo_matrix(mtx)
    edg = np.column_stack((mtx.col,mtx.row))
    edg = np.random.permutation(edg)
    
    xmtx = coo_matrix(1-mtx.toarray())
    xedg = np.column_stack((xmtx.col,xmtx.row))
    xedg = np.random.permutation(xedg)

    nselect = int(prand*len(edg))
    edg[:nselect] = xedg[:nselect]
    edg = np.random.permutation(edg)
    print('pre ' , len(edg))
    edg = edg[:nE]
    mtx = coo_matrix((np.ones(len(edg)), (edg[:,0], edg[:,1])), shape=(N,N))
    return mtx
    
def sfNet(N,nE):  
    k = nE/N
    mm = N - np.sqrt(N**2-4*k*N)
    # mm = N - np.sqrt(N**2-8*nE)
    m = int(mm/2)+1

    gb = nx.barabasi_albert_graph(N, m)
    mtx = nx.to_numpy_array(gb)
    dxmtx = np.where(1-mtx)
    rndMtx = np.random.rand(N,N)
    rndMtx[dxmtx] = 0
    trup = np.triu(rndMtx)
    trlw = 1-trup.T
    trlw[trlw==1]=0
    tradd = trup + trlw
    mtx[tradd<0.5]=0
    
    mtx = coo_matrix(mtx)
    edg = np.column_stack((mtx.col,mtx.row))
    edg = np.random.permutation(edg)
    print('pre ' , len(edg))
    edg = edg[:nE]
    mtx = coo_matrix((np.ones(len(edg)), (edg[:,0], edg[:,1])), shape=(N,N))

    return mtx


def getNet(N,k,iname):
    if iname==0: #"er"
        edg = erNet(N,k)
    elif iname==1: #"sw"
        edg = swNet(N,k)
    elif iname==2: #"sf"
        edg = sfNet(N,k)
    else:
        # print
        return ValueError, "iname should be 'er', 'sw' or 'sf'"
    return edg
    

def relabelingNeurons(cmtx, perm=None):
    '''
    relabeling the sparse matrix. 
    i neuron in input matrix is perm[i] neuron in the output matrix, to avoid computational bias.
    '''
    if perm is None:
        perm = np.random.permutation(cmtx.shape[0])
    row_indices, col_indices = cmtx.nonzero()
    new_row_indices = perm[row_indices]
    new_col_indices = perm[col_indices]

    return coo_matrix((cmtx.data, (new_row_indices, new_col_indices)), shape=cmtx.shape)
    
    
def generateRelabelSave(params):
    if len(params)==2:
        iname, ireal = params
        N = 3611
        nE = 1404419
    else:
        iname, ireal, N, nE = params
    print(iname, ireal)
    permuter = np.load('%s/node_permutations_3611x10.npy'%pdir)[ireal]
    mtx = getNet(N,nE,iname)
    mtx1 = relabelingNeurons(mtx, permuter) # to destroy the generation order structure
    mtx2 = relabelingNeurons(mtx1, permuter) # to destroy the precedence of I neurons in labeling
    name = nettype[iname]
    # save_npz('%s/%s_%d_doubleRelabeled.npz'%(netdir, name, ireal), mtx2)
    return mtx, mtx1, mtx2



pdir = '../../../../nongit/data/pc_data'
netdir = 'data'
nettype = ['er', 'sw', 'sf']





