from parameters import *


def permute(sparsematrix, perm): 
    src, tgt = sparsematrix.nonzero()
    wgt = sparsematrix.data   
    return coo_matrix((wgt[perm], (src[perm],tgt[perm])), shape=sparsematrix.shape)

def degreeFromSparceMatrix(cmtx):
    # cmtx = cmtx.tocsr()
    '''
    - coomtx is the sparse coo_matrix as input
    - it retuns the outdegrees and degrees of the nodes in the network
    ''' 
    row_degrees = np.bincount(cmtx.row, minlength=cmtx.shape[0])
    col_degrees = np.bincount(cmtx.col, minlength=cmtx.shape[1])
    degrees = row_degrees + col_degrees
    return col_degrees, degrees

def relabelingNeurons(cmtx, perm=[]):
    if not len(perm):
        perm = np.random.permutation(cmtx.shape[0])
    row_indices, col_indices = cmtx.nonzero()
    new_row_indices = perm[row_indices]
    new_col_indices = perm[col_indices]

    return coo_matrix((cmtx.data, (new_row_indices, new_col_indices)), shape=cmtx.shape)
        
def relabelingNeuronBlocks(cmtx, block1_permutation=[], block2_permutation=[]):
    '''
    This just relabels nodes with some random permutations (two of them for I and E). 
    '''
    # if isspmatrix_coo(smtx):
#         smtx = smtx.tocsr()
    row_indices, col_indices = cmtx.nonzero()
    data = smtx.data
    
    # reindexing inhibitory and excitatory neurons separately
    if  not (len(block1_permutation) * len(block2_permutation)):
        block1_permutation = np.random.permutation(NI)
        block2_permutation = np.random.permutation(NE)

    # new indices
    new_row_indices = row_indices.copy()
    new_col_indices = col_indices.copy()

    # shuffling rows and columns in for I population
    new_row_indices[row_indices < NI] = block1_permutation[row_indices[row_indices < NI]]
    new_col_indices[col_indices < NI] = block1_permutation[col_indices[col_indices < NI]]

    # shuffling rows and columns in for E population
    new_row_indices[row_indices >= NI] = NI + block2_permutation[row_indices[row_indices >= NI] - NI]
    new_col_indices[col_indices >= NI] = NI + block2_permutation[col_indices[col_indices >= NI] - NI]

    # creating the shuffled sparse matrix
    shuffled_smtx = coo_matrix((data, (new_row_indices, new_col_indices)), shape=smtx.shape)

    # # csr version, if preferred for efficiency
    # shuffled_smtx_csr = shuffled_smtx.tocsr()
    
    return shuffled_smtx
    
def sortNeurons(cmtx, idxPrun, permID=[]):
    '''
    cmtx is a sparse coo_matrix
    idxPrun is the index of a neuronal death
    permID is the new labels of neurons such that 
        the first NI IDs in PermID are inhibitory 
        the last NE IDs correspond to the excitatory neurons
    
    output is sorted I neurons and E neurons based on a degenerative strategy
    '''
    if not len(permID):
        permID = np.arange(cmtx.shape[0])
    new_Inrn = permID[:NI]
    new_Enrn = permID[NI:]
    
    outdegree, degree = degreeFromSparceMatrix(cmtx)
    
    prrm = np.arange(N0)
    np.random.shuffle(prrm)
    
    odd = np.column_stack((outdegree, degree, prrm, nrns))    
    if  idxPrun==0:  #iout 
        sort = odd[:,3][odd[:,0].argsort()]
    elif idxPrun==1: #ideg
        sort = odd[:,3][odd[:,1].argsort()]
    elif idxPrun==2: # random
        sort = odd[:,3][odd[:,2].argsort()]
    elif idxPrun==3: # ddeg
        sort = odd[:,3][odd[:,1].argsort()]
        sort = sort[::-1]
    elif idxPrun==4: # dout
        sort = odd[:,3][odd[:,0].argsort()]
        sort = sort[::-1]
    else:
        print( 'index of pruning exceeds availability')
        return 0
    sort = sort.astype(int)
    
    isort = sort[np.isin(sort, new_Inrn)]
    esort = sort[np.isin(sort, new_Enrn)]
    
    return isort, esort
    
    
def sortSynapse(smtx, idxPrun):
    row_indices, col_indices = smtx.nonzero()
    if idxPrun==0:   # out
        sorted_indices = np.argsort(col_indices)
    elif idxPrun==1: # in 
        sorted_indices = np.argsort(row_indices)
    elif idxPrun==2: # rand
        sorted_indices = np.random.permutation(len(smtx.data))
    elif idxPrun==3: # ord 
        sorted_indices = np.arange(len(row_indices))
    elif idxPrun==4: # res
        sorted_indices = np.arange(len(row_indices))[::-1]
    else:
        print( 'index of pruning exceeds availability')

    # Re-arrange the row, column indices, and data according to the sorted order
    sorted_row_indices = row_indices[sorted_indices]
    sorted_col_indices = col_indices[sorted_indices]
    sorted_data = smtx.data[sorted_indices]

    # Create new sorted COO matrix
    # 
    return [sorted_row_indices, sorted_col_indices, sorted_data] 

def trim_neurons(pmtx, istage, isort, esort):
    nidel = int(del_frac * NI) * istage
    nedel = int(del_frac * NE) * istage

    to_delete = np.concatenate((isort[:nidel],esort[:nedel]))
    remaining = np.concatenate((isort[nidel:],esort[nedel:]))
    row_mask = np.isin(pmtx.row, to_delete)
    col_mask = np.isin(pmtx.col, to_delete)

    # Combine the masks to filter the data
    combined_mask = ~(row_mask | col_mask)

    # Get the new row and column indices
    filtered_rows = pmtx.row[combined_mask]
    filtered_cols = pmtx.col[combined_mask]

    # Get the new data based on the mask
    new_data = pmtx.data[combined_mask]


    # The following is reindexing neurons, making Inh neuons preceding Exc neurons, so that identifying Inh and Exc neuorns is straight forward.  

    Nnew = len(remaining)

    rmask = remaining.argsort()
    sorted_old_labels = remaining[rmask]
    new_labels = np.arange(Nnew)[rmask]


    relabeled_rows = new_labels[np.searchsorted(sorted_old_labels, filtered_rows)]
    relabeled_cols = new_labels[np.searchsorted(sorted_old_labels, filtered_cols)]

    return coo_matrix((new_data, (relabeled_rows, relabeled_cols)),shape=(Nnew, Nnew))
    
    
    
def deleteNodeOrLink(cmtx, idtyp, idxprun, istage, permm=[]):
    '''
    - cmtx is coo_matrix where the edges are sorted according to a desired synorder (in, out, rand, res or ord)
    - idtyp is 1 for neuronal death or 0 for synaptic death
    - istage is the stage of pruning (normally 10 stages where each stage removes 10% of links or nodes from the network). if istage=0, no pruning; istage =1, 10 percent pruning; istage 2 20 percent pruning
    - esort and isort are the sorted order of exc nodes or inh nodes according to a desired node order (iout, ideg, rand, ddeg, dout).
    - del_frac is by default 0.1, to cut 10% of the nodes or links in the degeneration process.

    The algorithm returns a coo matrix (which is sparse matrix) after the desired degneration
    '''
    if not len(permm):
        permm = np.arange(cmtx.shape[0])
    if idtyp: # 1 for neuronal death
        isort, esort = sortNeurons(cmtx, idxprun, permm)
        ncmtx = trim_neurons(cmtx, istage, isort, esort)
        
    else: # 0 for synaptic death
        sorted_row_indices, sorted_col_indices, sorted_data = sortSynapse(cmtx, idxprun)
        
        # ncutt is the number to be removed
        ncutt = istage*ncutt_default
        
        ncmtx = coo_matrix((sorted_data[ncutt:], (sorted_row_indices[ncutt:], sorted_col_indices[ncutt:])), shape=cmtx.shape)
        
    return ncmtx
    
    
    
    
   

# def effNdeg(wmtx, nwE, nwI,g):
#     mtx = (wmat!=0).astype(int)
#     N = len(mtx)
#     emtx = adjmtx[nwE,:]
#     imtx = adjmtx[nwI,:]
#     indE = np.sum(emtx, 0)# exc in-degree to each neuron
#     indI = np.sum(imtx, 0)
#
#     indI = -g*indI
#     efdeg = indE+indI#np.concatenate((indE, indI))
#     mnef = np.mean(efdeg)
#     sdef = np.std(efdeg)
#
#     indeg = np.sum(mtx, 0)
#     oudeg = np.sum(mtx, 1)
#     dggg = indeg+oudeg
#
#     mndeg = np.mean(dggg)
#     sddeg = np.std(indeg)
#
#
#
#     shh = outdg*(outdg-1)/2.
#     shE = shh[nwE]/N
#     mnsh = np.mean(shE)
#     pdens = np.sum(mtx)/N/N
#
#
#     return [mnef, sdef, mndeg, sddeg, mnsh, pdens, g]

   
# def killingNeurons(cmtx, indices_to_delete):
#     remaining_indices = np.delete(np.arange(cmtx.shape[0]), indices_to_delete)
#     Nnew = len(remaining_indices)
#
#     row_mask = np.isin(cmtx.row, indices_to_delete)
#     col_mask = np.isin(cmtx.col, indices_to_delete)
#
#     combined_mask = ~(row_mask | col_mask)
#
#     new_rows = np.searchsorted(remaining_indices, cmtx.row[combined_mask])
#     new_cols = np.searchsorted(remaining_indices, cmtx.col[combined_mask])
#
#     new_data = cmtx.data[combined_mask]
#
#     trimmed_sparse_matrix = coo_matrix((new_data, (new_rows, new_cols)),shape=(Nnew, Nnew))
#
#     return trimmed_sparse_matrix







