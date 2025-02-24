from scipy.sparse import coo_matrix, load_npz
from parameters import *


def degreeFromSparceMatrix(cmtx):
    '''
    - coomtx is the sparse coo_matrix as input
    - it retuns the outdegrees and degrees of the nodes in the network
    ''' 
    row_degrees = np.bincount(cmtx.row, minlength=cmtx.shape[0])
    col_degrees = np.bincount(cmtx.col, minlength=cmtx.shape[1])
    degrees = row_degrees + col_degrees
    return col_degrees, degrees

def weightedFromAdjacency(matrix, how_many_Ineurons, orderIE=[]):
    '''
    - matrix: a sparse binary matrix for simple adjacency
    - how_many_Ineurons: the number of Inh neurons out of all
    - orderIE: is the list of neurons, where the first how_many_Ineurons are Inh neurons, followed by ExcExc neurons. If it is not provided, Inh neurons are simply range(how_many_neurons). 
    - output: converts the binary connectivity to weighted one, based on synaptic strength.  
    '''
    if not len(orderIE):
        orderIE = np.arange(matrix.shape[0])
    
    nrnI = orderIE[:how_many_Ineurons]
    nrnE = orderIE[how_many_Ineurons:]
    

    matrix = matrix.toarray()
    matrix[np.ix_(nrnE, nrnE)] *= wee
    matrix[np.ix_(nrnI, nrnE)] *= wei
    matrix[np.ix_(nrnI, nrnI)] *= wii
    matrix[np.ix_(nrnE, nrnI)] *= wie
    
    return coo_matrix(matrix)
    
    
def relabelingNeurons(cmtx, perm=[]):
    '''
    relabeling the sparse matrix. 
    i neuron in input matrix is perm[i] neuron in the output matrix, to avoid computational bias.
    '''
    if not len(perm):
        perm = np.random.permutation(cmtx.shape[0])
    row_indices, col_indices = cmtx.nonzero()
    new_row_indices = perm[row_indices]
    new_col_indices = perm[col_indices]

    return coo_matrix((cmtx.data, (new_row_indices, new_col_indices)), shape=cmtx.shape)
        

    

    
def synapseSorter(cp_index, idxPrun):
    '''
    - cp_index: the identifier to load the desired permuter and permuted array
    - idxPrun: is the index of a neuronal death
    output: the sparse matrix where edges are sorted according to a pruning strategy
    '''
    string_id = str(cp_index).zfill(2)
    smtx = load_npz('%s/sparse_relabeld_and_ordered_%s.npz'%(dirr, string_id))
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
    return coo_matrix((sorted_data, (sorted_row_indices, sorted_col_indices)), shape=(N0,N0))
    
       
def sortNeurons(cp_index, idxPrun):
    '''
    - cp_index: the identifier to load the desired permuter and permuted array
    - idxPrun: is the index of a neuronal death
    output: sorted I neurons and E neurons based on a degenerative strategy
    '''
    string_id = str(cp_index).zfill(2)
    cmtx = load_npz('%s/sparse_relabeld_and_ordered_%s.npz'%(dirr, string_id))
    permuter = np.arange(cmtx.shape[0])
    if cp_index: # if we are working with a reindexed version
        permuter = np.load('%s/node_permutations_3611x10.npy'%dirr)[cp_index]
    new_Inrn = permuter[:NI]
    new_Enrn = permuter[NI:]
    
    outdegree, degree = degreeFromSparceMatrix(cmtx)
    
    prrm = np.arange(N0) # for random nodal attack
    np.random.shuffle(prrm)
    
    odd = np.column_stack((outdegree, degree, prrm, np.arange(N0)))    
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

def trim_synapses(trim_params):
    '''
    - cp_index: the identifier to load the desired permuter and permuted array
    - idxprun: the type of synaptic degenerative strategy
    - istage: the stage of degeneration 
    output: prunned network with their original labels, INH indices preceding EXC indices 
    '''
    cp_index, idxprun, istage = trim_params
    matrix = synapseSorter(cp_index, idxprun)
    ncutt = int(istage*del_frac*len(matrix.data))
    matrix = coo_matrix((matrix.data[ncutt:], (matrix.row[ncutt:], matrix.col[ncutt:])), shape=matrix.shape)
    permuter = np.load('%s/node_permutations_3611x10.npy'%dirr)[cp_index]
    
    return relabelingNeurons(matrix, perm=permuter.argsort())
    
            
def trim_neurons(trim_params):
    
    ''' 
    - cp_index is the index to load the corresponding connectivity coo_matrix as well as permuter. 
        - cp_index=0 corresponds to the original data, no relabeling, permuter is also simply a range
    - idxprun is the type of nodal attack
    - istage is the stage of attack ... in our scenario, istage*10 percent removal
    
    output: trimmed connectivity data where neurons are relabeled yet again to put inh neurons at the beginning. This additional ordering step in this algo is necessary when we want to track inh and exc populations separately.  
    '''
    cp_index, idxprun, istage = trim_params
    isort, esort = sortNeurons(cp_index, idxprun)
    
    
    nidel = int(del_frac * NI) * istage
    nedel = int(del_frac * NE) * istage

    to_delete = np.concatenate((isort[:nidel],esort[:nedel]))
    remaining = np.concatenate((isort[nidel:],esort[nedel:])) # we make sure Inh are indexed first in order
    
    string_id = str(cp_index).zfill(2)
    pmtx = load_npz('%s/sparse_relabeld_and_ordered_%s.npz'%(dirr, string_id))
    pmtx = pmtx.toarray()
    pmtx = pmtx[np.ix_(remaining, remaining)]
    
    return coo_matrix(pmtx)
    
        
    
def trimming(type_and_trim_params):
    '''
    -- type_and_trim_params has of the form [idtyp, cp_index, idxprun, istage]. 
    - cp_index is the index to load the corresponding connectivity coo_matrix as well as permuter. 
        - cp_index=0 corresponds to the original data, no relabeling, permuter is also simply a range
    - idxprun is the type of degeneration scheme
    - istage is the stage of pruning (normally 10 stages where each stage removes 10% of links or nodes from the network). if istage=0, no pruning; istage =1, 10 percent pruning; istage 2 20 percent pruning
    output: a coo matrix (which is sparse matrix) after the desired degneration and resorted so that inh nodes have indices less than exc nodes, for easier access.
    '''  
    trim_params = type_and_trim_params[1:]
    idtyp = type_and_trim_params[0]
    if idtyp:
        return trim_neurons(trim_params)
    else:
        return trim_synapses(trim_params)
        

# def relabelingNeuronBlocks(cmtx, block1_permutation=[], block2_permutation=[]):
#     '''
#     This just relabels two blocks of nodes (I, E) independently with some random permutations.
#     '''
#     # if isspmatrix_coo(smtx):
# #         smtx = smtx.tocsr()
#     row_indices, col_indices = cmtx.nonzero()
#     data = smtx.data
#
#     # reindexing inhibitory and excitatory neurons separately
#     if  not (len(block1_permutation) * len(block2_permutation)):
#         block1_permutation = np.random.permutation(NI)
#         block2_permutation = np.random.permutation(NE)
#
#     # new indices
#     new_row_indices = row_indices.copy()
#     new_col_indices = col_indices.copy()
#
#     # shuffling rows and columns in for I population
#     new_row_indices[row_indices < NI] = block1_permutation[row_indices[row_indices < NI]]
#     new_col_indices[col_indices < NI] = block1_permutation[col_indices[col_indices < NI]]
#
#     # shuffling rows and columns in for E population
#     new_row_indices[row_indices >= NI] = NI + block2_permutation[row_indices[row_indices >= NI] - NI]
#     new_col_indices[col_indices >= NI] = NI + block2_permutation[col_indices[col_indices >= NI] - NI]
#
#     # creating the shuffled sparse matrix
#     shuffled_smtx = coo_matrix((data, (new_row_indices, new_col_indices)), shape=smtx.shape)
#
#     # # csr version, if preferred for efficiency
#     # shuffled_smtx_csr = shuffled_smtx.tocsr()
#
#     return shuffled_smtx






