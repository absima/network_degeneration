from scipy.sparse import coo_matrix, load_npz
from parameters import *


from scipy.sparse import coo_matrix, load_npz
from parameters import *

def loadParentAndPermuter(netname, index, ndir=None, pdir=None):
    '''
    - netname: name category of a network (eg. er, sw)
    - index: the realization/trial index
    - ndir: the directory of stored parent networks 
    - pdir: the directory of stored array of ,currently10 permutations (size=10xN)
    
    output: returns the sparse matrix and the corresponding permuter
    '''
    mfold = mfolds[netname in ['er', 'sw', 'sf']] 
    netfldr = pfold+mfold+netfold
    prmfldr = pfold+mfold+permfold
    
    if ndir is None:
        ndir = netfldr
    if pdir is None:
        pdir = prmfldr
    if netname=='emp':
        netstring = 'relabeld_and_ordered'
    elif netname in ['er', 'sw', 'sf']:
        netstring = 'doubleRelabeled_ordered'    
    smtx = load_npz('%s/%s_%d_%s.npz'%(ndir, netname, index, netstring))
    permuter = np.load('%s/node_permutations_3611x10.npy'%pdir)[index]
    data = [smtx, permuter]
    return data
    
def loadSpikeData(iparams):
    '''
    - netname: name category of a network (eg. er, sw)
    - cp_index: the realization/trial index
    
    output: returns the sparse matrix and the corresponding permuter
    '''
    netname, idtyp, cp_index, idxprun, istage, g = iparams
    mfold = mfolds[netname in ['er', 'sw', 'sf']] 
    spkdir = pfold+mfold+spkfold
    
    newNI = NI-idtyp*int(del_frac*istage*NI)
    newNE = NE-idtyp*int(del_frac*istage*NE)
    data = np.load('%s/spikeData_%s_%d_%d_%d_%d_%d.npz'%(spkdir, netname, idtyp, cp_index, idxprun, istage, int(g)))['data']
    return [data, newNI, newNE]

       
def degreeFromSparceMatrix(cmtx):
    '''
    - coomtx is the sparse coo_matrix as input
    - it retuns the outdegrees and degrees of the nodes in the network
    ''' 
    row_degrees = np.bincount(cmtx.row, minlength=cmtx.shape[0])
    col_degrees = np.bincount(cmtx.col, minlength=cmtx.shape[1])
    degrees = row_degrees + col_degrees
    return col_degrees, degrees

def weightedFromAdjacency(matrix, how_many_Ineurons, weight=None, orderIE=None):
    '''
    - matrix: a sparse binary matrix for simple adjacency
    - how_many_Ineurons: the number of Inh neurons out of all
    - orderIE: is the list of neurons, where the first how_many_Ineurons are Inh neurons, followed by ExcExc neurons. If it is not provided, Inh neurons are simply range(how_many_neurons). 
    - output: converts the binary connectivity to weighted one, based on synaptic strength.  
    '''
    if weight is None:
        weight = np.array([wii, wie, wei, wee])
    if orderIE is None:
        orderIE = np.arange(matrix.shape[0])
    
    nrnI = orderIE[:how_many_Ineurons]
    nrnE = orderIE[how_many_Ineurons:]
    
    j_i2i, j_i2e, j_e2i, j_e2e = weight
    
    matrix = matrix.toarray()
    matrix[np.ix_(nrnI, nrnI)] *= j_i2i
    matrix[np.ix_(nrnE, nrnI)] *= j_i2e
    matrix[np.ix_(nrnI, nrnE)] *= j_e2i
    matrix[np.ix_(nrnE, nrnE)] *= j_e2e
    
    return coo_matrix(matrix)
    
    
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
        

       
def synapseSorter(synparams):
    '''
    - cp_index: it can be an integer index or a list. index case is used to load data when we have saved permuted arrays with the permuters. list case is when we permute the arry online with a permuter and in which case [permuted_array, permuter] is the list form to pass. 
    - idxPrun: is the index of a neuronal death
    output: the sparse matrix where edges are sorted according to a pruning strategy
    '''
    netname, cp_index, idxprun, istage = synparams
    if isinstance(cp_index, (int, np.integer)):
        smtx = loadParentAndPermuter(netname, cp_index)[0]
    elif isinstance(cp_index, list):
        smtx = cp_index[0]
    else:
        raise ValueError(f"Invalid parameter: {cp_index}. Please provide a valid value.")
    if istage==0:
        return smtx
    row_indices, col_indices = smtx.nonzero()
    if idxprun==0:   # out
        sorted_indices = np.argsort(col_indices)
    elif idxprun==1: # in 
        sorted_indices = np.argsort(row_indices)
    elif idxprun==2: # rand
        sorted_indices = np.random.permutation(len(smtx.data))
    elif idxprun==3: # ord 
        sorted_indices = np.arange(len(row_indices))
    elif idxprun==4: # res
        sorted_indices = np.arange(len(row_indices))[::-1]
    else:
        raise ValueError(f"Invalid pruning index: {idxprun}. Please provide a valid value.")

    # Re-arrange the row, column indices, and data according to the sorted order
    sorted_row_indices = row_indices[sorted_indices]
    sorted_col_indices = col_indices[sorted_indices]
    sorted_data = smtx.data[sorted_indices]

    # Create new sorted COO matrix
    # 
    return coo_matrix((sorted_data, (sorted_row_indices, sorted_col_indices)), shape=(N0,N0))
    
       
def sortNeurons(nroparams):
    '''
    - cp_index: it can be an integer index or a list. index case is used to load data when we have saved permuted arrays with the permuters. list case is when we permute the arry online with a permuter and in which case [permuted_array, permuter] is the list form to pass.  
    - idxPrun: is the index of a neuronal death: 
    output: sorted I neurons and E neurons based on a degenerative strategy
    '''
    netname, cp_index, idxprun, istage = nroparams
    
    if isinstance(cp_index, (int, np.integer)):
        cmtx, permuter = loadParentAndPermuter(netname, cp_index)
    elif isinstance(cp_index, list):
        cmtx, permuter = cp_index
    else:
        raise ValueError(f"Invalid parameter: {cp_index}. Please provide a valid value.")
        
    new_Inrn = permuter[:NI]
    new_Enrn = permuter[NI:]
    
    if istage==0:# no pruning
        return new_Inrn, new_Enrn
        
    outdegree, degree = degreeFromSparceMatrix(cmtx)
    
    prrm = np.arange(N0) # for random nodal attack
    np.random.shuffle(prrm)
    
    odd = np.column_stack((outdegree, degree, prrm, np.arange(N0)))    
    if  idxprun==0:  #iout 
        sort = odd[:,3][odd[:,0].argsort()]
    elif idxprun==1: #ideg
        sort = odd[:,3][odd[:,1].argsort()]
    elif idxprun==2: # random
        sort = odd[:,3][odd[:,2].argsort()]
    elif idxprun==3: # ddeg
        sort = odd[:,3][odd[:,1].argsort()]
        sort = sort[::-1]
    elif idxprun==4: # dout
        sort = odd[:,3][odd[:,0].argsort()]
        sort = sort[::-1]
    else:
        raise ValueError(f"Invalid pruning index: {idxprun}. Please provide a valid value.")
    sort = sort.astype(int)
    
    isort = sort[np.isin(sort, new_Inrn)]
    esort = sort[np.isin(sort, new_Enrn)]
    
    return isort, esort

def trim_synapses(trim_params):
    '''
    - cp_index: it can be an integer index or a list. index case is used to load data when we have saved permuted arrays with the permuters. list case is when we permute the arry online with a permuter and in which case [permuted_array, permuter] is the list form to pass. 
    - idxprun: the type of synaptic degenerative strategy
    - istage: the stage of degeneration 
    output: prunned network with their original labels, INH indices preceding EXC indices 
    '''
    netname, cp_index, idxprun, istage = trim_params
    if isinstance(cp_index, (int, np.integer)):
        permuter = loadParentAndPermuter(netname, cp_index)[1] 
    elif isinstance(cp_index, list):
        permuter = cp_index[1]
    else:
        raise ValueError(f"Invalid parameter: {cp_index}. Please provide a valid value.")
    
    matrix = synapseSorter(trim_params)
    ncutt = int(istage*del_frac*len(matrix.data))
    matrix = coo_matrix((matrix.data[ncutt:], (matrix.row[ncutt:], matrix.col[ncutt:])), shape=matrix.shape)
    
    return relabelingNeurons(matrix, perm=permuter.argsort())
    
            
def trim_neurons(trim_params):
    
    ''' 
    - cp_index: it can be an integer index or a list. index case is used to load data when we have saved permuted arrays with the permuters. list case is when we permute the arry online with a permuter and in which case [permuted_array, permuter] is the list form to pass. 
    - idxprun is the type of nodal attack
    - istage is the stage of attack ... in our scenario, istage*10 percent removal
    
    output: trimmed connectivity data where neurons are relabeled yet again to put inh neurons at the beginning. This additional ordering step in this algo is necessary when we want to track inh and exc populations separately.  
    '''
    netname, cp_index, idxprun, istage = trim_params
    if isinstance(cp_index, (int, np.integer)):
        pmtx = loadParentAndPermuter(netname, cp_index)[0]
    elif isinstance(cp_index, list):
        pmtx = cp_index[0]
    else:
        raise ValueError(f"Invalid parameter: {cp_index}. Please provide a valid value.")
    isort, esort = sortNeurons(trim_params)
    
    
    nidel = int(del_frac * NI) * istage
    nedel = int(del_frac * NE) * istage

    to_delete = np.concatenate((isort[:nidel],esort[:nedel]))
    remaining = np.concatenate((isort[nidel:],esort[nedel:])) # we make sure Inh are indexed first in order
    
    
    pmtx = pmtx.toarray()
    pmtx = pmtx[np.ix_(remaining, remaining)]
    
    return coo_matrix(pmtx)
    
        
    
def trimming(params):
    '''
    -- params has the form 
        [netname, idtyp, cp_index, idxprun, istage] 
    
    - netfldr: the directory of stored parent networks 
    - prmflder: the directory of stored array of ,currently10 permutations (size=10xN)  
    - netname: name category of a network (eg. er, sw)
    - idtyp: is degeneration type as in 0=synaptic and 1=neuronal
    - cp_index: it can be an integer index or a list. index case is used to load data when we have saved permuted arrays with the permuters. list case is when we permute the arry online with a permuter and in which case [permuted_array, permuter] is the list form to pass. 
    - idxprun is the type of degeneration scheme
    - istage is the stage of pruning (normally 10 stages where each stage removes 10% of links or nodes from the network). if istage=0, no pruning; istage =1, 10 percent pruning; istage 2 20 percent pruning
    output: a coo matrix (which is sparse matrix) after the desired degneration and resorted so that inh nodes have indices less than exc nodes, for easier access.
    '''  
    netname, idtyp, cp_index, idxprun, istage = params
    trim_params = [netname, cp_index, idxprun, istage]
    
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



# def trimmedNet(iparams, flag=None):
#     idtyp, cp_index, idxprun, istage = iparams
#     newNI = NI-idtyp*int(del_frac*istage*NI)
#     newNE = NE-idtyp*int(del_frac*istage*NE)
#
#     if flag=='unweightedAdjacency':
#         data = trimming(iparams)
#         data = [data, newNI, newNE]
#     elif flag=='weightedAdjacency':
#         if weight is None:
#             weight = np.array([wii, wie, wei, wee])
#         data = trimming(iparams)
#         data = weightedFromAdjacency(data, newNI, weight=weight)
#         data = [data, newNI, newNE]
#     else:
#         raise ValueError(f"Invalid flag: {flag}. Please provide a valid value.")
#
#     return data
    
# def loadData(iparams, flag='indexPerumtedParent', weight=None):
#     # if type(iparams)==int:
#     if isinstance(iparams, (int, np.integer)):
#         cp_index = iparams
#     elif isinstance(iparams, list) and len(iparams)==4:
#         idtyp, cp_index, idxprun, istage = iparams
#         newNI = NI-idtyp*int(del_frac*istage*NI)
#         newNE = NE-idtyp*int(del_frac*istage*NE)
#     else:
#         raise ValueError(f"Invalid parameter: {iparams}. Please provide a valid value.")
#
#     if flag=='indexPerumtedParent':
#         string_id = str(cp_index).zfill(2)
#         smtx = load_npz('%s/sparse_relabeld_and_ordered_%s.npz'%(indir, string_id))
#         permuter = np.load('%s/node_permutations_3611x10.npy'%indir)[cp_index]
#         data = [smtx, permuter]
#     elif flag=='unweightedAdjacency':
#         data = trimming(iparams)
#         data = [data, newNI, newNE]
#     elif flag=='weightedAdjacency':
#         if weight is None:
#             weight = np.array([wii, wie, wei, wee])
#         data = trimming(iparams)
#         data = weightedFromAdjacency(data, newNI, weight=weight)
#         data = [data, newNI, newNE]
#     elif flag=='spikes':
#         data = np.load('%s/spikeData_%d_%d_%d_%d.npz'%(outdir, idtyp, cp_index, idxprun, istage))['data']
#         data = [data, newNI, newNE]
#     else:
#         raise ValueError(f"Invalid flag: {flag}. Please provide a valid value.")
#    return data



