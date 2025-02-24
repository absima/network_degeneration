import numpy as np
from scipy.sparse import coo_matrix, load_npz, save_npz

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from funcDegeneration import *
from parameters import *
    

       
def trim_neurons2(trim_params):

    '''
    - cp_index is the index to load the corresponding connectivity coo_matrix as well as permuter.
        - cp_index=0 corresponds to the original data, no relabeling, permuter is also simply a range
    - idxprun is the type of nodal attack
    - istage is the stage of attack ... in our scenario, istage*10 percent removal

    output: trimmed connectivity data where neurons are relabeled yet again to put inh neurons at the beginning. This additional ordering step in this algo is necessary when we want to track inh and exc populations separately.
    '''
    wmatrix, cp_index, idxprun, istage = trim_params 
    
    isort, esort = sortNeurons(cp_index, idxprun)


    nidel = int(del_frac * NI) * istage
    nedel = int(del_frac * NE) * istage

    to_delete = np.concatenate((isort[:nidel],esort[:nedel]))
    remaining = np.concatenate((isort[nidel:],esort[nedel:]))

    wmatrix = wmatrix.toarray()
    wmatrix = wmatrix[np.ix_(remaining, remaining)]

    return coo_matrix(wmatrix)

def synapseSorter2(smtx, idxprun):
    # string_id = str(cp_index).zfill(2)
    # smtx = load_npz('%s/sparse_relabeld_and_ordered_%s.npz'%(dirr, string_id))
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
        print( 'index of pruning exceeds availability')

    # Re-arrange the row, column indices, and data according to the sorted order
    sorted_row_indices = row_indices[sorted_indices]
    sorted_col_indices = col_indices[sorted_indices]
    sorted_data = smtx.data[sorted_indices]

    # Create new sorted COO matrix
    # 
    return coo_matrix((sorted_data, (sorted_row_indices, sorted_col_indices)), shape=(N0,N0))
    

       
        
def trim_synapses2(trim_params):
    matrix, cp_index, idxprun, istage = trim_params 
    matrix = synapseSorter2(matrix.copy(), idxprun)
    
    ncutt = int(istage*del_frac*len(matrix.data))
    matrix = coo_matrix((matrix.data[ncutt:], (matrix.row[ncutt:], matrix.col[ncutt:])), shape=matrix.shape)
    
    permuter = np.load('%s/node_permutations_3611x10.npy'%dirr)[cp_index]
    return relabelingNeurons(matrix, perm=permuter.argsort()) 
    



        
def trimming2(idtyp, trim_params):
    if idtyp:
        return trim_neurons2(trim_params)
    else:
        return trim_synapses2(trim_params)
    
        

    





boundaries = [-610, -10, 10, 250, 350]  # For example: data values in the range [0, 1), [1, 2), ...
cmap = ListedColormap(['#339BFF', '#ffffff', '#FF3333', '#900C3F'])
norm = BoundaryNorm(boundaries, cmap.N)



demoTypes = [
    ['parent relabeled 1', 'synaptic loss 1', 'neuronal loss 1'], 
    ['parent relabeled 2', 'synaptic loss 2', 'neuronal loss 2'],
]


idxpr = 2 # 2 is for random deletion
istage = 5 # 5 is for 5*10 percent of the original to delete  


plt.close('all')
fig, axes = plt.subplots(2,3, figsize=(12,7)) 

for idxnet, cp_index in enumerate([1, 4]):    
    amtx = load_npz('%s/sparse_relabeld_and_ordered_%s.npz'%(dirr, str(cp_index).zfill(2)))
    permx = np.load('%s/node_permutations_3611x10.npy'%dirr)[cp_index]
    wmtx = weightedFromAdjacency(amtx, NI, permx)
    # prmmm = np.load('%s/node_permutations_3611x10.npy'%dirr)[cp_index]
    # pwmtx = relabelingNeurons(wmtx, perm=prmmm.argsort())
    axes[idxnet, 0].matshow(wmtx.toarray(), cmap = cmap, norm=norm) 
    for idtyp in range(2):
        trim_params = [wmtx, cp_index, idxpr, istage] 
        tmtx = trimming2(idtyp, trim_params)
        axes[idxnet, idtyp+1].matshow(tmtx.toarray(), cmap = cmap, norm=norm) 
plt.tight_layout()
plt.show()




# fig, axes = plt.subplots(2,3, figsize=(12,7))
#
# for idxnet, cp_index in enumerate([1, 4]):
#     amtx = load_npz('%s/sparse_relabeld_and_ordered_%s.npz'%(dirr, str(cp_index).zfill(2)))
#     permx = np.load('%s/node_permutations_3611x10.npy'%dirr)[cp_index]
#     wmtx = weightedFromAdjacency(amtx, NI, permx)
#     axes[idxnet, 0].matshow(wmtx.toarray(), cmap = cmap, norm=norm)
#     for idtyp in range(2):
#         dtparams = [idtyp, cp_index, idxpr, istage]
#         tmtx0 = trimming(dtparams)
#         newNI = NI-idtyp*int(del_frac*istage*NI)
#         newNE = NE-idtyp*int(del_frac*istage*NE)
#         tmtx = weightedFromAdjacency(tmtx0, newNI, [])
#         print(newNI+newNE, tmtx.shape[0])
#         axes[idxnet, idtyp+1].matshow(tmtx.toarray(), cmap = cmap, norm=norm)
# plt.tight_layout()
# plt.show()