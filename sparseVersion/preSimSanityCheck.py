import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm

from funcDegeneration import *
from parameters import *

'''
This is sort of verifying the two trimming algorithms (syn, neuro) and mainly correct weight assignment to the binary nets. While in synaptic pruning, the nodes retain their original labels, in nodal pruning, I block and E block are relabled separately but retain the same structure as the original in that I block have smaller indices (ordered first in the matrix) than the E blocks.  
'''

demoTypes = [
    ['parent (original)', 'parent permute A', 'child of A (synaptic-)', 'child of A (neuro-)'], 
    ['parent (original)', 'parent permute B', 'child of B (synaptic-)', 'child of B (neuro-)'],
]

np.random.seed(4866867)

idxprun = 2 # 2 is for random deletion
istage = 5 # 5 is for 5*10 percent of the original to delete  


boundaries = [-610, -10, 10, 250, 350]  # For example: data values in the range [0, 1), [1, 2), ...
# boundaries = [wii+0.1*wii, wii-0.1*wii, wee-0.1*wee, wei+0.1*wei ]
cmap = ListedColormap(['#339BFF', '#ffffff', '#FF3333', '#900C3F'])
norm = BoundaryNorm(boundaries, cmap.N)
plt.close('all')
fig, axes = plt.subplots(len(demoTypes),len(demoTypes[0]), figsize=(12,5)) 
for idx in range(2): 
    lister = []
    #### loading the original matrix; .... alternatively one can create own network
    lister.append(load_npz('../../nongit/connMatSparse_10percent_NI680_NE2931_ordered.npz'))
    permuter = np.random.permutation(N0)
    #### relabeling with a permuter
    lister.append(relabelingNeurons(lister[0], perm=permuter))
    
    #### link loss 
    lister.append(trimming([0, [lister[1], permuter], idxprun, istage]))
    #### node loss 
    lister.append(trimming([1, [lister[1], permuter], idxprun, istage]))
    
    #### weight assignment to sparse matrices
    numNI = [NI, NI, NI, NI-int(del_frac*istage*NI)]
    weight = [-606, -606, 303, 202]
    lister = [weightedFromAdjacency(lister[i], numNI[i], weight=weight) for i in range(len(lister))]
    
    #### extracting the weight matrices
    lister = [i.toarray() for i in lister]
    
    for imatrix, matrix in enumerate(lister):
        ax= axes[idx, imatrix]
        ax.matshow(matrix, cmap = cmap, norm=norm)
        ax.set_title(demoTypes[idx][imatrix], fontweight='bold', fontsize=12)
        ax.set_xticklabels([])

plt.suptitle('Degeneration, syn+neuro, randPrun, stage5', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

plt.savefig('Degeneration_synaptic_neuronal_random_stage5.pdf')
    




