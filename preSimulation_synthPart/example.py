import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm

from funcGeneration import *
# from parameters import *
N = 3611
nE = 1404419
iname =1
idx = 1
params = [iname, idx, N, nE]
a,b,c = generateRelabelSave(params)

perm = np.load('%s/node_permutations_3611x10.npy'%pdir)[idx]
d = relabelingNeurons(c, perm.argsort()) # this is b
d = relabelingNeurons(d, perm.argsort()) # this has to be a (verifying tthe code)

arr = [a,b,c,d]

plt.close('all')
fig = plt.figure()
for imtx, mtx in enumerate(arr):
    ax=fig.add_subplot(1,4,imtx+1)
    ax.matshow(mtx.toarray())
    ax.axis('off')
    
plt.tight_layout()
plt.show()



