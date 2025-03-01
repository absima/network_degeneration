import numpy as np
from funcGeneration import *

nettype = ['er', 'sw', 'sf']
### directed nets
N = 3611
density = 0.11
numEdges = 1404419#int(density*N*(N-1))
# k = 2*density*(N-1)



#### creating a network
iname = 1 #'sw'
mtx = getNet(N,numEdges,iname)

### reindexing / relabeling
# save as original
mtx = relabelingNeurons(mtx) # we don't need to store perm

# save as relabeled
# here we can use stored permuters for exp data, to mix I and E blocks up
mtx = relabelingNeurons(mtx, perm)


### ordering 


### 

# import matplotlib.pyplot as plt
# plt.close('all')
# fig = plt.figure()
# for i,j in enumerate(['er', 'sw', 'sf']):
#     mtx = getNet(N,numEdges,j)
#     print(len(mtx.data))
#     ax = fig. add_subplot(1,3,i+1)
#     ax.matshow(mtx.toarray())
#
# plt.tight_layout()
# plt.show()