import numpy as np
import scipy.io 

from reformzen import *


sprse = scipy.io.loadmat('jmatSparse_10percent.mat')['csr']
edgez = np.column_stack((sprse.nonzero()))[:,::-1]

g0 = zenGraph(edgez)
# MM = zen.maximum_matching(g0)
# print(len(MM))


dd = orderMMEdge(g0)		







