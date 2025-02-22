import numpy as np
import scipy.io
from scipy.sparse import csr_matrix, coo_matrix, load_npz, save_npz


######### saving the csr and coo matrices
# sprse = scipy.io.loadmat('jmatSparse_10percent.mat')['csr']
#
# edz = np.column_stack(sprse.nonzero())
# dta = sprse.data
# #
# # mapp = {tuple(edz[i]):i for i in range(len(edz))}
# #
# ordedz = np.loadtxt('orderedMM_landau_10percent.txt').astype(int)
#
# colind, rowind = ordedz.T
# # # tuplist = list(zip(rowind, colind))
# #
# ndta = dta[np.argsort(np.lexsort((rowind, colind)))]
# nspmtx = coo_matrix((ndta, (rowind, colind)), shape=sprse.shape)
# #
# # # scipy.io.savemat('orderedSparse_Landau_10prc.mat', {
# # #     'rowind': rowind,
# # #     'colind': colind,
# # #     'ndta': ndta,
# # #     'shape': sprse.shape
# # # })
# # # # scipy.io.savemat('jmatSparse_10percent_ordered.mat', {'coo':nspmtx})
#
# # data = loadmat('orderedSparse_Landau_10prc.mat')
# # xmtx = coo_matrix((data['ndta'], (data['rowind'], data['colind'])), shape=tuple(data['shape'][0]))
# #
# #
# # osprse = scipy.io.loadmat('jmatSparse_10percent.mat')['csr']
# #
# # ddict = scipy.io.loadmat('orderedSparse_Landau_10prc.mat')
# # nsprse = coo_matrix((ddict['ndta'][0], (ddict['rowind'][0], ddict['colind'][0])), shape=tuple(ddict['shape'][0]))
#
# scipy.sparse.save_npz('jmatSparse_10percent.npz', sprse)
# scipy.sparse.save_npz('jmatSparse_10percent_ordered.npz', nspmtx)


#############
osprse = scipy.sparse.load_npz('jmatSparse_10percent.npz')
nsprse = scipy.sparse.load_npz('jmatSparse_10percent_ordered.npz')

a = osprse.toarray()
b = nsprse.toarray()

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(121)
ax.matshow(a, cmap='jet')

bx = fig.add_subplot(122)
bx.matshow(b, cmap='jet')

plt.tight_layout()
plt.show()


edg1 = np.column_stack(nsprse.nonzero())
edg2 = np.loadtxt('orderedMM_landau_10percent.txt').astype(int)

