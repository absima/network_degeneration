import pylab as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from funcDegeneration import *   



np.random.seed(23098)

permu = np.random.permutation(N0)
# permu = np.arange(N0)
pmtx = relabelingNeurons(cmtx, perm=permu)

idtyp = 1
idxprun = 2
istage = 5
del_frac = .1
# isort, esort = sortNeurons(pmtx, idxprun, permu)
# tmtx = trim_neurons(pmtx, isort, esort)

tmtx = deleteNodeOrLink(pmtx, idtyp, idxprun, istage, permu)



cc = cmtx.toarray()
pp = pmtx.toarray()
tt = tmtx.toarray()



cmap = ListedColormap(['#339BFF', '#ffffff', '#FF3333', '#900C3F'])
boundaries = [-61, -1, 1, 25, 35]  # For example: data values in the range [0, 1), [1, 2), ...
norm = BoundaryNorm(boundaries, cmap.N)

plt.close('all')
fig = plt.figure(figsize=(10,4))
az1 = fig.add_subplot(131)
az1.set_title('original', fontsize=13, fontweight='bold')
cav = az1.matshow(cc, cmap = cmap, norm=norm)
# cbar = plt.colorbar(cav, ticks = [-31.,    0.,  13.,  30.])
# cbar.set_ticklabels([-60, 0, 20, 30])

az2 = fig.add_subplot(132)
az2.set_title('relabeled', fontsize=13, fontweight='bold')
cav = az2.matshow(pp, cmap = cmap, norm=norm)
# cbar = plt.colorbar(cav, ticks = [-31.,    0.,  13.,  30.])
# cbar.set_ticklabels([-60, 0, 20, 30])
az3 = fig.add_subplot(133)
# az.set_title('trimmed', fontsize=13, fontweight='bold')
# cav = az.matshow(qq, cmap = cmap, norm=norm)
#
# az = fig.add_subplot(144)
az3.set_title('trimmed (%d%%)'%(10*istage), fontsize=13, fontweight='bold')
cav = az3.matshow(tt, cmap = cmap, norm=norm)
# cbar = plt.colorbar(cav, ticks = [-31.,    0.,  13.,  30.])
# cbar.set_ticklabels([-60, 0, 20, 30])


plt.tight_layout()
plt.show()

    
    




    
# idtyp = 1
# idxprun = 0
# istage = 0
# delfrac = 0.1
#
#
# cmtx = ordspmtx.copy()
# # newmx = deletingNroSyn(cmtx, idtyp, idxprun, istage, frac)
#
# for istage in range(10):
#     newmx = deletingNroSyn(cmtx, idtyp, idxprun, istage, delfrac)
#     newN = newmx.shape[0]
#     newNI = NI-int(istage*delfrac*idtyp*NI) #idtyp is just to reduce only during neuroDeath
#     newNE = NE-int(istage*delfrac*idtyp*NI)
#     print('')
#     print(istage)
#     print('lee', len(newmx.data))
#     print('size ', newmx.shape[0], ' newNI,newNE:', (newNI, newNE))
#
#
#





# deletes edges or nodes at istage and retuns connMatrix 
# def deletingNroSyn(cmtx, idtyp, istage, esort, isort, frac):
#     '''
#     - cmtx is coo_matrix where the edges are sorted according to a desired synorder (in, out, rand, res or ord)
#     - idtyp is 1 for neuronal death or 0 for synaptic death
#     - istage is the stage of pruning (normally 10 stages where each stage removes 10% of links or nodes from the network). if istage=0, no pruning; istage =1, 10 percent pruning; istage 2 20 percent pruning
#     - esort and isort are the sorted order of exc nodes or inh nodes according to a desired node order (iout, ideg, rand, ddeg, dout).
#     - del_frac is by default 0.1, to cut 10% of the nodes or links in the degeneration process.
#
#     The algorithm returns a coo matrix (which is sparse matrix) after the desired degneration
#     '''
#
#     if idtyp:
#         nidel = int(del_frac * NI) * istage
#         nedel = int(del_frac * NE) * istage
#
#         ### first sort the idels and edels
#         idel = isort[:nidel]
#         edel = esort[:nedel]
#
#         eidel = np.concatenate((idel,edel))
#
#         mtx = np.delete(mtx, eidel, axis=0)
#         mtx = np.delete(mtx, eidel, axis=1)
#
#     else:
#         # ncutt is the number to be removed
#         ncutt = istage*ncutt_default
#         # edgOut = edgIn[ncutt:]
#         torem = edgIn[:ncutt]
#         mtx[torem[:,0], torem[:,1]] = 0
#     return mtx
#
#
#

    
       




# synPruns = ['ord', 'res', 'in', 'out', 'rand']
#
# for iptyp, ptyp in enumerate(synPruns):
#     pmtx = sortSynDegree(nsprse, inorout= ptyp)
#     edg = np.column_stack(pmtx.nonzero())
#     print('')
#     print(ptyp)
#     print(edg[:10])
    
    


# N0     = 98#10000 #<<<<<<<<
# NI, NE = [ 17, 81]
# # NX = 311 # external
#
# np.random.seed(27879)
# mtx = np.random.binomial(1, .1, (N0,N0))
# np.fill_diagonal(mtx, 0)
# smtx = csr_matrix(mtx)
# smtx = smtx.tocoo()
# # scipy.io.savemat('smatrix98.mat', {'smtx': smtx})
# smtx = scipy.io.loadmat('smatrix98.mat')['smtx'] ## coo matrix

# np.random.seed(27879)
#
# N0     = 20#10000 #<<<<<<<<
# NI, NE = [ 4, 16]
# # NX = 311 # external
#
# mtx = np.random.binomial(1, .1, (N0,N0))
# np.fill_diagonal(mtx, 0)
# smtx = csr_matrix(mtx)
# smtx = smtx.tocoo()
# sr,tg = smtx.nonzero()
# smtx = coo_matrix((np.arange(1,len(sr)+1), (sr,tg)), shape=smtx.shape)
# # scipy.io.savemat('smatrix98.mat', {'smtx': smtx})
#
# # src, tgt = smtx.nonzero()
# permu = np.arange(len(smtx.data))
# np.random.shuffle(permu)
# qmtx = permute(smtx.copy(), permu)
#
# shuf = reindexingNeurons(smtx)
#
# ousort = sortSynDegree(smtx, 'out')
# insort = sortSynDegree(smtx, 'in')
# rdsort = sortSynDegree(smtx, 'rand')





# #### ordering
# # def orderingNroSyn(scooMtx, idtyp, idxPrun):
# idtyp = 1
#
# cmtx = smtx.tocoo()
#
# if idtyp:
#     # src, tgt = edgIn.T
#     # mtx = np.zeros((N0,N0), dtype=int)
#     # mtx[src,tgt] = 1
#
#     # otdg = np.sum(mtx,1) # rows are considered sources
#     # aldg = np.sum(mtx,0) + np.sum(mtx,1)
#     otdeg, aldg = degreeFromSparceMatrix(cmtx)
#     prrm = np.arange(N0)
#     np.random.shuffle(prrm)
#
#     odd = np.column_stack((otdg, aldg, prrm, nrns))
#     if  idxPrun==0:
#         sort = odd[:,3][odd[:,0].argsort()]
#     elif idxPrun==1:
#         sort = odd[:,3][odd[:,1].argsort()]
#     elif idxPrun==2:
#         sort = odd[:,3][odd[:,2].argsort()]
#     elif idxPrun==3:
#         sort = odd[:,3][odd[:,1].argsort()]
#         sort = sort[::-1]
#     elif idxPrun==4:
#         sort = odd[:,3][odd[:,0].argsort()]
#         sort = sort[::-1]
#     else:
#         print( 'specify type')
#         return 0
#     sort = sort.astype(int)
#     isort = sort[sort<NI]
#     esort = sort[sort>=NI]
#     # # idxE = sort[sort%5!=0]
#     # # idxI = sort[sort%5==0]
#     # isort = sort[nrnI0].astype(int)
#     # esort = sort[nrnE0].astype(int)
#     edgOut = edgIn.copy()
# else:
#     ptyp = synPruns[idxPrun]
#     if ptyp == 'ord':## ordered
#         edgOut = edgIn.copy()
#     elif ptyp == 'res':
#         edgOut = edgIn[::-1]
#     elif ptyp == 'out':
#         src,tgt = edgIn.T
#         edgOut = edgIn[src.argsort()]
#     elif ptyp=='in':## out
#         src,tgt = edgIn.T
#         edgOut = edgIn[tgt.argsort()]
#     elif ptyp == 'rnd':## random
#         prrr = np.arange(len(edgIn))
#         np.random.shuffle(prrr)
#         edgOut = edgIn[prrr]
#
#     else:
#         print( 'specify type')
#         return 0
#     esort = nrnE0
#     isort = nrnI0
# return edgOut, esort, isort
# #







