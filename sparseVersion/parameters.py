import numpy as np
from scipy.sparse import coo_matrix

# ordspmtx = scipy.sparse.load_npz('jmatSparse_10percent_ordered.npz')
# permuters = pp = np.load('node_permutations_3611x10.npy')
# #
# N0     = 3611#10000 #<<<<<<<<
# NI, NE = [ 680, 2931]
# # NX = 311 # external
#
# # permutations = np.array([np.random.permutation(N0) for _ in range(nperm)])
# # np.save('node_permutations_3611x10.npy', permutations)
#
# lee = len(ordspmtx.data)
# rho = lee/N0/N0
# # k               = int(0.1*N0)
#
# ncutt_default   = int(0.1*lee)
# del_frac = 0.1
# nrns = np.arange(N0)
#
#
# nrnE0 = nrns[NI:]
# nrnI0 = nrns[:NI]
#
# synPruns = ['out', 'in', 'rnd', 'ord', 'res'] ###
# nroPruns = ['incOut', 'incDeg', 'rnd', 'decDeg', 'decOut']
#
# glist = [5., 7.]
# Degens = [synPruns, nroPruns]
# # Nets     = ['er', 'sw', 'ba']
# Nets = ['exp']
#
# nglist = len(glist)
# nPruns = len(nroPruns)
# nDegens = len(Degens)
# nNets = len(Nets)
# nBizat = 10
#
# tsec     = 6.
# # window     = 50.
# start     = 1000.#10000. #wuptime
#
# simtime = tsec*1000.
# nettime = simtime-start
# scaler  = nettime/1000.
#
# J          = .1#0.2        # postsynaptic amplitude in mV
#
# eta     = 2.#2.5
#
# delay = 1.5
#
# mij_e = .5
# J_bg = 5.
# p_rate = 15000.




N0 = 100
NI = 20
NE = 80

nrns = np.arange(N0)


nrnE0 = nrns[NI:]
nrnI0 = nrns[:NI]



jii = -60
jie = -60
jei = 30
jee = 20


mtx = np.random.binomial(1, .2, (N0,N0))
np.fill_diagonal(mtx, 0)
mtx[:,:NI]*=jii
mtx[:NI,NI:]*=jei
mtx[NI:,NI:]*=jee
cmtx = coo_matrix(mtx)

del_frac = 0.1
