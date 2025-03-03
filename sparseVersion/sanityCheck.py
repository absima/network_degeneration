import numpy as np
import matplotlib.pyplot as plt

from funcDegeneration import *
from parameters import *


'''
This is sort of verifying the overall steps, preSimulation(mainly trimming), simulation and postSimulation. The trimming algorithms (syn + neuro) and weight assignment to the binary nets are verified. While in synaptic pruning, the nodes retain their original labels, in nodal pruning, I block and E block are relabled separately but retain the same structure as the original in that I block have smaller indices (ordered first in the matrix) than the E blocks. In the end we inspect the spike data.  
'''


np.random.seed(4866867)

   
# ################## -1- ##################


# ########## 1. preSimulation ##########
# #### loading the relabeled and perm and restoring the actual ... followed by trimming
#
# def plotmat(ax, mtx, icol, irow, colname=None, rowname=None):
#     '''
#     to plot a matrix plot on (irow,icol) position
#     '''
#     ax.matshow(mtx, cmap=cmap, norm=norm)
#     ax.set_aspect('auto')
#
#     ax.locator_params(axis='x', nbins=4)
#     ax.locator_params(axis='y', nbins=4)
#     if icol:
#         ax.set_yticklabels([])
#     if not icol:
#         ax.set_ylabel('Node ID', fontsize=10, fontweight='bold')
#     if irow==0:
#         ax.set_title(colname, fontsize=12, fontweight='bold')
#     if irow==3:
#         ax.xaxis.set_ticks_position('bottom')
#         ax.set_xlabel('Node ID', color ='white', fontsize=10, fontweight='bold')
#         ax.set_xticklabels(ax.get_xticklabels(), color='white', rotation=45)
#     else:
#         ax.set_xticklabels([])
#
#     # if not icol:
#     #     ax.text(-(1.2)*len(mtx), len(mtx)/2, rowname, fontsize=12, fontweight='bold')
#     if icol==3:
#         ax.text((1.1)*len(mtx), len(mtx)/2, rowname, fontsize=12, fontweight='bold')
#     return
#    
#
# index = 0 # net variant
# istage = 5 # 5 is for 5*10 percent of the original to delete
# wgt_array = [-600, -606, 302, 203]
# wii, wie, wei, wee = wgt_array
# boundaries = np.array([wie-0.1, wii-0.1, -0.1, 0.1, wee+0.1, wei+0.1  ])
#
# from matplotlib.colors import ListedColormap, BoundaryNorm
# cmap = ListedColormap(['#323ea8',  '#339BFF', '#ffffff', '#FF3333', '#900C3F'])
# norm = BoundaryNorm(boundaries, cmap.N)
#
# colnames = ['empirical-3611', 'erdos-renyii', 'small-world', 'scale-free']
# rownames = ['original', 'relabeled', 'synPrun', 'neuroPrun']
# plt.close('all')
# fig, axes = plt.subplots(4,4, figsize=(8,7.5))
#
# # for icat in range(2):
# #     nets = netnames[icat]
# for icol, netname in enumerate(all_network_types):
#     print(netname)
#     rmtx, perm = loadParentAndPermuter(netname, index)
#     wrmtx = weightedFromAdjacency(rmtx, NI, weight=wgt_array)
#
#     omtx = relabelingNeurons(rmtx, perm.argsort())
#     if icol:
#         omtx = relabelingNeurons(omtx, perm.argsort())
#     womtx = weightedFromAdjacency(omtx, NI, weight=wgt_array)
#
#     ax = axes[0, icol]
#     # ax.matshow(womtx.toarray(), cmap=cmap, norm=norm)
#     plotmat(ax, womtx.toarray(), icol, 0, colnames[icol], rownames[0])
#     ax = axes[1, icol]
#     plotmat(ax, wrmtx.toarray(), icol, 1, colnames[icol], rownames[1])
#
#     # ax.matshow(wrmtx.toarray(), cmap=cmap, norm=norm)
#     # ax.locator_params(axis='x', nbins=4)
#     for idtp, dtp in enumerate(['syn', 'neuro']):
#         params = [netname, idtp, index, 2, 5, ]
#         tmtx = trimming(params)
#         newNI = NI-int(idtp*del_frac*istage*NI)
#         wtmtx = weightedFromAdjacency(tmtx, newNI, weight=wgt_array)
#
#         ax = axes[idtp+2, icol]
#         plotmat(ax, wtmtx.toarray(), icol, idtp+2, colnames[icol], rownames[idtp+2])
#         # ax.matshow(wtmtx.toarray(), cmap=cmap, norm=norm)
#         # ax.locator_params(axis='x', nbins=4)
#         #
# plt.suptitle('Network Categories', fontweight='bold', fontsize=16)
# # cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
# # fig.colorbar(axes[0, 0].images[0], cax=cbar_ax)
# cbar_ax = fig.add_axes([0.1, 0.05, 0.742, 0.03])  # [left, bottom, width, height]
# cbar = fig.colorbar(axes[3, 0].images[0], cax=cbar_ax, orientation='horizontal')
#
# tix = (boundaries[:-1]+boundaries[1:])/2.
# cbar.set_ticks(tix)  # Set the tick positions
# cbar.set_ticklabels([
#     r'$w_{ie}$',
#     r'$w_{ii}$',
#     r'$0.0$',
#     r'$w_{ee}$',
#     r'$w_{ei}$'], fontweight='bold')
#
# plt.tight_layout()
# plt.show()
#
# # plt.savefig('demoFigs/orig_relabeled_synprun_neuroprun')
# # plt.savefig('demoFigs/orig_relabeled_synprun_neuroprun.pdf')



# ################## -2- ##################


# ########## 2. simulation ##########
# ############## 2.1. simulator part ###############

# ####checking the simulator with a single set of params
# from funcNest import simulateAndStore
# netname, idtyp, cp_index, idxprun, istage, gvalue = ['er', 0, 0, 2, 5, 7.]
# spkparams = [netname, idtyp, cp_index, idxprun, istage, gvalue]
# simulateAndStore(spkparams)


# ############## 2.2. parallel tasking part ###############

# #### checking the parallel processing algo with multiple sets of params
# import time, itertools
# from multiprocessing import Pool, cpu_count
# from funcNest import simulateAndStore
#
# network_names = all_network_types
# degen_indices = all_degeneration_indices
# network_iters = [0]
# pruning_indices = [1,2]
# pruning_stages = [0,5]
# rel_gI_values = [7.]
# p_stage, d_stage = [[0], [5]]
#
# if __name__ == '__main__':
#     __spec__ = None
#     start_time = time.strftime("%H:%M:%S", time.localtime())
#     for ips, pruning_stages in enumerate([p_stage, d_stage]):
#         pr_indices = [[0], pruning_indices][ips]
#         paramList = list(itertools.product(
#             network_names,
#             degen_indices,
#             network_iters,
#             pr_indices,
#             pruning_stages,
#             rel_gI_values))
#         ppl = Pool(processes=cpu_count())
#         ppl.map(simulateAndStore, paramList)
#
#         finish_time = formatted_time = time.strftime("%H:%M:%S", time.localtime())
#
#     print('started at: ', start_time)
#     print('stopped at: ', finish_time)






# # ################# -3- ##################


# # ########## 3. postSimulation ##########
# # ############## 3.1. raster ###############

# colnames = ['empirical-3611', 'erdos-renyii', 'small-world', 'scale-free']
# rownames = ['parent', 'synPrun', 'neuroPrun']
# cpstages = [0, 5, 5]
#
# idxreal = 0
# gvalue = 7.
#
# plt.close('all')
# fig, axes = plt.subplots(3,4, figsize=(13,5))
# for icol, netname in enumerate(all_network_types):
#     for irow, rowname in enumerate(rownames):
#         istage = bool(irow)*5
#         idtyp = bool(irow)*(irow-1)
#         if istage==0:
#             idxprun = 0
#         else:
#             idxprun = 2
#
#         prm = [netname, idtyp, idxreal, idxprun, istage, int(gvalue)]
#         spk0, newNI, newNE = loadSpikeData(prm)
#
#         timebound = 9000
#         spk = spk0[spk0[:,1]>=timebound]
#
#         sinh = newNI - 20#68
#         bexc = newNI + 80#293
#         fspk = spk[spk[:,0]>sinh]
#         fspk = fspk[fspk[:,0]<bexc]
#         inhfspk = fspk[fspk[:,0]<newNI]
#         excfspk = fspk[fspk[:,0]>=newNI]
#
#         ax = axes[irow, icol]
#         ax.plot(inhfspk[:,1], inhfspk[:,0], 'b.', ms=.5)
#         ax.plot(excfspk[:,1], excfspk[:,0], 'r.', ms=.5)
#
#         if irow==0:
#             ax.set_title(colnames[icol], fontsize=12, fontweight='bold')
#         if icol==0:
#             print(icol)
#             ax.text(timebound-850,newNI+20, rownames[irow], fontsize=12, fontweight='bold')
#         if irow<2:
#             ax.set_xticklabels([])
#         else:
#             ax.set_xlabel('time (ms)', fontsize=10, fontweight='bold')
#         if icol>0:
#             ax.set_yticklabels([])
#         else:
#             ax.set_ylabel('neuron ID', fontsize=10, fontweight='bold')
#
# # plt.tight_layout()
# plt.show()
#
# # # ############## 3.2. Analysis ###############

# #### 3.2.1. functional quantities - ###
# from funcAnalysis import *
# netname, idtyp, cp_index, idxprun, istage, gvalue = ['er', 0, 0, 2, 5, 7.]
# spkparams = [netname, idtyp, cp_index, idxprun, istage, gvalue]
# # spkdt, newNI, newNE = loadSpikeData(spkparams)
# rr = firingRate(spkparams)
# ff = fanoFactor(spkparams)
# cv = cvISI(spkparams)
#
# for netname in all_network_types:
#     spkparams[0] = netname
#     rfc = dynPart(spkparams)
#     print(netname)
#     print(rfc)
#     print('')


# #### 3.2.2. structural quantities - ###
from funcAnalysis import *
sparams = ['er', 0, 0, 2, 5] #netname, idtyp, cp_index, idxprun, istage
weight = np.array([-2.5, -2.5,  0.5,  0.5])
#
dg = meanDegree(sparams)
esw = meanEffectiveLinkWeight(sparams, weight)
sh = contributionToPairwiseSharing(sparams)

for netname in all_network_types:
    sparams[0] = netname
    des = strPart(sparams)
    print(netname)
    print(des)
    print('')

