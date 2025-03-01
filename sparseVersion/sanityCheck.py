import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm

from funcDegeneration import *
from funcNest import simulateAndStore
from parameters import *


'''
This is sort of verifying the overall steps, preSimulation(mainly trimming), simulation and postSimulation. The trimming algorithms (syn + neuro) and weight assignment to the binary nets are verified. While in synaptic pruning, the nodes retain their original labels, in nodal pruning, I block and E block are relabled separately but retain the same structure as the original in that I block have smaller indices (ordered first in the matrix) than the E blocks. In the end we inspect the spike data.  
'''



def plotmat(ax, mtx, icol, irow, colname=None, rowname=None):
    ''' 
    to plot a matrix plot on (irow,icol) position
    '''
    ax.matshow(mtx, cmap=cmap, norm=norm)
    ax.set_aspect('auto')
    
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=4)
    if icol:
        ax.set_yticklabels([])
    if not icol:
        ax.set_ylabel('Node ID', fontsize=10, fontweight='bold')
    if irow==0:
        ax.set_title(colname, fontsize=12, fontweight='bold')
    if irow==3:
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Node ID', color ='white', fontsize=10, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), color='white', rotation=45)
    else:
        ax.set_xticklabels([])
    
    # if not icol:
    #     ax.text(-(1.2)*len(mtx), len(mtx)/2, rowname, fontsize=12, fontweight='bold')
    if icol==3:
        ax.text((1.1)*len(mtx), len(mtx)/2, rowname, fontsize=12, fontweight='bold')
    return 
    



netcat = ['emp', 'synth']
netnames = [['emp'], ['er', 'sw', 'sf']]

netnamestring = ['relabeld_and_ordered', 'doubleRelabeled_ordered']
pfold = '../../empsynthData'
mfolds = ['/empNets', '/synthNets']
permfold = '/permutations'
netfold = '/netOrdered'
spkfold = '/spikeData'

idxprun = 2 # 2 is for random deletion
istage = 5 # 5 is for 5*10 percent of the original to delete  

dtyp = [0,1]
dindx = [0]
dprun = [0,2]
dstag = [0,5]
gvals = [5]

# name_dtyp_indx_dxprun_dxstage_g


np.random.seed(4866867)

index = 0 # net variant

wgt_array = [-600, -606, 302, 203]
wii, wie, wei, wee = wgt_array 
boundaries = np.array([wie-0.1, wii-0.1, -0.1, 0.1, wee+0.1, wei+0.1  ])



cmap = ListedColormap(['#323ea8',  '#339BFF', '#ffffff', '#FF3333', '#900C3F'])
norm = BoundaryNorm(boundaries, cmap.N)

colnames = ['empirical-3611', 'erdos-renyii', 'small-world', 'scale-free']
rownames = ['original', 'relabeled', 'synPrun', 'neuroPrun']
 
 
 
 
    
plt.close('all')




# ################## -1- ##################
# ########## preSimulation ##########
# #### loading the relabeled and perm and restoring the actual ... followed by trimming
# fig, axes = plt.subplots(4,4, figsize=(8,7.5))
#
# for icat in range(2):
#     nets = netnames[icat]
#     for inet, netname in enumerate(nets):
#         print(netname)
#         mfld = mfolds[icat]
#         ndir = pfold+mfld+netfold
#         pdir = pfold+mfld+permfold
#
#         rmtx, perm = loadParentAndPermuter(netname, index, ndir, pdir)
#         wrmtx = weightedFromAdjacency(rmtx, NI, weight=wgt_array)
#
#         omtx = relabelingNeurons(rmtx, perm.argsort())
#         if icat:
#             omtx = relabelingNeurons(omtx, perm.argsort())
#
#         womtx = weightedFromAdjacency(omtx, NI, weight=wgt_array)
#
#         icol = inet+icat
#         ax = axes[0, icol]
#         # ax.matshow(womtx.toarray(), cmap=cmap, norm=norm)
#         plotmat(ax, womtx.toarray(), icol, 0, colnames[icol], rownames[0])
#         ax = axes[1, icol]
#         plotmat(ax, wrmtx.toarray(), icol, 1, colnames[icol], rownames[1])
#
#         # ax.matshow(wrmtx.toarray(), cmap=cmap, norm=norm)
#         # ax.locator_params(axis='x', nbins=4)
#         for idtp, dtp in enumerate(['syn', 'neuro']):
#             params = [ndir, pdir, netname, idtp, index, 2, 5, ]
#             tmtx = trimming(params)
#             newNI = NI-int(idtp*del_frac*istage*NI)
#             wtmtx = weightedFromAdjacency(tmtx, newNI, weight=wgt_array)
#
#             ax = axes[idtp+2, icol]
#             plotmat(ax, wtmtx.toarray(), icol, idtp+2, colnames[icol], rownames[idtp+2])
#             # ax.matshow(wtmtx.toarray(), cmap=cmap, norm=norm)
#             # ax.locator_params(axis='x', nbins=4)
#             #
# plt.suptitle('Network Categories', fontweight='bold', fontsize=16)
# # cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
# # fig.colorbar(axes[0, 0].images[0], cax=cbar_ax)
# cbar_ax = fig.add_axes([0.1, 0.05, 0.742, 0.03])  # [left, bottom, width, height]
# cbar = fig.colorbar(axes[3, 0].images[0], cax=cbar_ax, orientation='horizontal')
#
# tix = (boundaries[:-1]+boundaries[1:])/2.
# cbar.set_ticks(tix)  # Set the tick positions
# cbar.set_ticklabels([
#     r'$w_{ie}=%.1f$'%wie,
#     r'$w_{ii}=%.1f$'%wii,
#     r'$0.0$',
#     r'$w_{ee}=%.1f$'%wee,
#     r'$w_{ei}=%.1f$'%wei], fontweight='bold')
#
# plt.tight_layout()
# plt.show()
#
# plt.savefig('orig_relabeled_synprun_neuroprun')
# plt.savefig('orig_relabeled_synprun_neuroprun.pdf')





################## -2- ##################
# ########## simulation ##########
# idxreal = 0
# idxprun = 2
# gvalue = gvals[0]
# for ids, (idtyp, istage) in enumerate([(di, si) for di in dtyp for si in dstag]):
#     if ids==2:# duplicate with ids=0
#         continue
#     for inet, netname in enumerate(netnames[0]+netnames[1]):
#         mfld = mfolds[bool(inet)]
#         ndir = pfold+mfld+netfold
#         pdir = pfold+mfld+permfold
#         params = [ndir, pdir, netname, idtyp, idxreal, idxprun, istage, gvalue]
#
#         simulateAndStore(params)
        




# ################# -3- ##################
# ########## postSimulation ##########
# idxreal = 0
# idxprun = 2
# gvalue = 5
#
# plt.close('all')
# fig, axes = plt.subplots(3,4, figsize=(13,5))
# for ids, (idtyp, istage) in enumerate([(di, si) for di in dtyp for si in dstag]):
#     if ids==2: # duplicate with ids=0
#         continue
#     if ids<2:
#         count = ids
#     else:
#         count = ids-1
#     for inet, netname in enumerate(netnames[0]+netnames[1]):
#         mfld = mfolds[bool(inet)]
#         ndir = pfold+mfld+netfold
#         pdir = pfold+mfld+permfold
#         # params = [ndir, pdir, netname, idtyp, idxreal, idxprun, istage, gvalue]
#
#         prm = tuple([outdir, netname, idtyp, idxreal, idxprun, istage, int(gvalue)])
#
#         spk = np.load('%s/spikeData_%s_%d_%d_%d_%d_%d.npz'%prm)['data']
#         newNI = NI-idtyp*int(del_frac*istage*NI)
#         sinh = newNI - 20#68
#         bexc = newNI + 80#293
#         fspk = spk[spk[:,0]>sinh]
#         fspk = fspk[fspk[:,0]<bexc]
#         inhfspk = fspk[fspk[:,0]<newNI]
#         excfspk = fspk[fspk[:,0]>=newNI]
#
#         print(len(spk))
#         print(len(fspk))
#         # print(count, inet)
#         # print(spk.shape)
#         # print('')
#         ax = axes[count, inet]
#         ax.plot(inhfspk[:,1], inhfspk[:,0], 'b.', ms=.5)
#         ax.plot(excfspk[:,1], excfspk[:,0], 'r.', ms=.5)
#
#         if count==0:
#             ax.set_title(colnames[inet], fontsize=12, fontweight='bold')
#         if inet==0:
#             ax.text(-700, newNI+20, ['parent', 'synPrun', 'neuroPrun'][count], fontsize=12, fontweight='bold')
#         if count==2:
#             ax.set_xlabel('time (in ms)')
# plt.tight_layout()
# plt.show()
#
# # plt.savefig('raster_demo.pdf')
# # plt.savefig('raster_demo')



















#
# demoTypes = [
#     ['parent (original)', 'parent permute A', 'child of A (synaptic-)', 'child of A (neuro-)'],
#     ['parent (original)', 'parent permute B', 'child of B (synaptic-)', 'child of B (neuro-)'],
# ]
# ## figure
# plt.close('all')
# boundaries = [-610, -10, 10, 250, 350]  # For example: data values in the range [0, 1), [1, 2), ...
# # boundaries = [wii+0.1*wii, wii-0.1*wii, wee-0.1*wee, wei+0.1*wei ]
# cmap = ListedColormap(['#339BFF', '#ffffff', '#FF3333', '#900C3F'])
# norm = BoundaryNorm(boundaries, cmap.N)
#
# ####case-1: loading the relabeled and perm and restoring the actual ... followed by trimming
# fig, axes = plt.subplots(len(demoTypes),len(demoTypes[0]), figsize=(12,5))
# for idx in range(2):
#     lister = []
#     #### loading the relabeled stored matrix with its permutation
#     string_id = str(idx).zfill(2)
#     rmtx = load_npz('%s/sparse_relabeld_and_ordered_%s.npz'%(indir, string_id))
#     permuter = np.load('%s/node_permutations_3611x10.npy'%indir)[idx]
#     #### restore the original
#     omtx = relabelingNeurons(rmtx, perm=permuter.argsort())
#
#     #append them in a list
#     lister = lister+[omtx, rmtx]
#
#     # parameters for synTrim and neuroTrim
#     iparamss = ['emp', 0, idx, idxprun, istage]
#     iparamsn = ['emp', 1, idx, idxprun, istage]
#     #### link loss
#     lister.append(trimming(iparamss))
#     #### node loss
#     lister.append(trimming(iparamsn))
#
#     #### weight assignment to sparse matrices
#     numNI = [NI, NI, NI, NI-int(del_frac*istage*NI)]
#     weight = [-606, -606, 303, 202]
#     lister = [weightedFromAdjacency(lister[i], numNI[i], weight=weight) for i in range(len(lister))]
#
#     #### extracting the weight matrices
#     lister = [i.toarray() for i in lister]
#
#     for imatrix, matrix in enumerate(lister):
#         ax= axes[idx, imatrix]
#         ax.matshow(matrix, cmap = cmap, norm=norm)
#         ax.set_title(demoTypes[idx][imatrix], fontweight='bold', fontsize=12)
#         ax.set_xticklabels([])
#
# plt.suptitle('Degeneration, syn+neuro, randPrun, stage5', fontweight='bold', fontsize=14)
# plt.tight_layout()
# plt.show()
#
# # plt.savefig('Degeneration_synaptic_neuronal_random_stage5.pdf')
#
#
#
#
# # ####>>>>> case-2: creating actual networks and its relabeled version with the permutation online, ... followed by trimming
# # fig, axes = plt.subplots(len(demoTypes),len(demoTypes[0]), figsize=(12,5))
# # for idx in range(2):
# #     lister = []
# #     #### let's create random demo network here
# #     mtx = np.random.binomial(1, .1, (N0,N0))
# #     np.fill_diagonal(mtx,0)
# #     cmtx = coo_matrix(mtx)
# #     lister.append(cmtx)
# #     permuter = np.random.permutation(N0) # random permuter
# #     #### relabeling with a permuter
# #     lister.append(relabelingNeurons(lister[0], perm=permuter))
# #
# #     #### link loss
# #     lister.append(trimming(['emp', 0, [lister[1], permuter], idxprun, istage]))
# #     #### node loss
# #     lister.append(trimming(['emp', 1, [lister[1], permuter], idxprun, istage]))
# #
# #     #### weight assignment to sparse matrices
# #     numNI = [NI, NI, NI, NI-int(del_frac*istage*NI)]
# #     weight = [-606, -606, 303, 202]
# #     lister = [weightedFromAdjacency(lister[i], numNI[i], weight=weight) for i in range(len(lister))]
# #
# #     #### extracting the weight matrices
# #     lister = [i.toarray() for i in lister]
# #
# #     for imatrix, matrix in enumerate(lister):
# #         ax= axes[idx, imatrix]
# #         ax.matshow(matrix, cmap = cmap, norm=norm)
# #         ax.set_title(demoTypes[idx][imatrix], fontweight='bold', fontsize=12)
# #         ax.set_xticklabels([])
# #
# # plt.suptitle('Degeneration, syn+neuro, randPrun, stage5', fontweight='bold', fontsize=14)
# # plt.tight_layout()
# # plt.show()
# #
# # # plt.savefig('Degeneration_synaptic_neuronal_random_stage5.pdf')
