import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm

from funcDegeneration import *
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




################## -1- ##################
########## preSimulation ##########
#### loading the relabeled and perm and restoring the actual ... followed by trimming
fig, axes = plt.subplots(4,4, figsize=(8,7.5))

for icat in range(2):
    nets = netnames[icat]
    for inet, netname in enumerate(nets):
        print(netname)
        rmtx, perm = loadParentAndPermuter(netname, index)
        wrmtx = weightedFromAdjacency(rmtx, NI, weight=wgt_array)

        omtx = relabelingNeurons(rmtx, perm.argsort())
        if icat:
            omtx = relabelingNeurons(omtx, perm.argsort())

        womtx = weightedFromAdjacency(omtx, NI, weight=wgt_array)

        icol = inet+icat
        ax = axes[0, icol]
        # ax.matshow(womtx.toarray(), cmap=cmap, norm=norm)
        plotmat(ax, womtx.toarray(), icol, 0, colnames[icol], rownames[0])
        ax = axes[1, icol]
        plotmat(ax, wrmtx.toarray(), icol, 1, colnames[icol], rownames[1])

        # ax.matshow(wrmtx.toarray(), cmap=cmap, norm=norm)
        # ax.locator_params(axis='x', nbins=4)
        for idtp, dtp in enumerate(['syn', 'neuro']):
            params = [netname, idtp, index, 2, 5, ]
            tmtx = trimming(params)
            newNI = NI-int(idtp*del_frac*istage*NI)
            wtmtx = weightedFromAdjacency(tmtx, newNI, weight=wgt_array)

            ax = axes[idtp+2, icol]
            plotmat(ax, wtmtx.toarray(), icol, idtp+2, colnames[icol], rownames[idtp+2])
            # ax.matshow(wtmtx.toarray(), cmap=cmap, norm=norm)
            # ax.locator_params(axis='x', nbins=4)
            #
plt.suptitle('Network Categories', fontweight='bold', fontsize=16)
# cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
# fig.colorbar(axes[0, 0].images[0], cax=cbar_ax)
cbar_ax = fig.add_axes([0.1, 0.05, 0.742, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(axes[3, 0].images[0], cax=cbar_ax, orientation='horizontal')

tix = (boundaries[:-1]+boundaries[1:])/2.
cbar.set_ticks(tix)  # Set the tick positions
cbar.set_ticklabels([
    r'$w_{ie}$',
    r'$w_{ii}$',
    r'$0.0$',
    r'$w_{ee}$',
    r'$w_{ei}$'], fontweight='bold')

plt.tight_layout()
plt.show()

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
