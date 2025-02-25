import numpy as np
import nest, time

from funcDegeneration import *
from parameters import *


def spike_data(iparams):
    idtyp, cp_index, idxprune, istage = iparams
    wmtx = trimming(iparams)
    newNI = NI-idtyp*int(del_frac*istage*NI)
    wmtx = weightedFromAdjacency(wmtx, newNI, [])
    
    N = wmtx.shape[0]
    
    nest.ResetKernel()
    
    nest.SetKernelStatus({"resolution": dt, "print_time": True})
    
    nrnall = nest.Create('iaf_psc_alpha', N)
    nest.Connect(nrnall, nrnall, 'all_to_all', syn_spec={'weight': wmtx.toarray(), 'delay': delay})
    

    bg = nest.Create('poisson_generator', params={'rate':p_rate})#, 'start': start, 'stop': stop})
    nest.Connect(bg, nrnall, syn_spec={'weight':J_bg, 'delay':delay})
    # nest.Connect(bg,nrnall, [J_bg], delay)

    spkD = nest.Create('spike_recorder', params={'start': start, 'stop': simtime})
    nest.Connect(nrnall, spkD)

    endbuild=time.time()
    print( "Simulating.")
    nest.Simulate(simtime)
    endsimulate= time.time()

    spkSend = nest.GetStatus(spkD)[0]['events']['senders']
    spkTime = nest.GetStatus(spkD)[0]['events']['times']

    data = np.column_stack((spkSend, spkTime))
    data[:,1] = data[:,1]-start
    data[:,0] = data[:,0] - 1 ##### index from 0
    
    np.savez_compressed('%s/spikeData_%d_%d_%d_%d.npz'%(outdir, idtyp, cp_index, idxprune, istage), data=data)
    
    return 
    
# def lambNff(dta, NN, nmsec=None):
#     idd0, tmm0 = dta.T
#     bins = np.arange(-0.5, NN, 1)
#     lamm = np.histogram(idd0, bins)[0]
#     lamm = np.array(lamm)/scaler
#
#     mnlam     = np.mean(lamm)
#     sdlam     = np.std(lamm)
#
#     ref_time = int(np.ceil(tmm0[-1]))
#     if nmsec==None:
#         nmsec=50
#     bins = np.arange(-0.05, ref_time+0.05, nmsec)
#
#     psth = np.histogram(tmm0, bins)[0]
#     ff = np.var(psth)/np.mean(psth)
#
#     cvv = []
#     for j in range(N0):
#         dx0 = np.where(idd0==j)[0]
#         tmj = tmm0[dx0]
#         if len(tmj)>2:
#             dtmj = np.diff(tmj)
#             cvi = np.std(dtmj)/np.mean(dtmj)
#             # print (j, len(tmj), cvi)
#             cvv.append(cvi)
#     cvv = np.array(cvv)
#     mncv = np.mean(cvv)
#     return [mnlam, sdlam, ff, mncv]
#
# def structure(mtx, nwE, nwI,g):
#     N = len(mtx)
#     emtx = mtx[nwE,:]
#     imtx = mtx[nwI,:]
#     indE = np.sum(emtx, 0)# exc in-degree to each neuron
#     indI = np.sum(imtx, 0)
#
#     indI = -g*indI
#     efdeg = indE+indI#np.concatenate((indE, indI))
#     mnef = np.mean(efdeg)
#     sdef = np.std(efdeg)
#
#     indeg = np.sum(mtx, 0)
#     oudeg = np.sum(mtx, 1)
#     dggg = indeg+oudeg
#
#     mndeg = np.mean(dggg)
#     sddeg = np.std(indeg)
#
#     # return [mnef, sdef, mndeg, sddeg]
#
#     gg = nx.DiGraph()
#     gg.add_nodes_from(gnodes)
#     gg.add_edges_from(edgeIn)
#     N = len(gg)
#     mtx = nx.to_numpy_array(gg)
#     # mtx = np.array(mtx)
#     gndnx = np.array(gg.nodes())
#     nwE = np.where(gndnx%5!=0)[0]
#     nwI = np.where(gndnx%5==0)[0]
#
#     outdg = np.array([gg.out_degree(i) for i in gndnx])
#     shh = outdg*(outdg-1)/2.
#     shE = shh[nwE]/1./N
#     mnsh = np.mean(shE)
#     pdens = len(edgeIn)/1./N/N


