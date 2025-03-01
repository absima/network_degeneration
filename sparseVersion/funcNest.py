import numpy as np
import nest, time

from funcDegeneration import *
from parameters import *


def simulateAndStore(params):
    # idtyp, cp_index, idxprune, istage, gvalue = params  
    netfldr, prmflder, netname, idtyp, cp_index, idxprun, istage, gvalue = params
    
    wmtx = trimming(params[:-1])
    newNI = NI-idtyp*int(del_frac*istage*NI)
    
    weight = [-gvalue*mije, -gvalue*mije, mije, mije]
    wmtx = weightedFromAdjacency(wmtx, newNI, weight=weight)
    
    N = wmtx.shape[0]
    
    nest.ResetKernel()
    
    nest.SetKernelStatus({"print_time": True})
    
    nrnall = nest.Create('iaf_psc_alpha', N)
    nest.Connect(nrnall, nrnall, 'all_to_all', syn_spec={'weight': wmtx.toarray(), 'delay': delay})
    

    bg = nest.Create('poisson_generator', params={'rate':p_rate})#, 'start': start, 'stop': stop})
    nest.Connect(bg, nrnall, syn_spec={'weight':J_bg, 'delay':delay})
    # nest.Connect(bg,nrnall, [J_bg], delay)
    ms_simtime = simulation_time*1000
    ms_recstart = start_record_time*1000
    spkD = nest.Create('spike_recorder', params={'start': ms_recstart, 'stop': ms_simtime})
    nest.Connect(nrnall, spkD)

    endbuild=time.time()
    print( "Simulating.")
    nest.Simulate(ms_simtime)
    endsimulate= time.time()

    spkSend = nest.GetStatus(spkD)[0]['events']['senders']
    spkTime = nest.GetStatus(spkD)[0]['events']['times']

    data = np.column_stack((spkSend, spkTime))
    data[:,1] = data[:,1]-ms_recstart
    data[:,0] = data[:,0] - 1 ##### index from 0
    
    np.savez_compressed('%s/spikeData_%s_%d_%d_%d_%d_%d.npz'%(outdir, netname, idtyp, cp_index, idxprun, istage, int(gvalue)), data=data)
    
    return 
    



