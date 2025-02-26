import numpy as np

from funcDegeneration import loadData
# from parameters import *


#### the functions assume a sorted network where the first block of neurons contains inhibiroty population. 

## firing rates
def firing_rate(data):
    '''
    - data is a list of spike_data, number of Inhibitory and number of excitatory nodes.
    
    output: a list of two lists where the first list contains mean firing rate for (E+I, I, E), while the second contains rate variability of the corresponding population blocks
    '''
    data, newNI, newNE = data
    network_size = newNI + newNE
    node_ids, spike_times = data.T
    
    node_ids = node_ids.astype(int)
    firing_rates = np.bincount(node_ids, minlength=network_size)
    recording_time = (simulation_time - start_record_time)/1000 # in seconds
    firing_rates = firing_rates/recording_time

    mean_rates = [
        np.mean(firing_rates), 
        np.mean(firing_rates[:newNI]),
        np.mean(firing_rates[newNI:])
    ]# for population, I population, Epopulation
    sd_rates = [
        np.std(firing_rates), 
        np.std(firing_rates[:newNI]),
        np.std(firing_rates[newNI:])
    ]# for population, I population, Epopulation
    
    return [mean_rates, sd_rates]



def fano_factor(data, ff_binsize=None):
    '''
    - data is a list of spike_data, number of Inhibitory and number of excitatory nodes.
    
    output: a list of fano-factors for (E+I, I, E)
    '''
    def meanff(spktime):
        '''
        - spktime is a spike time array
        output: the fano-factor 
        '''
        last_spike_time = int(np.ceil(spktime[-1]))
        bins = np.arange(-0.05, last_spike_time + 0.05, ff_binsize)
        psth, _ = np.histogram(spktime, bins)
        mean_psth = np.mean(psth)
        return int(mean_psth != 0) * np.var(psth) / mean_psth
    data, newNI, newNE = data
    network_size = newNI + newNE
    node_ids, spike_times = data.T
    
    if ff_binsize is None:
        ff_binsize = 50
    ff_all = meanff(spike_times)
    ff_inh = meanff(spike_times[node_ids<newNI])
    ff_exc = meanff(spike_times[node_ids>=newNI])
    
    return [ff_all, ff_inh, ff_exc]
    
    
    
def cv_ISI(data):
    '''
    - data is a list of spike_data, number of Inhibitory and number of excitatory nodes.
    
    output: a list of CV of the inter-spike-intervals of population blocks (E+I, I, E)
    '''
    data, newNI, newNE = data
    network_size = newNI + newNE
    node_ids, spike_times = data.T
    
    unique_nodes = np.unique(node_ids)
    cv_values = np.zeros(len(unique_nodes))
    for idx, node in enumerate(unique_nodes):
        node_spikes = spike_times[node_ids == node]   
        if len(node_spikes) > 1:  # as isi needs at least two spikes
            isis = np.diff(node_spikes)  
            mean_isi = np.mean(isis)
            std_isi = np.std(isis)
            cv_values[idx] = std_isi / mean_isi if mean_isi != 0 else 0
        else:
            cv_values[idx] = np.nan  
    mean_cv_all = np.nanmean(cv_values)
    mean_cv_inh = np.nanmean(cv_values[:newNI])
    mean_cv_exc = np.nanmean(cv_values[newNI:])
    
    return [mean_cv_all, mean_cv_inh, mean_cv_exc]


def degreeFromSparceMatrix(cmtx):
    '''
    - coomtx is the sparse coo_matrix as input
    - it retuns the outdegrees and degrees of the nodes in the network
    ''' 
    row_degrees = np.bincount(cmtx.row, minlength=cmtx.shape[0])
    col_degrees = np.bincount(cmtx.col, minlength=cmtx.shape[1])
    degrees = row_degrees + col_degrees
    return col_degrees, degrees
    
    

# def structure(mtx, nwE, nwI, g):
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
#
#     N = len(gg)
#     mtx = nx.to_numpy_array(gg)
#     # mtx = np.array(mtx)
#     gndnx = np.array(gg.nodes())
#     nwE = np.where(gndnx%5!=0)[0]
#     nwI = np.where(gndnx%5==0)[0]
#
#     outdg = np.array([gg.out_degree(i) for i in gndnx])
#
#     shh = outdg*(outdg-1)/2.
#     shE = shh[nwE]/1./N
#     mnsh = np.mean(shE)
#     pdens = len(edgeIn)/1./N/N



# simulation_time, start_record_time = [6000., 1000.]
simulation_time = 6. # in s
start_record_time = 1. # in s

iparams = [1, 0, 2, 7]
flag = 'spikes'
data = loadData(iparams, flag)
rr = firing_rate(data)
ff = fano_factor(data, ff_binsize=None)
cv = cv_ISI(data)

flag = 'weightedAdjacency'
data = loadData(iparams, flag)

print('source: ',data[0].col)
print('target: ',data[0].row)
print('weight: ',data[0].data)
    
        


