import numpy as np

from funcDegeneration import loadSpikeData, trimming, weightedFromAdjacency
from parameters import *


#### the functions assume a sorted network where the first block of neurons contains inhibiroty population. 

## firing rates
def firingRate(iparams):
    '''
    - data is a list of spike_data, number of Inhibitory and number of excitatory nodes.
    
    output: a list of two lists where the first list contains mean firing rate for (E+I, I, E), while the second contains rate variability of the corresponding population blocks
    '''
    data = loadSpikeData(iparams)    
    
    data, newNI, newNE = data
    network_size = newNI + newNE
    node_ids, spike_times = data.T
    
    node_ids = node_ids.astype(int)
    firing_rates = np.bincount(node_ids, minlength=network_size)
    recording_time = (simulation_time - start_record_time) # in seconds
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
    
    mnrr = np.mean([mean_rates, sd_rates])
    
    return mnrr



def fanoFactor(iparams, ff_binsize=None):
    '''
    - data is a list of spike_data, number of Inhibitory and number of excitatory nodes.
    
    output: a list of fano-factors for (E+I, I, E)
    '''
    def meanff(spktime):
        '''
        - spktime is a spike time array
        output: the fano-factor 
        '''
        if ff_binsize is None:
            ff_binsize = 50
        last_spike_time = int(np.ceil(spktime[-1]))
        bins = np.arange(-0.05, last_spike_time + 0.05, ff_binsize)
        psth, _ = np.histogram(spktime, bins)
        mean_psth = np.mean(psth)
        return int(mean_psth != 0) * np.var(psth) / mean_psth
        
    data = loadSpikeData(iparams)
    
    data, newNI, newNE = data
    network_size = newNI + newNE
    node_ids, spike_times = data.T
    
    ff_all = meanff(spike_times)
    ff_inh = meanff(spike_times[node_ids<newNI])
    ff_exc = meanff(spike_times[node_ids>=newNI])
    
    mnff = np.array([ff_all, ff_inh, ff_exc])
    
    return mnff
    
    
    
def cvISI(iparams):
    '''
    - data is a list of spike_data, number of Inhibitory and number of excitatory nodes.
    
    output: a list of CV of the inter-spike-intervals of population blocks (E+I, I, E)
    '''
    
    data = loadSpikeData(iparams)
    
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
    mncv = np.array([mean_cv_all, mean_cv_inh, mean_cv_exc])
    return mncv


def meanDegree(iparams): #<<<--- flag=unweighted to load
    '''
    - data is a list of sparse_data of unweighted adjacency matrix, number of Inhibitory and number of excitatory nodes.
    
    ''' 
    netname, idtyp, cp_index, idxprun, istage = iparams
    newNI = NI-idtyp*int(del_frac*istage*NI)
    newNE = NE-idtyp*int(del_frac*istage*NE)
    newNN = newNI + newNE
    
    data = trimming(iparams)        
    in_degrees = np.bincount(data.row, minlength=data.shape[0])
    out_degrees = np.bincount(data.col, minlength=data.shape[1])
    sum_degrees = in_degrees + out_degrees
    
    mean_in_degrees = np.mean(in_degrees)
    mean_out_degrees = np.mean(out_degrees)
    mean_sum_degrees = np.mean(sum_degrees)
    
    mean_in_degrees_I = np.mean(in_degrees[:newNI])
    mean_out_degrees_I = np.mean(out_degrees[:newNI])
    mean_sum_degrees_I = np.mean(sum_degrees[:newNI])
    
    mean_in_degrees_E = np.mean(in_degrees[newNI:])
    mean_out_degrees_E = np.mean(out_degrees[newNI:])
    mean_sum_degrees_E = np.mean(sum_degrees[newNI:])
    
    std_in_degrees = np.std(in_degrees)
    std_out_degrees = np.std(out_degrees)
    std_sum_degrees = np.std(sum_degrees)
    
    std_in_degrees_I = np.std(in_degrees[:newNI])
    std_out_degrees_I = np.std(out_degrees[:newNI])
    std_sum_degrees_I = np.std(sum_degrees[:newNI])
    
    std_in_degrees_E = np.std(in_degrees[newNI:])
    std_out_degrees_E = np.std(out_degrees[newNI:])
    std_sum_degrees_E = np.std(sum_degrees[newNI:])
    
        
    mean_degree = [mean_sum_degrees, mean_sum_degrees_I, mean_sum_degrees_E]
    mean_in_degree = [mean_in_degrees, mean_in_degrees_I, mean_in_degrees_E]
    mean_out_degree = [mean_out_degrees, mean_out_degrees_I, mean_out_degrees_E]
    
    std_degree = [std_sum_degrees, std_sum_degrees_I, std_sum_degrees_E]
    std_in_degree = [std_in_degrees, std_in_degrees_I, std_in_degrees_E]
    std_out_degree = [std_out_degrees, std_out_degrees_I, std_out_degrees_E]
    
    mndeg = [mean_degree, mean_in_degree, mean_out_degree]
    sddeg = [std_degree, std_in_degree, std_out_degree]   

    mndg = np.array([mndeg, sddeg])
    
    return mndg

def meanEffectiveLinkWeight(iparams, weight):
    netname, idtyp, cp_index, idxprun, istage = iparams
    newNI = NI-idtyp*int(del_frac*istage*NI)
    newNE = NE-idtyp*int(del_frac*istage*NE)
    newNN = newNI + newNE
    
    data = trimming(iparams)  
    data = weightedFromAdjacency(data, newNI, weight=weight)    
        
    incoming = np.sum(data.toarray(), 1)
    outgoing = np.sum(data.toarray(), 0)
    projecting = incoming+outgoing
    
    mean_in_weight = np.mean(incoming)
    mean_out_weight = np.mean(outgoing)
    mean_sum_weight = np.mean(projecting)
    
    mean_in_weight_I = np.mean(incoming[:newNI])
    mean_out_weight_I = np.mean(outgoing[:newNI])
    mean_sum_weight_I = np.mean(projecting[:newNI])
    
    mean_in_weight_E = np.mean(incoming[newNI:])
    mean_out_weight_E = np.mean(outgoing[newNI:])
    mean_sum_weight_E = np.mean(projecting[newNI:])
    
    std_in_weight = np.std(incoming)
    std_out_weight = np.std(outgoing)
    std_sum_weight = np.std(projecting)
    
    std_in_weight_I = np.std(incoming[:newNI])
    std_out_weight_I = np.std(outgoing[:newNI])
    std_sum_weight_I = np.std(projecting[:newNI])
    
    std_in_weight_E = np.std(incoming[newNI:])
    std_out_weight_E = np.std(outgoing[newNI:])
    std_sum_weight_E = np.std(projecting[newNI:])
    
    mean_weight = [mean_sum_weight, mean_sum_weight_I, mean_sum_weight_E]
    mean_in_weight = [mean_in_weight, mean_in_weight_I, mean_in_weight_E]
    mean_out_weight = [mean_out_weight, mean_out_weight_I, mean_out_weight_E]
    
    std_weight = [std_sum_weight, std_sum_weight_I, std_sum_weight_E]
    std_in_weight = [std_in_weight, std_in_weight_I, std_in_weight_E]
    std_out_weight = [std_out_weight, std_out_weight_I, std_out_weight_E]
    
    mnwgt = [mean_weight, mean_in_weight, mean_out_weight]
    sdwgt = [std_weight, std_in_weight, std_out_weight]
    
    mnesw = np.array([mnwgt, sdwgt])
    
    return mnesw


def contributionToPairwiseSharing(iparams):
    netname, idtyp, cp_index, idxprun, istage = iparams
    newNI = NI-idtyp*int(del_frac*istage*NI)
    newNE = NE-idtyp*int(del_frac*istage*NE)
    newNN = newNI + newNE
    
    data = trimming(iparams)  
        
    out_degrees = np.bincount(data.col, minlength=data.shape[1])
    
    sharing_pairs = out_degrees*(out_degrees - 1)/2
    shared_by_I = sharing_pairs[:newNI]
    shared_by_E = sharing_pairs[newNI:]
    
    mean_shared = np.mean(sharing_pairs)/newNN
    mean_shared_I = np.mean(shared_by_I)/newNN
    mean_shared_E = np.mean(shared_by_E)/newNN
    
    msh = np.array([mean_shared, mean_shared_I, mean_shared_E])
    
    return mnsh
    
def symmetricity(data):
    return     
  
def strPart(sparams, weight=None):
    dg = meanDegree(sparams)
    if weight is None:
        weight = np.array([-2.5, -2.5,  0.5,  0.5])
    esw = meanEffectiveLinkWeight(sparams, weight)
    sh = contributionToPairwiseSharing(sparams)
    return np.row_stack((dg, esw, sh))

def dynPart(dparams):
    rr = firingRate(dparams)
    ff = fanoFactor(dparams)
    cv = cvISI(dparams)
    return np.row_stack((rr, ff, cv))

      

netname = 'emp'
idtyp = 0
cp_index = 0
idxprun = 2
istage = 5
g = 5.


# ############ dyn
dparams = [netname, idtyp, cp_index, idxprun, istage, g]

# # rfc = dynPart(dparams)
# rr = firingRate(dparams)
# ff = fanoFactor(dparams)
# cv = cvISI(dparams)

spkdt, newNI, newNE = loadSpikeData(dparams)

# ############ str
# netdir = pfold+mfold+netfold
# prmdir = pfold+mfold+permfold
# sparams = netdir, prmdir, netname, idtyp, cp_index, idxprun, istage
# weight = np.array([-2.5, -2.5,  0.5,  0.5])
# des = strPart(sparams, weight=weight)

# dg = meanDegree(sparams)
# esw = meanEffectiveLinkWeight(sparams, weight)
# sh = contributionToPairwiseSharing(sparams)


