import numpy as np

## size of the network blocks
NI, NE = [ 680, 2931]
N0 = NI+NE

indir = '../../nongit/data/pc_data'
outdir = '../outdata'
del_frac = 0.1

nDegen = 2
nIndex = 10
nPrune = 5
nStage = 10
degeneration_indices = np.arange(nDegen) # 0 for link removal and 1 for node removal
network_indices = np.arange(nIndex) # indices to load the desired network
pruning_indices = np.arange(nPrune) # indices for one of the five strategies from the two schemes
gvalues = np.array([3.,4.,5.,6.,7.]) #the relative strength of inhibition: wINH = -g*wEXC


#parent nets are simulated and alyzed separately to avoid repeated iterations. 
parents = [0] # parent stage -- preDegeneration
children = np.arange(1,nStage) # stages of degeneration
samples = [8,9]# some samples to check 


### starter simulation parameters  
g_default = 5.
mije = .5
wii = -g_default*mije
wie = -g_default*mije
wei = mije
wee = mije
J_bg = 5.
p_rate = 15000.
delay = 1.5

simulation_time = 2. #11. # in s
start_record_time = 1. # in s
dt = 0.1 # time resolution in ms




