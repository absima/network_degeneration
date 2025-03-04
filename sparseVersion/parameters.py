import numpy as np

## size of the network blocks
NI, NE = [ 680, 2931]
N0 = NI+NE

# data directories
pfold = '/mnt/sdb/sima/data/nester'
pfold = '/Users/sima/netProject/nestStuff/degenProject/empsynthData'
mfolds = ['/empNets', '/synthNets']
permfold = '/permutations' # for permuters
netfold = '/netOrdered' # for networks
spkfold = '/spikeData' # for spikes
qntfold = '/qntData' # for estimated quantities

# netnamestring = ['relabeld_and_ordered', 'doubleRelabeled_ordered'] # emp, synth
all_network_types = ['emp', 'er', 'sw', 'sf']
all_degeneration_indices = list(range(2))
all_network_iterations = list(range(10))
all_pruning_indices = list(range(5))
all_pruning_stages = list(range(10))
all_g_values = [3.,4., 5., 6., 7.]


del_frac = 0.1

nDegen = 2
nIndex = 10
nPrune = 5
nStage = 10
degeneration_indices = np.arange(nDegen) # 0 for link removal and 1 for node removal
network_indices = np.arange(nIndex) # indices to load the desired network from a category
pruning_indices = np.arange(nPrune) # indices for one of the five strategies per scheme
gvalues = np.array([3.,4.,5.,6.,7.]) #the relative strength of inhibition: wINH = -g*wEXC


#parent nets are simulated and alyzed separately to avoid repeated iterations. 
parents = [0] # parent stage -- preDegeneration
children = np.arange(1,nStage) # stages of degeneration


### starter simulation parameters  
mije = .5
J_bg = 5.
p_rate = 15000.
delay = 1.5

simulation_time = 11. # in s
start_record_time = 1. # in s
dt = 0.1 # time resolution in ms




