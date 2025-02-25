import numpy as np

NI, NE = [ 680, 2931]
N0 = NI+NE



indir = '../../nongit/data/pc_data'
outdir = 'outdata'
del_frac = 0.1

nDegen = 2
nIndex = 10
nPrune = 5
nStage = 10


    
g = 5.
mije = .5
# wii, wie, wei, wee = [-606, -606, 303, 202]
wii, wie, wei, wee = [-g*mije, -g*mije, mije, mije]
J_bg = 5.
p_rate = 15000.

tsec = 6.
simtime = tsec*1000.
dt = 0.1

delay = 1.5

start = 1000.