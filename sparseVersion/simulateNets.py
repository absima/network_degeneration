import time, itertools
import numpy as np

from multiprocessing import Pool, cpu_count

from funcDegeneration import *
from funcNest import *
from parameters import *


## parent stage -- preDegeneration
parents = [0]
## stages of degeneration
degenerates = np.arange(1,nStage)
## some samples to check 

if __name__ == '__main__':
    __spec__ = None
    start_time = time.strftime("%H:%M:%S", time.localtime())
    
    for i in range(2):
        stages = [parents, degenerates][i]
        pr_indices = [[0], pruning_indices][i]
        paramList = list(itertools.product(
            degeneration_indices, 
            network_indices, 
            pr_indices, 
            stages, 
            gvalues))
        ppl = Pool(processes=cpu_count()-4)
        ppl.map(spike_data, paramList)

        finish_time = formatted_time = time.strftime("%H:%M:%S", time.localtime())
    
    print('started at: ', start_time)
    print('stopped at: ', finish_time)

