import time, itertools
import numpy as np

from multiprocessing import Pool, cpu_count

from funcDegeneration import *
from funcNest import *
from parameters import *


networks = all_network_types[:1]
## parent stage -- preDegeneration
parent_stage = all_pruning_stages[:1]
## stages of degeneration
degenerate_stage = all_pruning_stages[1:]

if __name__ == '__main__':
    __spec__ = None
    start_time = time.strftime("%H:%M:%S", time.localtime())

    for ips, pruning_stages in enumerate([parent_stage, degenerate_stage]): # in range(2):
        # pruning_stages = [parent_stage, degenerate_stage][i]
        pruning_indices = [[0], all_pruning_indices][ips]
        paramList = list(itertools.product(
            networks,
            all_degeneration_indices,
            all_network_iterations,
            pruning_indices,
            pruning_stages,
            all_pruning_indices))
        ppl = Pool(processes=cpu_count()-4)
        ppl.map(simulateAndStore, paramList)

        finish_time = formatted_time = time.strftime("%H:%M:%S", time.localtime())

    print('started at: ', start_time)
    print('stopped at: ', finish_time)





                    
    
