import time, itertools
import numpy as np

from multiprocessing import Pool, cpu_count

from funcDegeneration import *
from funcNest import *
from parameters import *



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
            all_network_types,
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









############## demo example -1 ###############
# ####checking the simulator with a single set of params
# netname, idtyp, cp_index, idxprun, istage, gvalue = ['er', 0, 0, 2, 5, 5.]
# spkparams = [netname, idtyp, cp_index, idxprun, istage, gvalue]
# simulateAndStore(spkparams)


############## demo example -2 ###############
# #### checking the parallel processing with multiple sets of params
# network_names = ['emp', 'er']
# degen_indices = all_degeneration_indices
# network_iters = [0]
# pruning_indices = [1,2]
# pruning_stages = [0,5]
# rel_gI_values = [5.]
#
# # if 0 in pruning_indices:
# #     parent_stage = [0]
# #     degenerate_stage = pruning_stages[:]
# #     degenerate_stage.remove(0)
#
# p_stage, d_stage = [[0], [5]]
#
# if __name__ == '__main__':
#     __spec__ = None
#     start_time = time.strftime("%H:%M:%S", time.localtime())
#     for ips, pruning_stages in enumerate([p_stage, d_stage]):
#         pr_indices = [[None], pruning_indices][ips]
#         paramList = list(itertools.product(
#             network_names,
#             degen_indices,
#             network_iters,
#             pr_indices,
#             pruning_stages,
#             rel_gI_values))
#         ppl = Pool(processes=cpu_count())
#         ppl.map(simulateAndStore, paramList)
#
#         finish_time = formatted_time = time.strftime("%H:%M:%S", time.localtime())
#
#     print('started at: ', start_time)
#     print('stopped at: ', finish_time)


                    
    
