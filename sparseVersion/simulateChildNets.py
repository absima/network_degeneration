import numpy as np
import nest, time, itertools

from multiprocessing import Pool, cpu_count

from funcDegeneration import *
from funcNest import *
from parameters import *


degens = np.arange(nDegen)
indices = np.arange(nIndex)
prunes = np.arange(nPrune)
stages = np.arange(1,nStage)

paramList = list(itertools.product(degens, indices, prunes, stages))
paramList = list(itertools.product([1], [0], [0,2], [7,8,9]))

if __name__ == '__main__':
    __spec__ = None
    start_time = time.strftime("%H:%M:%S", time.localtime())

    ppl = Pool(processes=cpu_count()-4)
    ppl.map(spike_data, paramList)

    finish_time = formatted_time = time.strftime("%H:%M:%S", time.localtime())
    
    print('started at: ', start_time)
    print('stopped at: ', finish_time)
    # print("finished in {} seconds".format(finish_time-start_time))

