import numpy as np
import time, itertools
from multiprocessing import Pool, cpu_count

from funcGeneration import *



numNets = 3
numIters = 10


paramList = list(itertools.product(np.arange(numNets), np.arange(numIters)))

if __name__ == '__main__':
    __spec__ = None
    start_time = time.strftime("%H:%M:%S", time.localtime())

    ppl = Pool(processes=cpu_count()-1)
    ppl.map(generateRelabelSave, paramList)

    finish_time = formatted_time = time.strftime("%H:%M:%S", time.localtime())
    
    print('started at: ', start_time)
    print('stopped at: ', finish_time)
