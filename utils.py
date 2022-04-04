from typing import Union
import torch.backends.cudnn as cudnn

import os
import torch
import random
import numpy as np

def setup_cuda(seed:Union[float, int]=None, local_rank:int=0):
    if seed is not None:
        seed = int(seed) + local_rank
        # setup cuda
        cudnn.enabled = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    #torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True # set to false if deterministic
    torch.set_printoptions(precision=10)
    # cudnn.deterministic = True
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()

def cuda_device_names()->str:
    return ', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])