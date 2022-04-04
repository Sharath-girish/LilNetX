
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/microsoft/archai/blob/master/archai/common/apex_utils.py

from typing import Optional, Sequence, Tuple, List
import os
import argparse
from pyparsing import Dict

import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor, nn
from torch.backends import cudnn
import torch.distributed as dist
from config import Config

import utils
import psutil


class DistUtils:
    def __init__(self, conf_dist: Dict, seed:int = None)->None:
        # region conf vars

        if conf_dist['enabled']:
            self.setup_distributed(conf_dist['local_rank'], conf_dist['address'], 
                                   conf_dist['port'], conf_dist['world_size'])
        self._dist_enable = conf_dist['enabled']
        self._sync_bn = conf_dist['sync_bn']
        self._seed = seed

        # endregion

        # defaults for non-distributed mode
        self._ddp = None
        self._set_ranks(conf_dist)

        # enable distributed processing
        if self.is_dist():
            from torch.nn import parallel
            self._ddp = parallel

            assert dist.is_available() # distributed module is available
            assert dist.is_nccl_available()
            assert dist.is_initialized()
            assert dist.get_world_size() == self.world_size
            assert dist.get_rank() == self.global_rank


        assert self.world_size >= 1
        assert self.local_rank >= 0 and self.local_rank < self.world_size
        assert self.global_rank >= 0 and self.global_rank < self.world_size

        assert self._gpu < torch.cuda.device_count()
        self.device = torch.device('cuda', self._gpu)
        self._setup_gpus(self._seed)

        if self._ddp is not None and dist.is_initialized():
            print(f'Dist initialized, gpu: {self._gpu}, '+\
                  f'local_rank: {self.local_rank}, global_rank: {self.global_rank}, pid: {os.getpid()}')


    def setup_distributed(self, gpu, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=gpu, world_size=world_size)
        torch.cuda.set_device(gpu)


    def _setup_gpus(self, seed:float):
        if seed is not None:
            utils.setup_cuda(seed, self.global_rank)

        torch.autograd.set_detect_anomaly(False)

        print({'gpu_names': utils.cuda_device_names(),
                    'gpu_count': torch.cuda.device_count(),
                    'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']
                        if 'CUDA_VISIBLE_DEVICES' in os.environ else 'NotSet',
                    'cudnn.enabled': cudnn.enabled,
                    'cudnn.benchmark': cudnn.benchmark,
                    'cudnn.deterministic': cudnn.deterministic,
                    'cudnn.version': cudnn.version()
                    })
        print({'memory': str(psutil.virtual_memory())})
        print({'CPUs': str(psutil.cpu_count())})


    def _set_ranks(self, conf_dist)->None:

        # this function needs to work even when torch.distributed is not available
        self.world_size = conf_dist['world_size']
        self.local_rank = conf_dist['local_rank']
        self.global_rank = conf_dist['local_rank']
        self._gpu = self.local_rank % torch.cuda.device_count()

    def is_dist(self)->bool:
        return self._dist_enable

    def is_master(self)->bool:
        return self.global_rank == 0

    def sync_devices(self)->None:
        if self.is_dist():
            torch.cuda.synchronize(self.device)

    def barrier(self)->None:
        if self.is_dist():
            dist.barrier() # wait for all processes to come to this point

    def to_dist(self, model:nn.Module)->nn.Module:

        # conver BNs to sync BNs in distributed mode
        if self.is_dist() and self._sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            print('BNs_converted')

        model = model.to(self.device)

        if self.is_dist():
            model = self._ddp.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank, 
                                                      find_unused_parameters=False)

        return model