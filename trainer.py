from itertools import chain
from re import I
import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np
from layers import Conv2d, Linear

import os
import time
import json
import math
import shutil
import config
import torchvision
import torch.nn as nn
from vgg import VGG, vgg16_bn
from uuid import uuid4
from typing import List
from pathlib import Path
from bisect import bisect
from utils import setup_cuda
from bitEstimator import BitEstimator

from config import Config
from dist_utils import DistUtils
from resnet import _resnet, BasicBlock, Bottleneck, BlurPoolConv2d
from resnet_cifar import _resnet_cifar, BasicBlockCifar

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

def reg_weight_scheduler(epoch, reg_weight, reg_weight_warmup):
    return reg_weight*(min(epoch,reg_weight_warmup)+1)/(reg_weight_warmup+1)

def prior_weight_scheduler(epoch, prior_weight, prior_weight_warmup):
    return prior_weight*(min(epoch,prior_weight_warmup)+1)/(prior_weight_warmup+1)

def get_step_lr(epoch, lr, step_ratio, decay_steps, epochs):
    if epoch >= epochs:
        return 0

    num_decay = bisect(decay_steps,epoch+1)
    cur_lr = lr*(step_ratio)**num_decay
    return cur_lr

def get_cosine_lr(epoch, lr, epochs, lr_peak_epoch, lr_plateau_epoch):
    xs = [0, lr_peak_epoch]
    ys = [1e-4 * lr, lr]
    T_total = epochs-lr_peak_epoch
    T_cur = epoch-lr_peak_epoch
    if epoch >= lr_plateau_epoch:
        if epoch == epochs:
            return 0
        T_cur = lr_plateau_epoch-lr_peak_epoch
        cur_lr = 0.5 * lr * (1 + math.cos(math.pi * T_cur / T_total))
        return cur_lr

    if epoch <lr_peak_epoch:
        return np.interp([epoch], xs, ys)[0]
    elif epoch == epochs:
        return 0
    else:
        cur_lr = 0.5 * lr * (1 + math.cos(math.pi * T_cur / T_total))
        return cur_lr


def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch, lr_plateau_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    if epoch >= lr_plateau_epoch:
        if epoch == epochs:
            return 0
        return np.interp([lr_plateau_epoch], xs, ys)[0]
    return np.interp([epoch], xs, ys)[0]  

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def save_checkpoint(state, is_best, dir, filename='checkpoint.pth'):
    ch.save(state, os.path.join(dir,filename))
    if is_best:
        shutil.copyfile(os.path.join(dir,filename), os.path.join(dir,'model_best.pth.tar'))

def gpu_wrapper(model: nn.Module, dist: DistUtils):
    # move network to GPU if available
    if ch.cuda.is_available():
        if not dist.is_dist():
            device = ch.device('cuda:0')
            model = nn.DataParallel(model).to(device)
        else:
            model = dist.to_dist(model)
    else:
        raise ValueError
    return model

def update_state_dict(cur_state_dict, new_state_dict):
    cur_keys = list(cur_state_dict.keys())
    new_keys = list(new_state_dict.keys())
    if not (cur_keys[0].startswith("module.") ^ new_keys[0].startswith("module.")):
        return new_state_dict
    elif cur_keys[0].startswith("module."):
        for key in new_keys:
            cur_state_dict["module."+key] = new_state_dict[key]
    else:
        for key in cur_keys:
            cur_state_dict[key] = new_state_dict["module."+key]
    return cur_state_dict

class Trainer:

    def __init__(self, conf: Config, dist:DistUtils):
        self.uid = str(uuid4())

        self.conf = conf
        conf_ffcv = config.get_conf_ffcv(self.conf)
        self.dist = dist
        self.train_loader = self.create_train_loader_ffcv() if conf_ffcv['enabled'] else self.create_train_loader()
        self.val_loader = self.create_val_loader_ffcv() if conf_ffcv['enabled'] else self.create_val_loader()
        self.model, self.prob_models, self.scaler = self.create_model_and_scaler()
        self.optimizer, self.prob_optimizer = self.create_optimizers()

    def calc_lr(self, epoch, lr_schedule_type):
        conf_trainer = config.get_conf_train(self.conf)
        conf_sched = config.get_conf_sched(self.conf)

        lr_schedules = {
            'cyclic': lambda epoch: get_cyclic_lr(epoch, conf_sched['arch_lr'], conf_trainer['epochs'], 
                                                  conf_sched['warmup_epochs'], conf_sched['plateau_epochs']),
            'cosine': lambda epoch: get_cosine_lr(epoch, conf_sched['arch_lr'], conf_trainer['epochs'], 
                                                  conf_sched['warmup_epochs'], conf_sched['plateau_epochs']),
            'step': lambda epoch: get_step_lr(epoch, conf_sched['arch_lr'], conf_sched['gamma'], conf_sched['decay_steps'], 
                                              conf_trainer['epochs'])
        }

        return lr_schedules[lr_schedule_type](epoch)

    def get_resolution(self, epoch):
        conf_ffcv = config.get_conf_ffcv(self.conf)
        min_res, max_res = conf_ffcv['min_res'], conf_ffcv['max_res']
        start_ramp, end_ramp = conf_ffcv['start_ramp'], conf_ffcv['end_ramp']
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    def create_optimizers(self):
        conf_network = config.get_conf_network(self.conf)
        conf_optimizer = config.get_conf_optimizer(self.conf)
        conf_sched = config.get_conf_sched(self.conf)
        conf_wd = config.get_conf_wd(self.conf)
        conf_gd = config.get_conf_gd(self.conf)
        prob_model_parameters = []
        for prob_model in self.prob_models.values():
            prob_model_parameters = chain(prob_model_parameters,prob_model.module.parameters())

        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k and 'bias' not in k)]
        bias_params = [v for k, v in all_params if ('bias' in k)]
        decoder_params = [v for k, v in all_params if ('decoder' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k or 'decoder' in k or 'bias' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': conf_wd['bn']
        }, {
            'params': other_params,
            'weight_decay': conf_wd['weights'] if conf_network['vanilla'] else 0.0
        }, {
            'params': bias_params,
            'weight_decay': conf_wd['bias']
        }, {
            'params': decoder_params,
            'weight_decay': conf_wd['decoder'] if not conf_network['vanilla'] else 0.0
        }]

        if conf_optimizer['name'] == 'adam':
            optimizer = ch.optim.Adam(param_groups, lr=conf_sched['arch_lr'])
        elif conf_optimizer['name'] == 'sgd':
            optimizer = ch.optim.SGD(param_groups, lr=conf_sched['arch_lr'], 
                            momentum=conf_optimizer['momentum'], nesterov=conf_optimizer['use_nesterov'])

        prob_optimizer = ch.optim.Adam(prob_model_parameters, lr = conf_sched['prob_lr'])

        self.criterion = nn.CrossEntropyLoss().to(self.dist.device)
        order_dict = {'l2':2, 'l1':1, 'linf':float('inf')}
        self.order = order_dict[conf_gd['order']]

        return optimizer, prob_optimizer

    def create_train_loader(self):

        import torchvision.transforms as transforms
        conf_dataset = config.get_conf_dataset(self.conf)
        conf_trainer = config.get_conf_train(self.conf)
        conf_dist = config.get_conf_dist(self.conf)

        if 'cifar' in conf_dataset['name']:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif conf_dataset['name'] == 'imagenet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                                ])
            
        if conf_dataset['name'] =='cifar10':
            conf_dataset['num_classes'] = 10
            trainset = torchvision.datasets.CIFAR10(
                root=conf_dataset['trainroot'], train=True, download=True, transform=transform_train)
        elif conf_dataset['name'] =='cifar100':
            conf_dataset['num_classes'] = 100
            trainset = torchvision.datasets.CIFAR100(
                root=conf_dataset['trainroot'], train=True, download=True, transform=transform_train)
        elif conf_dataset['name'] =='imagenet':
            conf_dataset['num_classes'] = 1000
            train_path = os.path.join(conf_dataset['trainroot'],'train')
            while os.path.exists(os.path.join(train_path,'train')):
                train_path = os.path.join(train_path,'train')


            trainset = torchvision.datasets.ImageFolder(
                                                train_path,
                                                transform_train)

        if self.dist.is_dist():
            train_sampler = ch.utils.data.distributed.DistributedSampler(trainset)
        else:
            train_sampler = ch.utils.data.RandomSampler(trainset)

        batch_size = conf_trainer['train_batch'] // conf_dist['world_size']

        trainloader = ch.utils.data.DataLoader(
            trainset, batch_size=batch_size, sampler=train_sampler, num_workers=conf_dataset['num_workers'])

        return trainloader


    def create_val_loader(self):
        import torchvision.transforms as transforms
        conf_dataset = config.get_conf_dataset(self.conf)
        conf_trainer = config.get_conf_train(self.conf)

        if 'cifar' in conf_dataset['name']:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif conf_dataset['name'] == 'imagenet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            transform_test = transforms.Compose([
                                             transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             normalize,
                                            ])
            
        if conf_dataset['name'] =='cifar10':
            testset = torchvision.datasets.CIFAR10(
                root=conf_dataset['valroot'], train=False, download=True, transform=transform_test)
    
        elif conf_dataset['name'] =='cifar100':
            testset = torchvision.datasets.CIFAR100(
                root=conf_dataset['valroot'], train=False, download=True, transform=transform_test)
        elif conf_dataset['name'] =='imagenet':
            val_path = os.path.join(conf_dataset['valroot'],'val')
            while os.path.exists(os.path.join(val_path,'val')):
                val_path = os.path.join(val_path,'val')

            testset = torchvision.datasets.ImageFolder(
                                                val_path,
                                                transform_test)

        test_sampler = ch.utils.data.SequentialSampler(testset)

        batch_size = conf_trainer['test_batch'] 

        testloader = ch.utils.data.DataLoader(
            testset, batch_size=batch_size, sampler=test_sampler, num_workers=conf_dataset['num_workers'])


        return testloader


    def create_train_loader_ffcv(self):
        from ffcv.pipeline.operation import Operation
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
            RandomHorizontalFlip, ToTorchImage
        from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder
        from ffcv.fields.basics import IntDecoder

        conf_dataset = config.get_conf_dataset(self.conf)
        conf_dataset['num_classes'] = 1000
        conf_trainer = config.get_conf_train(self.conf)
        conf_dist = config.get_conf_dist(self.conf)
        conf_ffcv = config.get_conf_ffcv(self.conf)
        train_dataset = conf_dataset['trainroot']
        num_workers = conf_dataset['num_workers']
        in_memory = conf_ffcv['in_memory']
        distributed = self.dist.is_dist()
        batch_size = conf_trainer['train_batch'] // conf_dist['world_size']
        this_device = f'cuda:{self.dist._gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader

    def create_val_loader_ffcv(self):
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
            ToTorchImage
        from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
        from ffcv.fields.basics import IntDecoder

        conf_dataset = config.get_conf_dataset(self.conf)
        conf_ffcv = config.get_conf_ffcv(self.conf)
        conf_trainer = config.get_conf_train(self.conf)
        conf_dist = config.get_conf_dist(self.conf)
        val_dataset = conf_dataset['valroot']
        num_workers = conf_dataset['num_workers']
        resolution = conf_ffcv['test_res']
        distributed = self.dist.is_dist()
        batch_size = conf_trainer['test_batch'] // conf_dist['world_size']
        this_device = f'cuda:{self.dist._gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    def create_model_and_scaler(self):
        cifar_networks = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 
                        'resnet1202', 'vgg16_bn', 'vgg16']
        conf_network = config.get_conf_network(self.conf)
        conf_dataset = config.get_conf_dataset(self.conf)
        conf_ffcv = config.get_conf_ffcv(self.conf)
        if conf_network['name'] in cifar_networks:
            if 'resnet' in conf_network['name']:
                model = _resnet_cifar(BasicBlockCifar, [3,3,3], width=conf_network['width'], init_type = conf_network['init_type'],\
                                    compress_bias = conf_network['compress_bias'], vanilla = conf_network['vanilla'], \
                                    mode = conf_network['mode'], boundary = conf_network['boundary'], 
                                    num_classes = conf_dataset['num_classes'], no_shift=conf_network['no_shift'])
            elif 'vgg16' in conf_network['name']:
                model = vgg16_bn(init_type = conf_network['init_type'], vanilla = conf_network['vanilla'], no_shift = conf_network['no_shift'],\
                                mode=conf_network['mode'], compress_bias = conf_network['compress_bias'], boundary = conf_network['boundary'], \
                                num_classes = conf_dataset['num_classes'])
        elif 'resnet' in conf_network['name']:
            if conf_network['name'] == 'resnet18':
                layers = [2, 2, 2, 2]
                block = BasicBlock
            elif conf_network['name'] == 'resnet50':
                layers = [3, 4, 6, 3]
                block = Bottleneck
            large = (conf_dataset['name'] == 'imagenet')
            model = _resnet(block, layers, width_per_group = 64*conf_network['width'], init_type = conf_network['init_type'],\
                            compress_bias = conf_network['compress_bias'], vanilla = conf_network['vanilla'], mode = conf_network['mode'], \
                            boundary = conf_network['boundary'], num_classes = conf_dataset['num_classes'], large = large, \
                            no_shift=conf_network['no_shift'])

        model.apply_straight_through(not conf_network['vanilla'])
        model.apply_compress_bias(conf_network['compress_bias'])
        print('Number of parameters: {:.4f} M'.format(sum([p.numel() for n,p in model.state_dict().items()])/10**6))
        if 'resnet' in conf_network['name']:
            prob_models = {
                        'conv3x3':BitEstimator(9 if not conf_network['single_prob_model'] else 1),
                        'dense':BitEstimator(1)
                        }
            if conf_network['name'] not in cifar_networks:
                prob_models.update({'conv1x1':BitEstimator(1)})
                if large:
                    prob_models.update({'conv7x7':BitEstimator(49)})
        elif 'mobilenet' in conf_network['name']:
            prob_models = {
                        'conv3x3':BitEstimator(9 if not conf_network['single_prob_model'] else 1),
                        'conv1x1':BitEstimator(1),
                        'dense':BitEstimator(1)
                        }
            if conf_network['first']:
                prob_models.update({'first':BitEstimator(9 if not conf_network['single_prob_model'] else 1)})
        elif 'vgg' in conf_network['name']:
            prob_models = {'conv3x3': BitEstimator(9 if not conf_network['single_prob_model'] else 1),
                            'dense1': BitEstimator(1), 'dense2': BitEstimator(2), 'dense3': BitEstimator(3)}
    
        if conf_ffcv['enabled']:
            use_blurpool = conf_network['apply_blur']
            scaler = GradScaler()
            def apply_blurpool(mod: ch.nn.Module):
                for (name, child) in mod.named_children():
                    if isinstance(child, Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                        setattr(mod, name, BlurPoolConv2d(child))
                    else: apply_blurpool(child)
            if use_blurpool: apply_blurpool(model)

            model = model.to(memory_format=ch.channels_last)
            model = model.to(self.dist.device)
            for group in prob_models:
                prob_models[group] = prob_models[group].to(memory_format=ch.channels_last)
                prob_models[group] = prob_models[group].to(self.dist.device)

            if self.dist.is_dist():
                model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.dist._gpu])
                for group in prob_models:
                    prob_models[group] = ch.nn.parallel.DistributedDataParallel(prob_models[group], device_ids=[self.dist._gpu])
        else:
            model = gpu_wrapper(model, self.dist)
            for group in prob_models:
                prob_models[group] = gpu_wrapper(prob_models[group],self.dist)
            scaler = None
    
        return model, prob_models, scaler


    def train(self, g):

        conf_wandb = config.get_conf_wandb(self.conf)
        conf_logger = config.get_conf_logger(self.conf)
        conf_ckpt = config.get_conf_checkpoint(self.conf)
        conf_common = config.get_conf_common(self.conf)
        conf_dist = config.get_conf_dist(self.conf)
        conf_trainer = config.get_conf_train(self.conf)
        conf_ffcv = config.get_conf_ffcv(self.conf)
        conf_network = config.get_conf_network(self.conf)
        
        if conf_ffcv['enabled']:
            ch.autograd.profiler.emit_nvtx(False)
            ch.autograd.profiler.profile(False)
        if conf_logger['use_ac']:
            import torchac
            self.torchac = torchac
        if conf_wandb['enabled']:
            import wandb
            self.wandb = wandb
            wandb.define_metric("epoch")
            wandb.define_metric("lr", step_metric="epoch")
            wandb.define_metric("loss_train", step_metric="epoch")
            wandb.define_metric("loss_reg_train", step_metric="epoch")
            wandb.define_metric("top1_train", step_metric="epoch")
            wandb.define_metric("top5_train", step_metric="epoch")
            wandb.define_metric("net_bytes", step_metric="epoch")
            wandb.define_metric("timings", step_metric="epoch")
            wandb.define_metric("loss_val", step_metric="epoch")
            wandb.define_metric("top1_val", step_metric="epoch")
            wandb.define_metric("top5_val", step_metric="epoch")
            wandb.define_metric("net_bytes_val", step_metric="epoch")
            wandb.define_metric("ac_bytes", step_metric="epoch")
            if conf_logger['calc_sparse_stats']:
                for suffix in ['discrete', 'decoded']:
                    for sparse_type in ['','in_','out_','slice_']:
                        wandb.define_metric(f"sparse_{sparse_type}{suffix}", step_metric="epoch")

        model, prob_models = self.model, self.prob_models
        optimizer, prob_optimizer = self.optimizer, self.prob_optimizer
        start_epoch = best_acc1 = best_epoch = 0

        if conf_ckpt['resume']:
            save_path = os.path.join(conf_ckpt['save_dir'],conf_ckpt['filename'])
            resume_path = save_path if not conf_ckpt['resume_path'] else conf_ckpt['resume_path']
            if os.path.exists(resume_path):
                ckpt = ch.load(resume_path, map_location='cpu')
                start_epoch = ckpt['epoch']
                model.load_state_dict(update_state_dict(model.state_dict(),ckpt['model']))
                for group_name in prob_models:
                    prob_models[group_name].load_state_dict(update_state_dict(prob_models[group_name].state_dict(),ckpt['prob_models'][group_name]))
                optimizer.load_state_dict(ckpt['optimizer'])
                prob_optimizer.load_state_dict(ckpt['prob_optimizer'])
                best_acc1 = ckpt['best_acc1']
                best_epoch = ckpt['best_epoch']
                print(f'Checkpoint found, continuing training from epoch {start_epoch}')
                new_seed = conf_common['seed']
                setup_cuda(new_seed,conf_dist['local_rank'])
                print(f'Changing random seed to {new_seed}')
                del ckpt
            else:
                print(f'Resume checkpoint {resume_path} not found, starting training from scratch...')
        epoch_time = time.time()
        start_time = time.time()

        for epoch in range(start_epoch, conf_trainer['epochs']):

            if conf_ffcv['enabled']:
                res = self.get_resolution(epoch)
                self.decoder.output_size = (res, res)

            # train for one epoch
            self.train_epoch(epoch, g)

            # evaluate on validation set
            acc1, bits, ac_bytes = self.validate()

            # remember best acc@1 and save checkpoint
            is_best = acc1.item() > best_acc1
            if is_best:
                best_acc1 = max(acc1.item(), best_acc1)
                best_epoch = epoch
                if conf_wandb['enabled']:
                    wandb.run.summary[f"best_val_top1"] = best_acc1
                    wandb.run.summary[f"best_bytes"] = ac_bytes
                    wandb.run.summary[f"best_epoch"] = epoch

            ckpt = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'prob_models': {group_name:prob_model.state_dict() for group_name,prob_model in prob_models.items()},
                'best_acc1': best_acc1,
                'best_epoch': best_epoch,
                'optimizer' : optimizer.state_dict(),
                'prob_optimizer' : prob_optimizer.state_dict(),
                'bits': bits,
                'ac_bytes': ac_bytes
            }
            if self.dist.is_master():
                save_checkpoint(ckpt, is_best, conf_ckpt['save_dir'], conf_ckpt['filename'])
            if self.dist.is_master() and conf_logger['save_freq'] and (epoch+1)%conf_logger['save_freq']==0:
                save_checkpoint(ckpt, is_best, conf_ckpt['save_dir'], f'checkpoint_{epoch+1}.pth')

            with ch.no_grad():
                if conf_logger['calc_sparse_stats'] and self.dist.is_master() and not conf_network['vanilla']:
                    sparse_discrete = self.calc_sparse_stats(is_discrete=True,is_decoded=False)
                    sparse_decoded = self.calc_sparse_stats(is_discrete=False,is_decoded=True)
                    print({k:s.item() for k,s in sparse_discrete.items()})
                    print({k:s.item() for k,s in sparse_decoded.items()})
                    if conf_wandb['enabled']:
                        wandb.log(sparse_discrete,commit=False)
                        wandb.log(sparse_decoded,commit=False)

            if conf_wandb['enabled']:
                wandb.log({"timings": time.time()-epoch_time, "epoch": epoch}, commit=True)
            epoch_time = time.time()

        if conf_wandb['enabled']:
            wandb.run.summary[f"final_val_top1"] = acc1
            wandb.run.summary[f"final_bytes"] = ac_bytes//1000 if ac_bytes is not None else bits//8000
        print(f'Training complete in {time.time()-start_time:.2f}s')

    def train_epoch(self, epoch, g):
        conf_losses = config.get_conf_losses(self.conf)
        conf_sched = config.get_conf_sched(self.conf)
        conf_network = config.get_conf_network(self.conf)
        conf_ffcv = config.get_conf_ffcv(self.conf)
        conf_wd = config.get_conf_wd(self.conf)
        conf_gd = config.get_conf_gd(self.conf)
        conf_optimizer = config.get_conf_optimizer(self.conf)
        conf_logger = config.get_conf_logger(self.conf)
        conf_wandb = config.get_conf_wandb(self.conf)
        conf_trainer = config.get_conf_train(self.conf)
        reg_weight = reg_weight_scheduler(epoch, conf_losses['reg_weight'], conf_losses['reg_weight_warmup'])

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.3e')
        losses_reg = AverageMeter('Loss_reg', ':.3e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        net_bytes = AverageMeter('Bytes', ':.2e')
        lr = AverageMeter('LR', ':.4f')
        train_loader = self.train_loader
        model, prob_models = self.model, self.prob_models
        optimizer, prob_optimizer =  self.optimizer, self.prob_optimizer
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, losses_reg, top1] + ([net_bytes] if not conf_network['vanilla'] else []),
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        end = time.time()

        lr_start, lr_end = self.calc_lr(epoch, conf_sched['type']), self.calc_lr(epoch + 1, conf_sched['type'])
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        weight_decay = conf_wd['weights']
        group_decay = conf_gd['weights']

        for i, (images, target) in enumerate(train_loader):

            if conf_trainer['max_batches']>0 and (i+1)%conf_trainer['max_batches']==0:
                break

            if ch.cuda.is_available() and not conf_ffcv['enabled']:
                images = images.to(self.dist.device)
                target = target.to(self.dist.device)
            cur_lr = lrs[i]
            adjust_learning_rate(optimizer, cur_lr)
            # measure data loading time
            data_time.update(time.time() - end)

            optimizer.zero_grad(set_to_none=True) 
            if not conf_network['vanilla']:
                prob_optimizer.zero_grad(set_to_none=True) 

            with autocast(enabled=conf_ffcv['enabled']):

                output = model(images)
                loss = self.criterion(output, target)
                if ch.isnan(loss):
                    print('NaN loss')
                    raise Exception('Loss is NaN')
            
                loss_reg = l2_norm = group_norm = 0
                if not conf_network['vanilla']:
                    weights = model.module.get_weights()
                    # if self.dist.is_dist():
                    #     weights = model.module.get_weights()
                    # else:
                    #     weights = model.module.get_weights()
                    bits = num_elems = 0
                    for group_name in weights:
                        if ch.any(ch.isnan(weights[group_name])):
                            raise Exception('Weights are NaNs')
                        cur_bits, prob = self_information(weights[group_name],prob_models[group_name], conf_network['single_prob_model'], is_val=False, g=g)
                        bits += cur_bits
                        num_elems += prob.size(0)
                    loss_reg = bits*reg_weight/num_elems
                    losses_reg.update(loss_reg.float().item())
                    net_bytes.update(bits.float().item()/8)

                if not conf_network['vanilla'] and (weight_decay>0.0 or group_decay>0.0):
                    for m in model.modules():
                        if isinstance(m, Linear) or isinstance(m, Conv2d):
                            p = m.weight
                            group_name = model.module.get_group_name(p)
                            if group_name in conf_wd['modules']:
                                l2_norm += weight_decay*ch.sum(ch.square(p))
                            if group_name in conf_gd['modules']:
                                group_norm += group_decay*ch.sum(ch.linalg.norm(p.reshape(p.size(0),p.size(1),-1),ord=self.order,dim=-1))
                                if conf_gd['enable_l1']:
                                    group_norm += group_decay*ch.sum(ch.linalg.norm(p.reshape(p.size(0),p.size(1),-1),ord=1,dim=-1))
                        
                net_loss = loss+l2_norm+group_norm+loss_reg
            
            if conf_ffcv['enabled']:
                self.scaler.scale(net_loss).backward()
            else:
                net_loss.backward()

            if conf_optimizer['grad_clip']>0.0:
                ch.nn.utils.clip_grad_norm_([p for n,p in model.named_parameters() if 'decoder' in n], conf_optimizer['grad_clip'])
            if conf_ffcv['enabled']:
                self.scaler.step(optimizer)
                if not conf_network['vanilla']:
                    self.scaler.step(prob_optimizer)
                self.scaler.update()
            else:
                optimizer.step()
                prob_optimizer.step()

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            lr.update(cur_lr)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % conf_logger['print_freq'] == 0 or (i+1)==len(train_loader) and self.dist.is_master():
                progress.display(i, cur_lr)

        if conf_wandb['enabled']:
            assert cur_lr == get_lr(optimizer)
            self.wandb.log({"lr": cur_lr,"loss_train": losses.avg, "loss_reg_train": losses_reg.avg,\
                    "top1_train": top1.avg, "top5_train": top5.avg, "net_bytes": net_bytes.avg/1000}, commit=False)
    

    def validate(self):
        conf_network = config.get_conf_network(self.conf)
        conf_ffcv = config.get_conf_ffcv(self.conf)
        conf_logger = config.get_conf_logger(self.conf)
        conf_wandb = config.get_conf_wandb(self.conf)

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        val_loader = self.val_loader
        model = self.model
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to evaluate mode
        model.eval()
        if not conf_network['vanilla']:

            weights = model.module.get_weights()
            # if self.dist.is_dist():
            #     weights = model.module.get_weights()
            # else:
            #     weights = model.get_weights()

        with ch.no_grad():
            end = time.time()
            with autocast(enabled=conf_ffcv['enabled']):
                for i, (images, target) in enumerate(val_loader):

                    if ch.cuda.is_available() and not conf_ffcv['enabled']:
                        images = images.to(self.dist.device)
                        target = target.to(self.dist.device)
                    # compute output
                    output = model(images)
                    if conf_ffcv['enabled'] and conf_ffcv['flip_test']:
                        output += model(ch.flip(images, dims=[3]))

                    loss = self.criterion(output, target)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % conf_logger['print_freq'] == 0 or (i+1)==len(val_loader) and self.dist.is_master():
                        progress.display(i)

            bits = ac_bytes = 0
            if not conf_network['vanilla']:
                for group_name in weights:
                    cur_bits, prob = self_information(weights[group_name], self.prob_models[group_name], conf_network['single_prob_model'], is_val=True)
                    bits += cur_bits

                if conf_logger['use_ac']:
                    with ch.no_grad():
                        for group_name in weights:
                            weight = ch.round(weights[group_name])
                            if conf_network['single_prob_model']:
                                weight = weight.reshape(-1,1)
                            for dim in range(weight.size(1)):
                                weight_pos = weight[:,dim] - ch.min(weight[:,dim])
                                unique_vals, _ = ch.unique(weight[:,dim], return_counts = True)
                                unique_vals = ch.cat((ch.Tensor([unique_vals.min()-0.5]).to(unique_vals),\
                                                        (unique_vals[:-1]+unique_vals[1:])/2,
                                                        ch.Tensor([unique_vals.max()+0.5]).to(unique_vals)))
                                cdf = self.prob_models[group_name](unique_vals,single_channel=dim)
                                cdf = cdf.detach().cpu().unsqueeze(0).repeat(weight.size(0),1)
                                weight_pos = weight_pos.long()
                                unique_vals = ch.unique(weight_pos)
                                mapping = ch.zeros((weight_pos.max().item()+1))
                                mapping[unique_vals] = ch.arange(unique_vals.size(0)).to(mapping)
                                weight_pos = mapping[weight_pos]
                                ac_bytes += len(self.torchac.encode_float_cdf(cdf.detach().cpu(), weight_pos.detach().cpu().to(ch.int16), \
                                                                check_input_bounds=True))

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)+ \
                  'Net Bytes {bytes}'.format(bytes=bits//8000) if not conf_network['vanilla'] else ''+ \
                 f' Ac Bytes {ac_bytes//1000}' if conf_logger['use_ac'] and not conf_network['vanilla'] else '')
        if conf_wandb['enabled']:
            self.wandb.log({"loss_val": losses.avg, "top1_val": top1.avg, "top5_val": top5.avg, "net_bytes_val": bits//8000}, commit=False)
            if conf_logger['use_ac']:
                self.wandb.log({"ac_bytes":ac_bytes//1000}, commit=False)

        return top1.avg, bits, ac_bytes if conf_logger['use_ac'] else 0


    def calc_sparse_stats(self, is_discrete=False, is_decoded=False):
        model = self.model
        # cur_model = model.module if self.dist.is_dist() else model
        assert not (is_discrete and is_decoded)
        net_in = 0
        net_out = 0
        sparse_in = 0
        sparse_out = 0
        net_slice = 0
        sparse_slice = 0
        net = 0
        sparse = 0
        for m in model.modules():
            if isinstance(m, Conv2d):
                param = m.weight
                group_name = model.module.get_group_name(param)
                if is_discrete:
                    param = ch.round(param)
                elif is_decoded:
                    param = model.module.weight_decoders[group_name](ch.round(param))
                weight_oi = ch.sum(ch.abs(param),dim=(2,3))
                net_in += weight_oi.size(1)
                net_out += weight_oi.size(0)
                net_slice += weight_oi.numel()
                net += param.numel()
                sparse_in += weight_oi.size(1)-ch.count_nonzero(ch.sum(weight_oi,dim=0))
                sparse_out += weight_oi.size(0)-ch.count_nonzero(ch.sum(weight_oi,dim=1))
                sparse_slice += weight_oi.numel()-ch.count_nonzero(weight_oi)
                sparse += param.numel()-ch.count_nonzero(param)
                
        sparse_in = sparse_in/net_in 
        sparse_out = sparse_out/net_out 
        sparse_slice = sparse_slice/net_slice 
        sparse = sparse/net
        suffix = 'continuous'
        if is_discrete:
            suffix = 'discrete'
        elif is_decoded:
            suffix = 'decoded'
        return {
                f'sparse_in_{suffix}':sparse_in,
                f'sparse_out_{suffix}':sparse_out,
                f'sparse_slice_{suffix}':sparse_slice,
                f'sparse_{suffix}':sparse
                }

    def get_group_name(self, param):
        if param.dim()==2:
            return 'dense'
        else:
            h = param.size(2)
            w = param.size(3)
            return f'conv{h}x{w}'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, cur_lr=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        if cur_lr is not None:
            entries += ['Lr {:.3e}'.format(cur_lr)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def self_information(weight, prob_model, is_single_model=False, is_val=False, g=None):
    weight = (weight + ch.rand(weight.shape, generator=g).to(weight)-0.5) if not is_val else ch.round(weight)
    weight_p = weight + 0.5
    weight_n = weight - 0.5
    if not is_single_model:
        prob = prob_model(weight_p) - prob_model(weight_n)
    else:
        prob = prob_model(weight_p.reshape(-1,1))-prob_model(weight_n.reshape(-1,1))
    total_bits = ch.sum(ch.clamp(-1.0 * ch.log(prob + 1e-10) / math.log(2.0), 0, 50))
    return total_bits, prob

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with ch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
