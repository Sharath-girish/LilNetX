
import os
import time
import torch
import shutil
import config
import argparse
import torch.distributed
from config import Config
from trainer import Trainer
from utils import setup_cuda
from dist_utils import DistUtils

def check_config(conf):
    conf_sched = config.get_conf_sched(conf)
    conf_dataset = config.get_conf_dataset(conf)
    conf_network = config.get_conf_network(conf)
    conf_ffcv = config.get_conf_ffcv(conf)
    if conf_sched['type'] == 'nstep':
        assert min(conf_sched['decay_steps'])>conf_sched['warmup_epochs'], \
            'Cannot decay learning rate before completing warmup'

    cifar_networks = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 
                        'resnet1202', 'vgg16_bn', 'vgg16']
    network_name = conf_network['name']
    if network_name in cifar_networks:
        assert 'cifar' in conf_dataset['name'], f'Network cannot be {network_name} for non-cifar dataset' 

    if conf_ffcv['enabled']:
        assert conf_dataset['name'] == 'imagenet', 'FFCV needs to be disabled for non imagenet datasets'


def main(gpu, conf):
    conf_dist = config.get_conf_dist(conf)
    conf_dist['local_rank'] = gpu
    os.environ['LOCAL_RANK'] = str(conf_dist['local_rank'])
    start_time = time.time()

    conf_ckpt = config.get_conf_checkpoint(conf)
    save_path = os.path.join(conf_ckpt['save_dir'],conf_ckpt['filename'])
    resume_path = save_path if not conf_ckpt['resume_path'] else conf_ckpt['resume_path']

    conf_trainer = config.get_conf_train(conf)
    conf_common = config.get_conf_common(conf)
    dist = DistUtils(conf_dist, conf_common['seed'])

    if os.path.exists(resume_path) and (conf_ckpt['stop_if_complete'] or conf_ckpt['resume']):
        ckpt = torch.load(resume_path,map_location='cpu')
        epoch = ckpt['epoch']
        if epoch == conf_trainer['epochs']:
            print(f'Experiment finished at epoch {epoch}, exiting...')
            exit()
        del ckpt
        conf_common['new_seed'] = int(torch.rand(1)[0]*1e6)

        if dist.is_dist():
            new_seed = torch.Tensor([conf_common['new_seed']]).to(dist.device)
            torch.distributed.broadcast(new_seed,src=0)
            conf_common['new_seed'] = int(new_seed.item())

        if conf_common['seed'] is not None:
            new_seed = conf_common['new_seed']
            setup_cuda(new_seed, conf_dist['local_rank'])
            print(f'Changed random seed to {new_seed}')
            conf_common['seed'] = conf_common['new_seed']

    if os.path.exists(save_path) and conf_ckpt['clear_if_exists'] and dist.is_master():
            save_dir = conf_ckpt['save_dir']
            print(f'Clearing checkpoint dir {save_dir} to start training from scratch')
            shutil.rmtree(save_dir)
    if dist.is_master():
        os.makedirs(conf_ckpt['save_dir'],exist_ok=True)

    conf_wandb = config.get_conf_wandb(conf)
    if conf_wandb['enabled']:
        import wandb
        import hashlib
        id = hashlib.md5(conf_wandb['run_name'].encode('utf-8')).hexdigest()
        wandb_dir = conf_wandb['dir']
        wandb_config = conf #TODO
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir,exist_ok=True)
        wandb.init(project=conf_wandb['project_name'],
                    name=conf_wandb['run_name'],
                    config=wandb_config,
                    id=id,
                    resume=conf_ckpt['resume'] and os.path.exists(resume_path),
                    entity=conf_wandb['entity'],
                    dir=wandb_dir)

    trainer = Trainer(conf, dist)

    if conf_common['seed'] is not None:
        g = torch.Generator()
        g.manual_seed(conf_common['seed'])
    else:
        g = None

    if not conf_common['eval_only']:
        trainer.train(g)
    else:
        trainer.eval()

    if conf_wandb['enabled']:
        wandb.finish()

    if dist.is_dist():
        torch.distributed.destroy_process_group()

    print('Total time taken: {:.2f}s'.format(time.time()-start_time))

if __name__ == '__main__':
    conf = Config(use_args=True)
    conf_dist = config.get_conf_dist(conf)

    if conf_dist['enabled']:
        torch.multiprocessing.spawn(main,(conf,), nprocs=conf_dist['world_size'], join=True)
    else:
        main(0, conf)