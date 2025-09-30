import os
import random
import datetime
import numpy as np

import torch
import torch.nn as nn

import warnings
import torchvision

torchvision.disable_beta_transforms_warning()
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import arguments
import data_loaders as data_loaders
import train
import model
import argparse
from collections import OrderedDict

from dataset import ASVspoof2019_LA_Train, ASVspoof2021_DF_Eval, ASVspoof2019_LA_Eval, ASVspoof2021_LA_Eval
from logger import Logger
from utils import get_threshold
from data_processing import WaveformAugmetation
from transformers import AutoConfig, Wav2Vec2Model

def set_experiment_environment(args):
    # reproducible
    random.seed(args['rand_seed'])
    np.random.seed(args['rand_seed'])
    torch.manual_seed(args['rand_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DDP env
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args['port']
    args['rank'] = args['process_id']
    args['device'] = f'cuda:{args["process_id"]}'
    torch.distributed.init_process_group(
            backend='nccl', world_size=args['world_size'], rank=args['rank'])
    torch.cuda.empty_cache()

def run(process_id, args, experiment_args):
    # check parent process
    args['process_id'] = process_id
    args['flag_parent'] = process_id == 0
    
    # args
    set_experiment_environment(args)
    if args['attack_type'] == 'LA':
        trainer = train.ModelTrainer()
    trainer.args = args

    # logger
    if args['flag_parent']:
        trainer.logger = Logger.Builder(args['name'], args['project']
            ).tags(args['tags']
            ).description(args['description']
            ).save_source_files(args['path_scripts']
            ).use_local(args['path_log']
            #).use_neptune(args['neptune_user'], args['neptune_token']
            ).build()
        trainer.logger.log_parameter(experiment_args)

    if args['attack_type'] == 'LA':
        # dataset
        train_db_19la = ASVspoof2019_LA_Train(args['path_19LA'])
        test_db_19la = ASVspoof2019_LA_Eval(args['path_19LA'])
        test_db_21la = ASVspoof2021_LA_Eval(args['path_21LA'])
        test_db_21df = ASVspoof2021_DF_Eval(args['path_21DF'])
        train_set = []
        train_set.extend(train_db_19la.train_set)
        test_set = {'19LA':test_db_19la, '21LA':test_db_21la, '21DF':test_db_21df}

        # set class weight
        class_weight = [1, 1]
        if args['use_class_weight']:
            class_num = [args['ratio_spoof'], args['ratio_bona']]
            class_weight = [sum(class_num) / (class_num[0] * 2),       # spoofed_weight
                            sum(class_num) / (class_num[1] * 2)]       # bonafide_weight
            print('class_weight: ',class_weight)
        
        # Data loader
        trainer.train_loader_bona, trainer.sampler_bona, trainer.train_loader_spoof, trainer.sampler_spoof, \
            trainer.val_loader, trainer.eval_loader = data_loaders.get_loaders(args, train_set, train_db_19la.dev_set, test_set)
        trainer.num_val = len(train_db_19la.dev_set) 

    # Data augmentation
    #trainer.da = None
    trainer.da = WaveformAugmetation(['ACN'],
    {'sr': 16000, 'ACN':{'min_snr_in_db': 10, 'max_snr_in_db': 40, 'min_f_decay': -2.0, 'max_f_decay': 2.0, 'p': 1}}).to(args['device'])
    
    # Speech pre-trained model
    config = AutoConfig.from_pretrained(
        args['PLM_name'], 
        finetuning_task="audio-classification",
        revision="main",
    )

    trainer.plm = Wav2Vec2Model.from_pretrained(
        args['PLM_name'],
        from_tf=bool(".ckpt" in args['PLM_name']),
        config=config,
        revision="main",
        ignore_mismatched_sizes=False,
    ).to(args['device'])
    trainer.plm = nn.SyncBatchNorm.convert_sync_batchnorm(trainer.plm)
    trainer.plm = nn.parallel.DistributedDataParallel(
        trainer.plm, device_ids=[args['device']], find_unused_parameters=True
    )

    # classifier
    trainer.classifier = model.B_Linear(
        num_layer = args['num_layers'],
        hidden_size = args['hidden_size'],
        output_size = args['output_size'],
        agg_size=args['agg_size'],
        loss = model.EOCS_EMA(
            args['output_size'], r_real=args['r_real'],
            r_fake=args['r_fake'], alpha=args['alpha'], class_weight=class_weight,
            K=args['ema_k'], beta=args['ema_beta'], assignment=args['ema_assign'], tau=args['ema_tau']
        ),
        loss_bpl = model.BPL(alpha=args['alpha_cl'])
        ).to(args['device'])
    trainer.classifier = nn.SyncBatchNorm.convert_sync_batchnorm(trainer.classifier)
    trainer.classifier = nn.parallel.DistributedDataParallel(
        trainer.classifier, device_ids=[args['device']], find_unused_parameters=True
    )
    
    # optimizer
    trainer.optimizer = torch.optim.Adam(
        list(trainer.plm.parameters()) + list(trainer.classifier.parameters()),
        lr=args['lr'], 
        weight_decay=args['weight_decay']
    )


    # lr scheduler
    trainer.lr_step = 'epoch'
    trainer.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        trainer.optimizer,
        T_0=args['T_0'],
        T_mult=args['T_mult'],
        eta_min=args['lr_min']
    )


    # set thresghold for accuracy
    trainer.threshold = get_threshold(args['r_real'], args['r_fake'])
        
    trainer.run()

if __name__ == '__main__':
    # get arguments
    args, system_args, experiment_args = arguments.get_args()
    
    parser = argparse.ArgumentParser()
    parser_args = parser.parse_args()

    # set reproducible
    random.seed(args['rand_seed'])
    np.random.seed(args['rand_seed'])
    torch.manual_seed(args['rand_seed'])

    # set gpu device
    if args['usable_gpu'] is None: 
        args['gpu_ids'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['usable_gpu']
        args['gpu_ids'] = args['usable_gpu'].split(',')
    
    args['port'] = f'10{datetime.datetime.now().microsecond % 100}'
    assert 0 < len(args['gpu_ids']), 'Only GPU env are supported'

    # set DDP
    args['world_size'] = len(args['gpu_ids'])
    args['batch_size'] = args['batch_size'] // args['world_size']


    # start
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.spawn(
        run, 
        nprocs=args['world_size'], 
        args=(args, experiment_args)
    )