# CompOFA – Compound Once-For-All Networks for Faster Multi-Platform Deployment
# Under blind review at ICLR 2021: https://openreview.net/forum?id=IgIk8RRT-Z
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random

import horovod.torch as hvd
import torch

from ofa.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from ofa.elastic_nn.networks import OFAMobileNetV3
from ofa.imagenet_codebase.run_manager import DistributedImageNetRunConfig
from ofa.imagenet_codebase.run_manager.distributed_run_manager import DistributedRunManager
from ofa.imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop
from ofa.utils import download_url
from ofa.elastic_nn.training.progressive_shrinking import load_models

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='depth', choices=[
    'kernel', 'depth', 'expand', 'compound', 'teacher',
])
parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--fixed_kernel', action='store_true')
parser.add_argument('--heuristic', type=str, default='none', choices=['simple', 'none'])

args = parser.parse_args()
args.teacher_path = download_url('https://file.lzhu.me/projects/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7',
                                 model_dir='./downloads')

args.manual_seed = 0

args.lr_schedule_type = 'cosine'

args.base_batch_size = 256
args.valid_size = None

args.opt_type = 'sgd'
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = 'bn#bias'
args.fp16_allreduce = False

args.model_init = 'he_fout'
args.validation_frequency = 1
args.print_frequency = 10

args.n_worker = 8
args.resize_scale = 0.08
args.distort_color = 'tf'
args.image_size = '128,160,192,224'
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = 'proxyless'

args.width_mult_list = '1.0'
args.dy_conv_scaling_mode = -1
args.independent_distributed_sampling = False

args.kd_ratio = 1.0
args.kd_type = 'ce'


if args.task == 'kernel':
    args.path = f'runs/{args.name}/teacher2kernel'
    args.dynamic_batch_size = 1
    args.n_epochs = 120
    args.base_lr = 2.6 * args.base_batch_size/2048
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = '3,5,7'
    args.expand_list = '6'
    args.depth_list = '4'
elif args.task == 'depth':
    args.path = f'runs/{args.name}/base2depth/phase{args.phase}'
    args.dynamic_batch_size = 2
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 0.08 * args.base_batch_size/2048
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.expand_list = '6'
        args.depth_list = '3,4'
        args.ks_list = '3,5,7'
    else:
        args.n_epochs = 120
        args.base_lr = 0.24*args.base_batch_size/2048
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.expand_list = '6'
        args.depth_list = '2,3,4'
        args.ks_list = '3,5,7'
elif args.task == 'expand':
    args.path = f'runs/{args.name}/depth2depth_width/phase{args.phase}'
    args.dynamic_batch_size = 4
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 0.08 * args.base_batch_size/2048
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.expand_list = '4,6'
        args.depth_list = '2,3,4'
        args.ks_list = '3,5,7'
    else:
        args.n_epochs = 120
        args.base_lr = 0.24*args.base_batch_size/2048
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.expand_list = '3,4,6'
        args.depth_list = '2,3,4'
        args.ks_list = '3,5,7'
elif args.task == 'teacher':
    args.path = f'runs/teacher/{args.name}'
    args.dynamic_batch_size = 1
    args.n_epochs = 180
    args.base_lr = 2.6 * args.base_batch_size / 2048
    args.warmup_epochs = 0
    args.warmup_lr = -1
    args.kd_ratio = 0.0
    args.expand_list = '3,4,6'
    args.depth_list = '2,3,4'
    args.ks_list = '3,5,7'
elif args.task == 'compound':
    assert(args.heuristic=='simple')
    args.path = f'runs/{args.name}/compound/phase{args.phase}'
    args.dynamic_batch_size = 4
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 0.08*args.base_batch_size/2048
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.expand_list = '3,4,6'
        args.depth_list = '2,3,4'
        args.ks_list = '3,5,7'
    else:
        args.n_epochs = 120
        args.base_lr = 0.24*args.base_batch_size/2048
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.expand_list = '3,4,6'
        args.depth_list = '2,3,4'
        args.ks_list = '3,5,7'
else:
    raise NotImplementedError


if __name__ == '__main__':
    os.makedirs(args.path, exist_ok=True)

    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    num_gpus = hvd.size()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(',')]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 2
    run_config = DistributedImageNetRunConfig(**args.__dict__, num_replicas=num_gpus, rank=hvd.rank())

    # print run config information
    if hvd.rank() == 0:
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # build net from args
    args.width_mult_list = [float(width_mult) for width_mult in args.width_mult_list.split(',')]
    args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list = [int(e) for e in args.expand_list.split(',')]
    args.depth_list = [int(d) for d in args.depth_list.split(',')]

    net = OFAMobileNetV3(
        n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout, base_stage_width=args.base_stage_width, width_mult_list=args.width_mult_list,
        ks_list=args.ks_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list,
        compound=(args.heuristic!='none'), fixed_kernel=args.fixed_kernel,
    )
    # teacher model
    if args.kd_ratio > 0:
        args.teacher_model = OFAMobileNetV3(
            n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=0, width_mult_list=1.0, ks_list=7, expand_ratio_list=6, depth_list=4,
        )
        args.teacher_model.cuda()

    """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    distributed_run_manager = DistributedRunManager(
        args.path, net, run_config, compression, backward_steps=args.dynamic_batch_size, is_root=(hvd.rank() == 0)
    )

    if args.resume:
        distributed_run_manager.load_model()

    distributed_run_manager.save_config()
    # hvd broadcast
    distributed_run_manager.broadcast()

    # load teacher net weights
    if args.kd_ratio > 0:
        load_models(distributed_run_manager, args.teacher_model, model_path=args.teacher_path)

    # training
    from ofa.elastic_nn.training.progressive_shrinking import validate, train

    validate_func_dict = {'image_size_list': {224},
                          'width_mult_list': sorted({0, len(args.width_mult_list) - 1}),
                          'ks_list': None if args.fixed_kernel else sorted({min(args.ks_list), max(args.ks_list)}),
                          'expand_ratio_list': sorted({min(args.expand_list), max(args.expand_list)}),
                          'depth_list': sorted({min(net.depth_list), max(net.depth_list)})}

    if args.task == 'teacher':
        net.set_constraint([4], 'depth')
        net.set_constraint([6], 'expand_ratio')
        net.set_constraint([7], 'kernel_size')
        net.set_active_subnet(d=4, e=6, ks=7)
        train(distributed_run_manager, args,
                        lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict))
    elif args.task == 'kernel':
        validate_func_dict['ks_list'] = sorted(args.ks_list)
        if distributed_run_manager.start_epoch == 0:
            # model_path = download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7',
            #                           model_dir='.torch/ofa_checkpoints/%d' % hvd.rank())
            model_path = args.teacher_path
            load_models(distributed_run_manager, distributed_run_manager.net, model_path=model_path)
            distributed_run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
                                              validate(distributed_run_manager, **validate_func_dict), 'valid')
        train(distributed_run_manager, args,
              lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict))
    elif args.task == 'depth':
        from ofa.elastic_nn.training.progressive_shrinking import supporting_elastic_depth
        supporting_elastic_depth(train, distributed_run_manager, args, validate_func_dict)
    elif args.task == 'expand':
        from ofa.elastic_nn.training.progressive_shrinking import supporting_elastic_expand
        supporting_elastic_expand(train, distributed_run_manager, args, validate_func_dict)
    elif args.task == 'compound':
        from ofa.elastic_nn.training.progressive_shrinking import supporting_compound
        supporting_compound(train, distributed_run_manager, args, validate_func_dict)
    else:
        raise NotImplementedError
