# CompOFA - Compound Once-For-All Networks for Faster Multi-Platform Deployment
# Under blind review at ICLR 2021: https://openreview.net/forum?id=IgIk8RRT-Z
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import os
import time

import torch

from ofa.elastic_nn.networks import OFAMobileNetV3
from ofa.imagenet_codebase.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_codebase.run_manager import ImagenetRunConfig, RunManager

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--net',
        metavar='OFANET',
        help='OFA networks')

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='./results.csv',
        help='Output File'
    )
    parser.add_argument(
        '--imagenet_path',
        help='The path of ImageNet',
        type=str,
        required=True)
    parser.add_argument(
        '-g',
        '--gpu',
        help='The gpu(s) to use',
        type=str,
        default='all')
    parser.add_argument(
        '-b',
        '--batch-size',
        help='The batch on every device for validation',
        type=int,
        default=768)
    parser.add_argument(
        '-j',
        '--workers',
        help='Number of workers',
        type=int,
        default=128)

    args = parser.parse_args()
    if args.gpu == 'all':
        device_list = range(torch.cuda.device_count())
        args.gpu = ','.join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.batch_size = args.batch_size * max(len(device_list), 1)
    ImagenetDataProvider.DEFAULT_PATH = args.imagenet_path
    return args


def extract_subnet():
    net.sample_active_subnet()
    subnet = net.get_active_subnet(preserve_weight=True)
    return subnet


def validate(subnet, verbose=True):
    run_manager.reset_running_statistics(net=subnet)
    _, top1, _ = run_manager.validate(net=subnet)

    return top1


if __name__ == '__main__':
    args = parse_args()

    """
    Setup Compound OFA MobileNet with fixed kernel & D,W=compound([2,3,4],[3,4,6])
    If evaluating a different network, accordingly modify the net class (proxylessNAS) and/or settings (heuristic, elastic kernel)
    """
    net = OFAMobileNetV3(
            n_classes=1000, dropout_rate=0, width_mult_list=1, ks_list=[3,5,7],
            expand_ratio_list=[3,4,6], depth_list=[2,3,4],
            compound=True, fixed_kernel=True)
    net.load_weights_from_net(torch.load(args.net, map_location='cpu')['state_dict'])
    net.cuda()

    run_config = ImagenetRunConfig(
        test_batch_size=args.batch_size, n_worker=args.workers)

    run_manager = RunManager(
        '.tmp/eval_subnet', net, run_config, init=False)
    run_config.data_provider.assign_active_img_size(224)

    subnet = extract_subnet()
    top1 = validate(subnet)
