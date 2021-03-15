# CompOFA – Compound Once-For-All Networks for Faster Multi-Platform Deployment
# Under blind review at ICLR 2021: https://openreview.net/forum?id=IgIk8RRT-Z
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import sys
import torch
import time
import math
import copy
import random
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd

from torchvision import transforms, datasets
from matplotlib import pyplot as plt

sys.path.append("..")
from ofa.model_zoo import ofa_net
from ofa.utils import download_url
from accuracy_predictor import AccuracyPredictor
from flops_table import FLOPsTable
from latency_table import LatencyTable
from evolution_finder import EvolutionFinder
from imagenet_eval_helper import evaluate_ofa_subnet, evaluate_ofa_specialized

parser = argparse.ArgumentParser()
parser.add_argument(
    '-n',
    '--net',
    metavar='OFANET',
    help='OFA network',
    required=True)
parser.add_argument(
    '-t',
    '--target-hardware',
    metavar='TARGET_HARDWARE',
    help='Target Hardware',
    required=True)
parser.add_argument(
    '--imagenet-path',
    metavar='IMAGENET_PATH',
    help='The path of ImageNet',
    type=str,
    required=True)

args = parser.parse_args()
arch = {'compofa' : ('compofa', 'model_best_compofa_simple.pth.tar'),
        'compofa-elastic' : ('compofa-elastic', 'model_best_compofa_simple_elastic.pth.tar'),
        'ofa_mbv3_d234_e346_k357_w1.0' : ('ofa', 'ofa_mbv3_d234_e346_k357_w1.0'),
       }
hardware_latency = {'note10' : [15, 20, 25, 30],
                    'gpu' : [15, 25, 35, 45],
                    'cpu' : [12, 15, 18, 21]}
MODEL_DIR = '../ofa/checkpoints/%s' % (arch[args.net][1])
imagenet_data_path = args.imagenet_path 
# imagenet_data_path = '/srv/data/datasets/ImageNet/' 

# set random seed
random_seed = 3
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')

# Initialize the OFA Network
ofa_network = ofa_net(args.net, model_dir=MODEL_DIR, pretrained=True)
if args.target_hardware == 'cpu':
    ofa_network = ofa_network.cpu()
else:
    ofa_network = ofa_network.cuda()
print('The OFA Network is ready.')

# Carry out data transforms
if cuda_available:
    def build_val_transform(size):
        return transforms.Compose([
            transforms.Resize(int(math.ceil(size / 0.875))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(imagenet_data_path, 'val'),
            transform=build_val_transform(224)
        ),
        batch_size=250,  # test batch size
        shuffle=True,
        num_workers=16,  # number of workers for the data loader
        pin_memory=True,
        drop_last=False,
    )
    print('The ImageNet dataloader is ready.')
else:
    data_loader = None
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')


# set up the accuracy predictor
accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)
print('The accuracy predictor is ready!')
print(accuracy_predictor.model)

# set up the latency table
target_hardware = args.target_hardware
use_latency_table = True if target_hardware == 'note10' else False
latency_table = LatencyTable(device=target_hardware,
                             use_latency_table=use_latency_table,
                             network=args.net)

""" Hyper-parameters for the evolutionary search process
    You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
"""
latency_constraint = hardware_latency[args.target_hardware][0]  # ms
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': target_hardware, # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
    'arch' : arch[args.net][0],
}

# initialize the evolution finder and run NAS
finder = EvolutionFinder(**params)
result_lis = []
for latency in hardware_latency[args.target_hardware]:
    finder.set_efficiency_constraint(latency)
    best_valids, best_info = finder.run_evolution_search()
    result_lis.append(best_info)
print("NAS Completed!")

# evaluate the searched model on ImageNet
models = []
if cuda_available:
    for result in result_lis:
        _, net_config, latency = result
        print('Evaluating the sub-network with latency = %.1f ms on %s' % (latency, target_hardware))
        top1 = evaluate_ofa_subnet(
            ofa_network,
            imagenet_data_path,
            net_config,
            data_loader,
            batch_size=250,
            device='cuda:0' if cuda_available else 'cpu')
        models.append([net_config, top1, latency])

df = pd.DataFrame(models, columns=['Model', 'Accuracy', 'Latency'])
df.to_csv('NAS_results.csv')
print('NAS results saved to NAS_results.csv')
