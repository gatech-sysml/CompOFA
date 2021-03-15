# CompOFA – Compound Once-For-All Networks for Faster Multi-Platform Deployment
# Under blind review at ICLR 2021: https://openreview.net/forum?id=IgIk8RRT-Z
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import sys
sys.path.append("..")
import torch
from ofa.imagenet_codebase.utils.pytorch_utils import get_net_info
from ofa.model_zoo import ofa_net
import yaml
import copy
from ofa.utils import download_url
from ofa.model_zoo import ofa_net


class LatencyEstimator(object):

	def __init__(self, local_dir='~/.hancai/latency_tools/',
	             url='https://hanlab.mit.edu/files/proxylessNAS/LatencyTools/mobile_trim.yaml'):
		if url.startswith('http'):
			fname = download_url(url, local_dir, overwrite=True)
		else:
			fname = url

		with open(fname, 'r') as fp:
			self.lut = yaml.load(fp)

	@staticmethod
	def repr_shape(shape):
		if isinstance(shape, (list, tuple)):
			return 'x'.join(str(_) for _ in shape)
		elif isinstance(shape, str):
			return shape
		else:
			return TypeError

	def query(self, l_type: str, input_shape, output_shape, mid=None, ks=None, stride=None, id_skip=None,
	          se=None, h_swish=None):
		infos = [l_type, 'input:%s' % self.repr_shape(input_shape), 'output:%s' % self.repr_shape(output_shape), ]

		if l_type in ('expanded_conv',):
			assert None not in (mid, ks, stride, id_skip, se, h_swish)
			infos += ['expand:%d' % mid, 'kernel:%d' % ks, 'stride:%d' % stride, 'idskip:%d' % id_skip,
			          'se:%d' % se, 'hs:%d' % h_swish]
		key = '-'.join(infos)
		return self.lut[key]['mean']

	def predict_network_latency(self, net, image_size=224):
		predicted_latency = 0
		# first conv
		predicted_latency += self.query(
			'Conv', [image_size, image_size, 3],
			[(image_size + 1) // 2, (image_size + 1) // 2, net.first_conv.out_channels]
		)
		# blocks
		fsize = (image_size + 1) // 2
		for block in net.blocks:
			mb_conv = block.mobile_inverted_conv
			shortcut = block.shortcut

			if mb_conv is None:
				continue
			if shortcut is None:
				idskip = 0
			else:
				idskip = 1
			out_fz = int((fsize - 1) / mb_conv.stride + 1)
			block_latency = self.query(
				'expanded_conv', [fsize, fsize, mb_conv.in_channels], [out_fz, out_fz, mb_conv.out_channels],
				mid=mb_conv.depth_conv.conv.in_channels, ks=mb_conv.kernel_size, stride=mb_conv.stride, id_skip=idskip,
				se=1 if mb_conv.use_se else 0, h_swish=1 if mb_conv.act_func == 'h_swish' else 0,
			)
			predicted_latency += block_latency
			fsize = out_fz
		# final expand layer
		predicted_latency += self.query(
			'Conv_1', [fsize, fsize, net.final_expand_layer.in_channels],
			[fsize, fsize, net.final_expand_layer.out_channels],
		)
		# global average pooling
		predicted_latency += self.query(
			'AvgPool2D', [fsize, fsize, net.final_expand_layer.out_channels],
			[1, 1, net.final_expand_layer.out_channels],
		)
		# feature mix layer
		predicted_latency += self.query(
			'Conv_2', [1, 1, net.feature_mix_layer.in_channels],
			[1, 1, net.feature_mix_layer.out_channels]
		)
		# classifier
		predicted_latency += self.query(
			'Logits', [1, 1, net.classifier.in_features], [net.classifier.out_features]
		)
		return predicted_latency

	def predict_network_latency_given_spec(self, spec):
		image_size = spec['r'][0]
		predicted_latency = 0
		# first conv
		predicted_latency += self.query(
			'Conv', [image_size, image_size, 3],
			[(image_size + 1) // 2, (image_size + 1) // 2, 24]
		)
		# blocks
		fsize = (image_size + 1) // 2
		# first block
		predicted_latency += self.query(
			'expanded_conv', [fsize, fsize, 24], [fsize, fsize, 24],
			mid=24, ks=3, stride=1, id_skip=1, se=0, h_swish=0,
		)
		in_channel = 24
		stride_stages = [2, 2, 2, 1, 2]
		width_stages = [32, 48, 96, 136, 192]
		act_stages = ['relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
		se_stages = [False, True, False, True, True]
		for i in range(20):
			stage = i // 4
			depth_max = spec['d'][stage]
			depth = i % 4 + 1
			if depth > depth_max:
				continue
			ks, e = spec['ks'][i], spec['e'][i]
			if i % 4 == 0:
				stride = stride_stages[stage]
				idskip = 0
			else:
				stride = 1
				idskip = 1
			out_channel = width_stages[stage]
			out_fz = int((fsize - 1) / stride + 1)

			mid_channel = round(in_channel * e)
			block_latency = self.query(
				'expanded_conv', [fsize, fsize, in_channel], [out_fz, out_fz, out_channel],
				mid=mid_channel, ks=ks, stride=stride, id_skip=idskip,
				se=1 if se_stages[stage] else 0, h_swish=1 if act_stages[stage] == 'h_swish' else 0,
			)
			predicted_latency += block_latency
			fsize = out_fz
			in_channel = out_channel
		# final expand layer
		predicted_latency += self.query(
			'Conv_1', [fsize, fsize, 192],
			[fsize, fsize, 1152],
		)
		# global average pooling
		predicted_latency += self.query(
			'AvgPool2D', [fsize, fsize, 1152],
			[1, 1, 1152],
		)
		# feature mix layer
		predicted_latency += self.query(
			'Conv_2', [1, 1, 1152],
			[1, 1, 1536]
		)
		# classifier
		predicted_latency += self.query(
			'Logits', [1, 1, 1536], [1000]
		)
		return predicted_latency


class LatencyTable:
    def __init__(self, device='note10', resolutions=(160, 176, 192, 208, 224),
                 use_latency_table=False, network='compofa'):
        self.device = device
        self.use_latency_table = use_latency_table
        self.latency_tables = {}
        if self.use_latency_table:
            for image_size in resolutions:
                self.latency_tables[image_size] = LatencyEstimator(
                    url='checkpoints/latency_table@%s/%d_lookup_table.yaml' % (device, image_size)
                )
                print('Built latency table for image size: %d.' % image_size)
        else:
            from run_NAS import MODEL_DIR
            self.ofa_network = ofa_net(network, model_dir=MODEL_DIR, pretrained=True)
            if self.device == 'cpu':
                self.ofa_network = self.ofa_network.cpu()
            else:
                self.ofa_network = self.ofa_network.cuda()

    def predict_efficiency(self, spec: dict):
        # use latency table if present
        if self.use_latency_table:
            return self.latency_tables[spec['r'][0]].predict_network_latency_given_spec(spec)
        else:
            self.ofa_network.set_active_subnet(e=spec['e'], d=spec['d'], ks=spec['ks'])
            # check if the sub-network has already been evaluated
            latency = self.latency_tables.get((tuple(spec['e']), tuple(spec['d']), tuple(spec['ks'])))
            if latency is not None: 
                return latency 
            subnet = self.ofa_network.get_active_subnet(preserve_weight=True)
            measure_latency = 'gpu64' if self.device == 'gpu' else 'cpu'
            net_info = get_net_info(subnet, input_shape=( 3, spec['r'][0], spec['r'][0]),
                                    measure_latency=measure_latency, print_info=False)
            latency = net_info['%s latency' % measure_latency]['val']
            self.latency_tables[(tuple(spec['e']), tuple(spec['d']), tuple(spec['ks']))] = latency
            return latency
