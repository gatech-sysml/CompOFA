# CompOFA – Compound Once-For-All Networks for Faster Multi-Platform Deployment
# Under blind review at ICLR 2021: https://openreview.net/forum?id=IgIk8RRT-Z
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random
from tqdm import tqdm
import numpy as np
from datetime import datetime

__all__ = ['EvolutionFinder']


class ArchManager:
    def __init__(self, arch='ofa'):
        self.num_blocks = 20
        self.num_stages = 5
        self.kernel_sizes = [3, 5, 7]
        self.expand_ratios = [3, 4, 6]
        self.depths = [2, 3, 4]
        self.resolutions = [160, 176, 192, 208, 224]
        self.arch = arch 

    def random_sample(self):
        if self.arch == 'ofa':
            sample = self.random_sample_ofa()
        elif 'compofa' in self.arch:
            sample = self.random_sample_compofa()
        return sample

    def random_sample_ofa(self):
        sample = {}
        d = []
        e = []
        ks = []
        for i in range(self.num_stages):
            d.append(random.choice(self.depths))

        for i in range(self.num_blocks):
            e.append(random.choice(self.expand_ratios))
            ks.append(random.choice(self.kernel_sizes))

        sample = {
            'wid': None,
            'ks': ks,
            'e': e,
            'd': d,
            'r': [random.choice(self.resolutions)]
        }

        return sample

    def random_sample_compofa(self):
        sample = {}
        depth_choice = [2, 3, 4]
        width_choice = {2:[3], 3:[4], 4:[6]}
        width_list = {0:[], 1:[], 2:[], 3:[], 4:[]}
        kernel_stages = [3, 3, 5, 3, 3, 5]
        d = list(np.random.choice(depth_choice, self.num_stages))
        e = []
        ks = []

        for unit, depth in enumerate(d):
            width = list(np.random.choice(width_choice[depth], 4))
            width_list[unit].extend(width)
        for w in sorted(width_list.keys()):
            e.extend(width_list[w])
        # Elastic Kernel
        if self.arch == 'compofa-elastic':
            ks = list(np.random.choice(self.kernel_sizes, self.num_blocks))
        # Fixed Kernel - CompOFA
        elif self.arch == 'compofa':
            for k in kernel_stages[1:]:
                ks.extend([k]*4)
			
        sample = {
            'wid': None,
            'ks': ks,
            'e': e,
            'd': d,
            'r': [random.choice(self.resolutions)]
        }
        return sample

    def random_resample(self, sample, width, depth=None, changeAll=False):
        if self.arch == 'ofa':
           self.random_resample_ofa(sample, width)
        elif 'compofa' in self.arch:
           self.random_resample_compofa(sample, width, depth, changeAll)

    def random_resample_ofa(self, sample, i):
        assert i >= 0 and i < self.num_blocks
        sample['ks'][i] = random.choice(self.kernel_sizes)
        sample['e'][i] = random.choice(self.expand_ratios)
	
    def random_resample_compofa(self, sample, width_idx, depth, changeAll=False):
        assert width_idx >= 0 and width_idx < self.num_blocks
        width_choice = {2:[3], 3:[4], 4:[6]}
        if changeAll:
            width = list(np.random.choice(width_choice[depth], 4))
            for idx in range(4):
                sample['e'][width_idx + idx] = width[idx]
        else:
            sample['e'][width_idx] = random.choice(width_choice[depth])
            # Elastic Kernel
            if self.arch == 'compofa-elastic':
                sample['ks'][width_idx] = random.choice(self.kernel_sizes)

    def random_resample_depth(self, sample, stage):
        if self.arch == 'ofa':
            self.random_resample_depth_ofa(sample, stage)
        elif 'compofa' in self.arch:
            self.random_resample_depth_compofa(sample, stage)

    def random_resample_depth_ofa(self, sample, stage):
        assert stage >= 0 and stage < self.num_stages
        sample['d'][stage] = random.choice(self.depths)

    def random_resample_depth_compofa(self, sample, stage):
        assert stage >= 0 and stage < self.num_stages
        old_depth = sample['d'][stage]
        new_depth = random.choice(self.depths)
        sample['d'][stage] = new_depth
        if new_depth != old_depth:
            stage_width_start_map = {0 : 0,
                                     1 : 4,
                                     2 : 8,
                                     3 : 12,
				     4 : 16}
            width_idx = stage_width_start_map[stage]
            self.random_resample_compofa(sample, width_idx, new_depth, True)

    def random_resample_resolution(self, sample):
        sample['r'][0] = random.choice(self.resolutions)

class EvolutionFinder:
    valid_constraint_range = {
        'flops': [150, 600],
        'note10': [15, 60],
	'gpu' : [4, 80],
	'cpu' : [10, 30]
    }

    def __init__(self, constraint_type, efficiency_constraint,
                 efficiency_predictor, accuracy_predictor, **kwargs):
        self.constraint_type = constraint_type
        if not constraint_type in self.valid_constraint_range.keys():
            self.invite_reset_constraint_type()
        self.efficiency_constraint = efficiency_constraint
        if not (efficiency_constraint <= self.valid_constraint_range[constraint_type][1] and
                efficiency_constraint >= self.valid_constraint_range[constraint_type][0]):
            self.invite_reset_constraint()

        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.arch = kwargs.get('arch', 'ofa')
        self.arch_manager = ArchManager(self.arch)
        self.num_blocks = self.arch_manager.num_blocks
        self.num_stages = self.arch_manager.num_stages

        self.mutate_prob = kwargs.get('mutate_prob', 0.1)
        self.population_size = kwargs.get('population_size', 100)
        self.max_time_budget = kwargs.get('max_time_budget', 500)
        self.parent_ratio = kwargs.get('parent_ratio', 0.25)
        self.mutation_ratio = kwargs.get('mutation_ratio', 0.5)

    def invite_reset_constraint_type(self):
        print('Invalid constraint type! Please input one of:', list(self.valid_constraint_range.keys()))
        new_type = input()
        while new_type not in self.valid_constraint_range.keys():
            print('Invalid constraint type! Please input one of:', list(self.valid_constraint_range.keys()))
            new_type = input()
        self.constraint_type = new_type

    def invite_reset_constraint(self):
        print('Invalid constraint_value! Please input an integer in interval: [%d, %d]!' % (
            self.valid_constraint_range[self.constraint_type][0],
            self.valid_constraint_range[self.constraint_type][1])
              )

        new_cons = input()
        while (not new_cons.isdigit()) or (int(new_cons) > self.valid_constraint_range[self.constraint_type][1]) or \
                (int(new_cons) < self.valid_constraint_range[self.constraint_type][0]):
            print('Invalid constraint_value! Please input an integer in interval: [%d, %d]!' % (
                self.valid_constraint_range[self.constraint_type][0],
                self.valid_constraint_range[self.constraint_type][1])
                  )
            new_cons = input()
        new_cons = int(new_cons)
        self.efficiency_constraint = new_cons

    def set_efficiency_constraint(self, new_constraint):
        self.efficiency_constraint = new_constraint

    def random_sample(self):
        constraint = self.efficiency_constraint
        while True:
            sample = self.arch_manager.random_sample()
            efficiency = self.efficiency_predictor.predict_efficiency(sample)
            if efficiency <= constraint:
                return sample, efficiency

    def mutate_sample(self, sample):
        def get_stage(block):
            if block >= 0 and block < 4:
                return 0
            elif block >= 4 and block < 8:
                return 1
            elif block >= 8 and block < 12:
                return 2
            elif block >= 12 and block < 16:
                return 3
            elif block >=16 and block < 20:
                return 4

        constraint = self.efficiency_constraint
        # Timeout constraint
        for i in range(50):
            new_sample = copy.deepcopy(sample)

            if random.random() < self.mutate_prob:
                self.arch_manager.random_resample_resolution(new_sample)

            for i in range(self.num_blocks):
                if random.random() < self.mutate_prob:
                    depth_idx = get_stage(i)
                    depth = sample['d'][depth_idx]
                    self.arch_manager.random_resample(new_sample, i, depth)

            for i in range(self.num_stages):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample_depth(new_sample, i)

            efficiency = self.efficiency_predictor.predict_efficiency(new_sample)
            if efficiency <= constraint:
                return new_sample, efficiency
        return None, None

    def crossover_sample(self, sample1, sample2):
        if self.arch == 'ofa':
            new_sample, efficiency = self.crossover_sample_ofa(sample1, sample2)
        elif 'compofa' in self.arch:
            new_sample, efficiency = self.crossover_sample_compofa(sample1, sample2)
        return new_sample, efficiency

    def crossover_sample_ofa(self, sample1, sample2):
        constraint = self.efficiency_constraint
        # Timeout constraint
        for i in range(50):
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list):
                    continue
                for i in range(len(new_sample[key])):
                    new_sample[key][i] = random.choice([sample1[key][i], sample2[key][i]])

            efficiency = self.efficiency_predictor.predict_efficiency(new_sample)
            if efficiency <= constraint:
                return new_sample, efficiency
        return None, None

    def crossover_sample_compofa(self, sample1, sample2):
        constraint = self.efficiency_constraint
        stage_width_start_map = {0 : 0,
                                 1 : 4,
                                 2 : 8,
                                 3 : 12,
                                 4 : 16}
        # Timeout constraint
        for i in range(50):
            new_sample = copy.deepcopy(sample1)
            # Elastic Kernel
            if self.arch == 'compofa-elastic':
                keys = ['d', 'r', 'ks']
            # Fixed Kernel
            elif self.arch == 'compofa':
                keys = ['d', 'r']
            for key in ['d', 'r', 'ks']: 
                if key == 'r' or key == 'ks':
                    for i in range(len(new_sample[key])):
                        new_sample[key][i] = random.choice([sample1[key][i], sample2[key][i]])
                elif key == 'd':
                    samples = [sample1, sample2]
                    choices = list(np.random.choice(samples, 5)) # depth is of length 5
                    for i, choice in enumerate(choices):
                        new_sample['d'][i] = choice['d'][i]
                        start_idx = stage_width_start_map[i]
                        for idx in range(4):
                            new_sample['e'][start_idx + idx] = choice['e'][start_idx + idx]
            efficiency = self.efficiency_predictor.predict_efficiency(new_sample)
            if efficiency <= constraint:
                return new_sample, efficiency
        return None, None


    def run_evolution_search(self, verbose=False):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        max_time_budget = self.max_time_budget
        population_size = self.population_size
        mutation_numbers = int(round(self.mutation_ratio * population_size))
        parents_size = int(round(self.parent_ratio * population_size))
        constraint = self.efficiency_constraint

        best_valids = [-100]
        population = []  # (validation, sample, latency) tuples
        child_pool = []
        efficiency_pool = []
        best_info = None
        if verbose:
            print('Generate random population...')
        for iter in range(population_size):
            sample, efficiency = self.random_sample()
            child_pool.append(sample)
            efficiency_pool.append(efficiency)

        accs = self.accuracy_predictor.predict_accuracy(child_pool)
        for i in range(mutation_numbers):
            population.append((accs[i].item(), child_pool[i], efficiency_pool[i]))

        if verbose:
            print('Start Evolution...')
        # After the population is seeded, proceed with evolving the population.
        for iter in tqdm(range(max_time_budget), desc='Searching with %s constraint (%s)' % (self.constraint_type, self.efficiency_constraint)):
            parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
            acc = parents[0][0]
            if verbose:
                print('Iter: {} Acc: {}'.format(iter - 1, parents[0][0]))

            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_info = parents[0]
            else:
                best_valids.append(best_valids[-1])

            population = parents
            child_pool = []
            efficiency_pool = []

            for i in range(mutation_numbers):
                new_sample = None
                while new_sample is None:
                    par_sample = population[np.random.randint(parents_size)][1]
                    # Mutate
                    new_sample, efficiency = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            for i in range(population_size - mutation_numbers):
                new_sample = None
                while new_sample is None:
                    par_sample1 = population[np.random.randint(parents_size)][1]
                    par_sample2 = population[np.random.randint(parents_size)][1]
                    # Crossover
                    new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            accs = self.accuracy_predictor.predict_accuracy(child_pool)
            for i in range(population_size):
                population.append((accs[i].item(), child_pool[i], efficiency_pool[i]))

        return best_valids, best_info
