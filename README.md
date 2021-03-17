# CompOFA – Compound Once-For-All Networks for Faster Multi-Platform Deployment 
### Accepted as a conference paper at ICLR 2021 [[OpenReview]](https://openreview.net/forum?id=IgIk8RRT-Z)
\
This implementation is adopted from the source code of [Once For All (Cai et al. 2019)](https://github.com/mit-han-lab/once-for-all)

## Compound Once-for-all Networks
![](figures/overview.png)

## Pareto-Optimality and Density Maintained
![](figures/pareto_curves.png)

## Reduced Train and Search Time
<img src="/figures/cost.png" height="264" width="528">

## Outperforms OFA on Overall Average Accuracy
![](figures/avg_accuracy.png)

## Dependencies
Tested with:
- Python 3.7
- `torch` 1.3.1
- `torchvision` 0.4.2
- `horovod` 0.19.3 for multi-GPU training

## Training CompOFA
```
[horovodrun -np <num_gpus> -H <node1:num_gpus>,<node2:num_gpus>...] python train_ofa_net.py --task compound --phase 1 --fixed_kernel --heuristic simple
```


## Pretrained Models
`./ofa/checkpoints/` directory contains pre-trained models for CompOFA-MobileNetV3 with fixed kernel and elastic kernel.


## Evaluating trained Models
See `eval_sampled_config.py` for example on sampling a random *compound* subnet of CompOFA and validating its top-1 accuracy
```
python eval_sampled_config.py --net <PRETRAINED_PATH> --imagenet_path <IMAGENET_PATH>
```


## Searching Trained Network
In the NAS directory run the following command to execute the Neural Architecture Search for finding the optimal sub-networks for its corresponginf target latency.
```
python run_NAS.py --net=<OFA_NETWORK> --target-hardware=<TARGET_HARDWARE> --imagenet-path <IMAGENET_PATH>
```
**--net** takes in the name of the specific type of model to carry out NAS on:
1. `'compofa'` : CompOFA with fixed kernel
2. `'compofa-elastic'` : CompOFA with elastic kernel
3. `'ofa_mbv3_d234_e346_k357_w1.0'` : OFA network

**--target-hardware** takes in the type of deployment hardware that guides the latency-specfic NAS:
1. `'note10'`
2. `'gpu'`
3. `'cpu'`

