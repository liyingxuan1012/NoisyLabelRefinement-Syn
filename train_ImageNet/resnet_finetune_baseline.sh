#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python resnet_finetune_baseline.py --dataset_dir data/CIFAR100_noisy/noisy_pair60 --train_dir data/cifar100_pair60_onestep_baseline --valid_dir data/CIFAR100/test --pretrained_model_dir models_pretrained/cifar100_pair60.pt --best_model_dir cifar100_pair60_onestep_baseline.pt --log_dir cifar100_pair60_onestep_baseline.log
