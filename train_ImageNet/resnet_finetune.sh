#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python resnet_finetune_iter.py --dataset_dir data/CIFAR100_noisy/noisy_pair30 --train_dir data/cifar100_pair30_onestep --valid_dir data/CIFAR100/test --pretrained_model_dir models_pretrained/cifar100_pair30.pt --best_model_dir cifar100_pair30_onestep_f+g.pt --log_dir cifar100_pair30_onestep_f+g.log

CUDA_VISIBLE_DEVICES=1 python resnet_finetune_iter.py --dataset_dir data/CIFAR100_noisy/noisy_sym30 --train_dir data/cifar100_sym30_onestep --valid_dir data/CIFAR100/test --pretrained_model_dir models_pretrained/cifar100_sym30.pt --best_model_dir cifar100_sym30_onestep_f+g.pt --log_dir cifar100_sym30_onestep_f+g.log