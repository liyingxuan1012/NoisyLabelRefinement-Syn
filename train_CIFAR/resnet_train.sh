#!/bin/bash

cd train_CIFAR

CUDA_VISIBLE_DEVICES=1 python resnet_train.py --train_dir data/CIFAR10_noisy/noisy_PMD35_U30 --model_dir models_pretrained/cifar10_PMD35_U30.pt --log_dir logs/cifar10_PMD35_U30.log

CUDA_VISIBLE_DEVICES=1 python resnet_train.py --train_dir data/CIFAR10_noisy/noisy_PMD35_U60 --model_dir models_pretrained/cifar10_PMD35_U60.pt --log_dir logs/cifar10_PMD35_U60.log