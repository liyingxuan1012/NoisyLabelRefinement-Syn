#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python resnet_train.py --train_dir data/CIFAR10_noisy/noisy_PMD35_A30 --model_dir models_pretrained/cifar10_PMD35_A30.pt --log_dir logs/cifar10_PMD35_A30.log
