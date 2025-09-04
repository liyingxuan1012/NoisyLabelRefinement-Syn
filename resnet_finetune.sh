#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python resnet_finetune.py --dataset_dir data/CIFAR10_noisy/noisy_PMD35_A30 --train_dir data/cifar10_PMD35_A30 --pretrained_model_dir models_pretrained/cifar10_PMD35_A30.pt --best_model_dir models/cifar10_PMD35_A30.pt --log_dir logs/cifar10_PMD35_A30.log
