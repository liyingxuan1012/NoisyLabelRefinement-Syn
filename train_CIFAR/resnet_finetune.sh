#!/bin/bash

cd train_CIFAR

CUDA_VISIBLE_DEVICES=0 python resnet_finetune.py --dataset_dir data/CIFAR100_noisy/noisy_PMD35 --train_dir data/cifar100_PMD35_onestep --pretrained_model_dir models_pretrained/cifar100_PMD35.pt --best_model_dir models/cifar100_PMD35_onestep.pt --log_dir logs/cifar100_PMD35_onestep.log
