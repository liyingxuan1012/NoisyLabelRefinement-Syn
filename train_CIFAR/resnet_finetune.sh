#!/bin/bash

cd train_CIFAR

CUDA_VISIBLE_DEVICES=0 python resnet_finetune.py --dataset_dir data/CIFAR100_noisy/noisy_PMD70 --train_dir data/noisy_PMD70_0.8 --pretrained_model_dir models_pretrained/cifar100_PMD70.pt --best_model_dir models/cifar100_PMD70_0.8.pt --log_dir logs/cifar100_PMD70_onestep.log
