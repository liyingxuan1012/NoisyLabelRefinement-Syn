#!/bin/bash

cd PMD

CUDA_VISIBLE_DEVICES=0 python resnet_finetune.py --dataset_dir data/CIFAR100_noisy/noisy_PMD70 --train_dir data/cifar100_PMD70_iter_f+g --pretrained_model_dir PMD70.pt --best_model_dir models/cifar100_PMD70_iter_f+g.pt --log_dir logs/cifar100_PMD70_iter_f+g.log --add_generated

CUDA_VISIBLE_DEVICES=0 python resnet_finetune.py --dataset_dir data/CIFAR100_noisy/noisy_PMD70 --train_dir data/cifar100_PMD70_iter --pretrained_model_dir PMD70.pt --best_model_dir models/cifar100_PMD70_iter.pt --log_dir logs/cifar100_PMD70_iter.log
