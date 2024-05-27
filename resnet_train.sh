#!/bin/bash

#$-l rt_AG.small=1
#$-l h_rt=05:00:00
#$-j y
#$-cwd

source /home/ace14550vm/.bashrc
python resnet_train.py