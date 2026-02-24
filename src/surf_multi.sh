#!/usr/bin/env bash

python surf_baseline_multi_main.py '/home/roseky/PycharmProjects/datasets/CASIA-SURF/phase1/train' 'average' 'multi_full_' 0 0 --p [1,1,1]
python surf_baseline_multi_main.py '/home/roseky/PycharmProjects/datasets/CASIA-SURF/phase1/train' 'average' 'multi_full_' 0 1 --p [1,1,1]
python surf_baseline_multi_main.py '/home/roseky/PycharmProjects/datasets/CASIA-SURF/phase1/train' 'average' 'multi_full_'  0 2 --p [1,1,1]



