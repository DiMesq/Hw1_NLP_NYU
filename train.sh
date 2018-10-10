#!/bin/bash

# max_vocab = 10000
python train.py 1 10000 20 &> logs/train/train_1_10000_20.out &
python train.py 1 20000 20 &> logs/train/train_1_20000_20.out &
python train.py 1 50000 20 &> logs/train/train_1_50000_20.out &

python train.py 1 50000 50 &> logs/train/train_1_50000_50.out &
python train.py 1 50000 100 &> logs/train/train_1_50000_100.out &

python train.py 2 50000 50 &> logs/train/train_2_50000_50.out &
python train.py 2 50000 100 &> logs/train/train_2_50000_100.out &

python train.py 3 50000 50 &> logs/train/train_3_50000_50.out &
python train.py 3 50000 100 &> logs/train/train_3_50000_100.out &
python train.py 3 50000 300 &> logs/train/train_3_50000_300.out &

python train.py 4 50000 100 &> logs/train/train_4_50000_100.out &
python train.py 4 150000 100 &> logs/train/train_4_150000_100.out &
python train.py 4 50000 300 &> logs/train/train_4_50000_300.out &
