#!/usr/bin/bash

#python ../codes/prepare_data.py

for seed in 1 2 3 ; do
  for leave in 7 15 31 63 ; do
    echo $seed $leave
    for lr in 0.01 0.05 0.1 ; do echo $lr ; done | xargs -P3 -I{} python ../codes/predict_model.py $seed 0.75 train_fe.ftr test_fe.ftr --learning_rate {} --num_leaves $leave
  done
done
