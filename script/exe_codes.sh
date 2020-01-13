#!/usr/bin/bash


#python ../codes/prepare_data.py

for seed in 1 2 3 ; do
  for lr in 0.01 0.05 0.1 ; do
    for leave in 7 15 31 63 ; do
      echo $seed $lr $leave
      python ../codes/train_model.py $seed train_fe.ftr test_fe.ftr --learning_rate $lr --num_leaves $leave
      python ../codes/predict_model.py $seed 0.5 train_fe.ftr test_fe.ftr --learning_rate $lr --num_leaves $leave
    done
  done
done
