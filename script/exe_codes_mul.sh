#!/usr/bin/bash

#python ../codes/prepare_data.py

for seed in 1 2 3 ; do echo $seed ; done | xargs -P6 -I{} python ../codes/train_model.py {} train_fe.ftr test_fe.ftr

for seed in 1 2 3 ; do
  for mul in 0.5 1 1.5 ; do echo $mul ; done | xargs -P6 -I{} python ../codes/predict_model.py $seed {} train_fe.ftr test_fe.ftr
done
