#!/usr/bin/bash


for mul in 0.5 1 1.5 ; do echo $mul ; done | xargs -P6 -I{} python ../predict_code/predict_model.py 1 {} train_fe.ftr test_fe.ftr
