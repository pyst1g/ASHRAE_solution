#!/usr/bin/bash

#python ../codes/prepare_data.py

python ../codes/train_model.py 1 train_fe.ftr test_fe.ftr --learning_rate 0.01 --num_leaves 15 --n_estimators 20000
python ../codes/predict_model.py 1 0.5 train_fe.ftr test_fe.ftr --learning_rate 0.01 --num_leaves 15 --n_estimators 20000
