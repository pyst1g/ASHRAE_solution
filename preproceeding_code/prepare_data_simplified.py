#!/usr/bin/env python
# coding: utf-8

import pandas as pd

train_fe = pd.read_feather('../prepare_data/train_fe.ftr')
test_fe = pd.read_feather('../prepare_data/test_fe.ftr')

remain_features_train = ['building_id', 'meter', 'timestamp', 'air_temperature', 
                   'building_meter_5', 'building_meter_95', 
                   'day', 'hour', 'weekday',
                  'site_id', 'frac_building_id_hour', 'meter_reading']

remain_features_test = ['building_id', 'meter', 'timestamp', 'air_temperature', 
                   'building_meter_5', 'building_meter_95', 
                   'day', 'hour', 'weekday',
                  'site_id', 'frac_building_id_hour']

train_fe = train_fe.loc[:,remain_features_train]
test_fe = test_fe.loc[:,remain_features_test]

train_fe.to_feather('../prepare_data/train_fe_simplified.ftr')
test_fe.to_feather('../prepare_data/test_fe_simplified.ftr')
