#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import _pickle as cPickle
import argparse
from copy import deepcopy
from datetime import date, datetime, timedelta
import japanize_matplotlib
import lightgbm as lgb
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import time
from tqdm import tqdm

import xgboost
import catboost


# In[2]:


parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('seed', type=int) # 何回ハイパラ探索するか
args = parser.parse_args()#args=['222'])

print(args)


# In[3]:


# pklファイルがある場合(to_pkl.ipynbで生成)

with open('../input/train_raw.pkl', 'rb') as f:
    train = pickle.load(f)


with open('../input/test_raw.pkl', 'rb') as f:
    test = pickle.load(f)

with open('../input/building_metadata_raw.pkl', 'rb') as f:
    building_meta = pickle.load(f)

with open('../input/weather_train_raw.pkl', 'rb') as f:
    weather_train = pickle.load(f)

with open('../input/weather_test_raw.pkl', 'rb') as f:
    weather_test = pickle.load(f)
    
    
# minificationされたデータを用いる場合(Data_minification.ipynbで生成)
# with open('../input/train_mini.pkl', 'rb') as f:
#     train = pickle.load(f)

# with open('../input/test_mini.pkl', 'rb') as f:
#     test = pickle.load(f)

# with open('../input/building_metadata_mini.pkl', 'rb') as f:
#     building_meta = pickle.load(f)

# with open('../input/weather_train_mini.pkl', 'rb') as f:
#     weather_train = pickle.load(f)

# with open('../input/weather_test_mini.pkl', 'rb') as f:
#     weather_test = pickle.load(f)


# pklファイルがない場合

# train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
# test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
# building_meta = pd.read_csv('../input/building_metadata.csv')
# weather_train = pd.read_csv('../input/weather_train.csv', parse_dates=['timestamp'])
# weather_test = pd.read_csv('../input/weather_test.csv', parse_dates=['timestamp'])


# In[4]:


# 覚え書き
# 連続で同じ値を取るやつを除去
# ただし、同じ値を取るやつが最小値だった場合は除去しない(電気データの場合、最小値=休みの日とかの可能性があるため)

del_list = list()

for building_id in range(1449):
    train_gb = train[train['building_id'] == building_id].groupby("meter")

    for meter, tmp_df in train_gb:
        print("building_id: {}, meter: {}".format(building_id, meter))
        data = tmp_df['meter_reading'].values
#         splited_value = np.split(data, np.where((data[1:] != data[:-1]) | (data[1:] == min(data)))[0] + 1)
#         splited_date = np.split(tmp_df.timestamp.values, np.where((data[1:] != data[:-1]) | (data[1:] == min(data)))[0] + 1)
        splited_idx = np.split(tmp_df.index.values, np.where((data[1:] != data[:-1]) | (data[1:] == min(data)))[0] + 1)
        for i, x in enumerate(splited_idx):
            if len(x) > 24:
#                 print("length: {},\t{}-{},\tvalue: {}".format(len(x), x[0], x[-1], splited_value[i][0]))
                del_list.extend(x[1:])
                
                
        print()

del tmp_df, train_gb


# In[5]:


def idx_to_drop(df):
    drop_cols = []
    electric_zero = df[(df['meter']==0)&(df['meter_reading']==0)].index.values.tolist()
    drop_cols.extend(electric_zero)
    not_summer = df[(df['timestamp'].dt.month!=7)&(df['timestamp'].dt.month!=8)]
    not_summer['cumsum'] = not_summer.groupby(['building_id','meter'])['meter_reading'].cumsum()
    not_summer['shifted'] = not_summer.groupby(['building_id','meter'])['cumsum'].shift(48)
    not_summer['difference'] = not_summer['cumsum']-not_summer['shifted']
    steam_zero = not_summer[(not_summer['difference']==0) & (not_summer['meter']==2)].index.values.tolist()
    hotwater_zero = not_summer[(not_summer['difference']==0) & (not_summer['meter']==3)].index.values.tolist()
    drop_cols.extend(steam_zero)
    drop_cols.extend(hotwater_zero)
    del not_summer
    not_winter = train[(df['timestamp'].dt.month!=12)&(df['timestamp'].dt.month!=1)]
    not_winter['cumsum'] = not_winter.groupby(['building_id','meter'])['meter_reading'].cumsum()
    not_winter['shifted'] = not_winter.groupby(['building_id','meter'])['cumsum'].shift(48)
    not_winter['difference'] = not_winter['cumsum']-not_winter['shifted']
    chilled_zero = not_winter[(not_winter['difference']==0) & (not_winter['meter']==1)].index.values.tolist()
    drop_cols.extend(chilled_zero)
    return drop_cols

del_list.extend(idx_to_drop(train))


# In[6]:


del_list_new = train.loc[del_list].index#query('timestamp < 20160901').index


# In[7]:


# 行の削除
train = train.drop(del_list_new)


# In[8]:


train = train.query('(not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")) & (not (meter==0 & meter_reading==0))')
train['meter_reading'] = np.log1p(train['meter_reading'])
train = train.reset_index(drop=True)


# In[9]:


# trainとtestの予測バイアス
# a = pd.read_csv('../output/submission.csv')
# test = test.merge(a,on='row_id',how='left')
# pd.concat([g_tra, g_te], axis=1)


# In[10]:


# g_tra = train.groupby(['building_id', 'meter'])['meter_reading'].mean()
# g_te = test.groupby(['building_id', 'meter'])['meter_reading'].mean()


# In[11]:


weather = pd.concat([weather_train, weather_test], axis=0).reset_index(drop=True)

# dataframeの定義
country = ['UnitedStates', 'England', 'UnitedStates', 'UnitedStates', 'UnitedStates',
           'England', 'UnitedStates', 'Canada', 'UnitedStates', 'UnitedStates',
           'UnitedStates', 'Canada', 'Ireland', 'UnitedStates', 'UnitedStates', 'UnitedStates']

city = ['Jacksonville', 'London', 'Phoenix', 'Philadelphia', 'San Francisco',
       'Loughborough', 'Philadelphia', 'Montreal', 'Jacksonville', 'San Antonio',
       'Las Vegas', 'Montreal', 'Dublin', 'Minneapolis', 'Philadelphia', 'Pittsburgh']

UTC_offset = [-4, 0, -7, -4, -9, 0, -4, -4, -4, -5, -7, -4, 0, -5, -4, -4]

location_data = pd.DataFrame(np.array([country, city, UTC_offset]).T, index=range(16), columns=['country', 'city', 'UTC_offset'])


# timestampの補正
for idx in location_data.index:
    weather.loc[weather['site_id']==idx, 'timestamp'] += timedelta(hours=int(location_data.loc[idx, 'UTC_offset']))


# In[12]:


def fill_weather_dataset(weather_df):
    
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(weather_df['timestamp'].min(),time_format)
    end_date = datetime.strptime(weather_df['timestamp'].max(),time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    missing_hours = []
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df,new_rows])

        weather_df = weather_df.reset_index(drop=True)           

    # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month
    
    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id','day','month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])
    weather_df.update(air_temperature_filler,overwrite=False)

    # Step 1
    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()
    # Step 2
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])

    weather_df.update(cloud_coverage_filler,overwrite=False)

    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])
    weather_df.update(due_temperature_filler,overwrite=False)

    # Step 1
    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    # Step 2
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])

    weather_df.update(sea_level_filler,overwrite=False)

    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])
    weather_df.update(wind_direction_filler,overwrite=False)

    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])
    weather_df.update(wind_speed_filler,overwrite=False)

    # Step 1
    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    # Step 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler,overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)
        
    return weather_df

weather['timestamp'] = weather['timestamp'].astype(str)
weather = fill_weather_dataset(weather)
weather['timestamp'] = pd.to_datetime(weather['timestamp'])


# ### 休日情報

# In[13]:


import holidays

en_holidays = holidays.England()
ir_holidays = holidays.Ireland()
ca_holidays = holidays.Canada()
us_holidays = holidays.UnitedStates()

en_idx = weather.query('site_id == 1 or site_id == 5').index
ir_idx = weather.query('site_id == 12').index
ca_idx = weather.query('site_id == 7 or site_id == 11').index
us_idx = weather.query('site_id == 0 or site_id == 2 or site_id == 3 or site_id == 4 or site_id == 6 or site_id == 8 or site_id == 9 or site_id == 10 or site_id == 13 or site_id == 14 or site_id == 15').index

weather['IsHoliday'] = 0
weather.loc[en_idx, 'IsHoliday'] = weather.loc[en_idx, 'timestamp'].apply(lambda x: en_holidays.get(x, default=0))
weather.loc[ir_idx, 'IsHoliday'] = weather.loc[ir_idx, 'timestamp'].apply(lambda x: ir_holidays.get(x, default=0))
weather.loc[ca_idx, 'IsHoliday'] = weather.loc[ca_idx, 'timestamp'].apply(lambda x: ca_holidays.get(x, default=0))
weather.loc[us_idx, 'IsHoliday'] = weather.loc[us_idx, 'timestamp'].apply(lambda x: us_holidays.get(x, default=0))

holiday_idx = weather['IsHoliday'] != 0
weather.loc[holiday_idx, 'IsHoliday'] = 1
weather['IsHoliday'] = weather['IsHoliday'].astype(np.uint8)


# In[14]:


target = train['meter_reading'].values
# train = train.drop('meter_reading', axis=1)
row_id = test['row_id']
test = test.drop('row_id', axis=1)

df = pd.concat([train.drop('meter_reading', axis=1), test], axis=0).reset_index(drop=True)
df = df.merge(building_meta, on='building_id', how='left')

df = df.merge(weather, on=['site_id', 'timestamp'], how='left')

df['day'] = df['timestamp'].dt.day #// 3
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.weekday

train = df.iloc[:len(target)].copy().reset_index(drop=True)
train['meter_reading'] = target#.values
test = df.iloc[len(target):].copy().reset_index(drop=True)


# ### lag feature

# precip_depth_1_hr, wind_direction
# 
# 昨日、今日、明日のmax, 中央値,75%, 25%分位点,平均, std
# 
# 18属性 * 2つ

# In[15]:


# col1 = ['wind_speed']

# g1 = weather.groupby(['site_id', weather['timestamp'].dt.date])[col1].agg(['min', 'max', 'median'])
# g1.columns = list(map(lambda s: s[0]+'_'+s[1], zip(g1.columns.get_level_values(0), g1.columns.get_level_values(1))))

# # for col in col1 :
# #     g1 = g1.drop(['{}_count'.format(col), '{}_min'.format(col)], axis=1)

# tmp = pd.concat([g1, g1.shift().add_suffix('_shift1')], axis=1)
# tmp = pd.concat([tmp, g1.shift(-1).add_suffix('_shift-1')], axis=1)
# g1 = tmp
# del tmp


# 気温(air_temperature, dew_temperature)
# 
# 昨日→今日
# 今日→明日
# 昨日→明日
# の<br>
# maxとminの差
# 
# 昨日、今日、明日の<br>
# 温度差絶対値、mean
# 
# 9属性*２つ

# In[16]:


# col2 = ['air_temperature', 'dew_temperature']
# g2 = weather.groupby(['site_id', weather['timestamp'].dt.date])[col2].agg(['min', 'std', 'mean', 'max'])
# g2.columns = list(map(lambda s: s[0]+'_'+s[1], zip(g2.columns.get_level_values(0), g2.columns.get_level_values(1))))

# for col in col2:
#     g2['{}_range'.format(col)] = g2['{}_max'.format(col)] - g2['{}_min'.format(col)]
    
#     g2['{}_absdiff_1'.format(col)] = g2.groupby('site_id')['{}_mean'.format(col)].diff().abs() # 昨日→今日のabsdiff
#     g2['{}_absdiff_-1'.format(col)] = g2.groupby('site_id')['{}_mean'.format(col)].diff(-1).abs() # 今日→明日のabsdiff
#     g2['{}_absdiff_1to-1'.format(col)] = g2['{}_absdiff_1'.format(col)] + g2['{}_absdiff_-1'.format(col)]
    
#     g2['{}_range_shift1'.format(col)] = g2['{}_range'.format(col)].groupby('site_id').shift()
#     g2['{}_range_shift-1'.format(col)] = g2['{}_range'.format(col)].groupby('site_id').shift(-1)
    
#     g2['{}_mean_shift1'.format(col)] = g2['{}_mean'.format(col)].groupby('site_id').shift()
#     g2['{}_mean_shift-1'.format(col)] = g2['{}_mean'.format(col)].groupby('site_id').shift(-1)


# In[17]:


# df['date'] = df['timestamp'].dt.date
# df = df.merge(g1, left_on=['site_id', 'date'], right_on=['site_id', 'timestamp'], how='left')
# df = df.merge(g2, left_on=['site_id', 'date'], right_on=['site_id', 'timestamp'], how='left')
# del df['date']


# In[18]:


def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in cols:
        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
        weather_df[f'{col}_max_lag{window}'] = lag_max[col]
        weather_df[f'{col}_min_lag{window}'] = lag_min[col]
        weather_df[f'{col}_std_lag{window}'] = lag_std[col]


# In[19]:


# add_lag_feature(weather)


# ### 時間関連

# In[20]:


def make_circulation_feature(prefix, range_max, range_min=0, plot=False):
    # 周期性(?)のある特徴を円周上に配置して特徴量を作る

    circular_info = pd.DataFrame(index=range(range_min, range_max+1))
    circular_info['%s_posX' % prefix] = np.sin(2 * np.pi * ((circular_info.index - range_min) / len(circular_info)))
    circular_info['%s_posY' % prefix] = np.cos(2 * np.pi * ((circular_info.index - range_min) / len(circular_info)))
    
    if plot:
        fig,ax = plt.subplots(figsize=(4,4))

        ax.scatter(circular_info['%s_posX' % prefix],circular_info['%s_posY' % prefix])
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        # 下の二行で各点に番号をつける
        for d,x,y in zip(circular_info.index, circular_info['%s_posX' % prefix], circular_info['%s_posY' % prefix]):
            ax.annotate(d,(x,y), size=13)
        
    return circular_info


# In[21]:


# day_r = make_circulation_feature('day', range_min=1, range_max=31)
# hour_r = make_circulation_feature('hour', range_min=0, range_max=23)
# weekday_r = make_circulation_feature('weekday', range_min=0, range_max=6)

# df = df.merge(day_r, left_on='day', right_index=True, how='left')
# df = df.merge(hour_r, left_on='hour', right_index=True, how='left')
# df = df.merge(weekday_r, left_on='weekday', right_index=True, how='left')


# In[22]:


# holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
#             "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
#             "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
#             "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
#             "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
#             "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
#             "2019-01-01"]

# df['is_holiday'] = df.timestamp.dt.date.astype("str").isin(holidays)
df['is_day_off_or_holiday'] = (df['weekday'] >= 5) | df['IsHoliday']


# ### 風

# In[23]:


# 循環特徴量にする場合
# wind_direction_r = make_circulation_feature('wind_direction', range_min=1, range_max=360)
# df = df.merge(wind_direction_r, left_on='wind_direction', right_index=True, how='left')

# categoricalにする場合
# df['wind_direction_cat'] = (df['wind_direction'] + 22.5) % 360 // 45


# ### target encoding

# In[24]:


def make_fraction(col1, col2):
    col2_frac = train.groupby([col1, col2])[['meter_reading']].median()
    col2_frac_idx = col2_frac.index
    col2_frac_sum = col2_frac.groupby(col1).sum().rename(columns = {'meter_reading':'sum'})
    col2_frac = col2_frac.merge(col2_frac_sum, on = col1, how='left')
    col2_frac.index = col2_frac_idx
    col2_frac['frac_{}_{}'.format(col1, col2)] = col2_frac['meter_reading'] / col2_frac['sum']
    col2_frac = col2_frac[['frac_{}_{}'.format(col1, col2)]]
    return col2_frac


# * fraction

# In[25]:


building_weekday_frac = make_fraction('building_id', 'weekday')
building_hour_frac = make_fraction('building_id', 'hour')
building_day_frac = make_fraction('building_id', 'day')

primary_weekday_frac = make_fraction('primary_use', 'weekday')
primary_hour_frac = make_fraction('primary_use', 'hour')
primary_day_frac = make_fraction('primary_use', 'day')


# In[26]:


df = df.merge(building_weekday_frac, on=['building_id', 'weekday'], how='left')
df = df.merge(building_hour_frac, on=['building_id', 'hour'], how='left')
df = df.merge(building_day_frac, on=['building_id', 'day'], how='left')

df = df.merge(primary_weekday_frac, on=['primary_use', 'weekday'], how='left')
df = df.merge(primary_hour_frac, on=['primary_use', 'hour'], how='left')
df = df.merge(primary_day_frac, on=['primary_use', 'day'], how='left')


# * median

# In[27]:


# df['median_building_id_weekday'] = train.groupby(['building_id', 'weekday'])['meter_reading'].transform('median')
# df['median_building_id_hour'] = train.groupby(['building_id', 'hour'])['meter_reading'].transform('median')
# df['median_building_id_day'] = train.groupby(['building_id', 'day'])['meter_reading'].transform('median')
# df['median_primary_use_weekday'] = train.groupby(['primary_use', 'weekday'])['meter_reading'].transform('median')
# df['median_primary_use_hour'] = train.groupby(['primary_use', 'hour'])['meter_reading'].transform('median')
# df['median_primary_use_day'] = train.groupby(['primary_use', 'day'])['meter_reading'].transform('median')

# df = df.drop(['median_building_id_weekday',
#       'median_building_id_hour',
#       'median_building_id_day',
#       'median_primary_use_weekday',
#       'median_primary_use_hour',
#       'median_primary_use_day'], axis=1)


# * wind_direction(frac)

# In[28]:


# train['wind_direction_div10'] = train['wind_direction']//10 % 36
# df['wind_direction_div10'] = df['wind_direction']//10 % 36
# site_and_winddir = train.groupby(['wind_direction_div10', 'site_id'])['meter_reading'].median().rename('site_and_winddir_median')

# df = df.merge(site_and_winddir, on=['wind_direction_div10', 'site_id'], how='left')
# del train['wind_direction_div10'], df['wind_direction_div10']


# In[29]:


# # 建物ごとの平均
# building_meter_average = train.groupby(['building_id', 'meter'])['meter_reading'].mean().rename('building_meter_average')
# df = df.merge(building_meter_average, on=['building_id', 'meter'], how='left')


# In[30]:


# 建物ごとの平均
building_meter_95 = train.groupby(['building_id', 'meter'])['meter_reading'].apply(lambda arr: np.percentile(arr, 95)).rename('building_meter_95')
df = df.merge(building_meter_95, on=['building_id', 'meter'], how='left')

# 建物ごとの平均
building_meter_5 = train.groupby(['building_id', 'meter'])['meter_reading'].apply(lambda arr: np.percentile(arr, 5)).rename('building_meter_5')
df = df.merge(building_meter_5, on=['building_id', 'meter'], how='left')


# In[31]:


# # minmaxscalingして予測したい場合
# building_meter_95 = train.groupby(['building_id', 'meter'])['meter_reading'].apply(lambda arr: np.percentile(arr, 95)).rename('building_meter_95')
# building_meter_95 += 0.5
# train = train.merge(building_meter_95, on=['building_id', 'meter'], how='left')

# train['meter_reading'] /= train['building_meter_95']
# target = train['meter_reading'].values


# ### 一部属性をカテゴリカル変数に変換

# In[32]:


is_categorical = ['meter', 'building_id', 'site_id', 'primary_use', 'hour', 'day', 'weekday']
df[is_categorical] = df[is_categorical].astype('category')
df['year_built_cat'] = df['year_built'].astype('category')


# In[33]:


drop_columns = []#, 'hour', 'day', 'weekday']
drop_df = df[drop_columns]
df = df.drop(drop_columns, axis=1)


# In[34]:


# train = df.iloc[:len(target)].copy().reset_index(drop=True)
# train['meter_reading'] = target#.values
# df = df.merge(train.groupby(['building_id','weekday'])['meter_reading'].agg(['mean', 'median']), on=['building_id','weekday'], how='left')


# In[35]:


train_fe = df.iloc[:len(target)].copy().reset_index(drop=True)
train_fe['meter_reading'] = target#.values
test_fe = df.iloc[len(target):].copy().reset_index(drop=True)
# train_fe = train_fe.query('~(meter==0 & meter_reading==0)')
target_fe = train_fe['meter_reading']
train_fe = train_fe.drop('meter_reading', axis=1)


# In[36]:


# train_fe_all = df.iloc[:len(target)].copy()
# train_fe_all['meter_reading'] = target#.values

# test_fe_all = df.iloc[len(target):].copy()
# test_fe_all['row_id'] = row_id.values

# with open('../input/train_fe_all.zip', 'wb') as f:
#     pickle.dump(train_fe_all, f)

# with open('../input/test_fe_all_2017.zip', 'wb') as f:
#     pickle.dump(test_fe_all.query('timestamp < 20180101'), f)
    
# with open('../input/test_fe_all_2018.zip', 'wb') as f:
#     pickle.dump(test_fe_all.query('20180101 <= timestamp'), f)


# In[37]:


# drop_feature = ['day', 'hour', 'weekday', 'year_built_cat', 'wind_direction']
# df_dropped = df[drop_feature].copy()
# df = df.drop(drop_feature, axis=1)


# In[38]:


# a = pd.concat([X_train, y_train], axis=1).groupby('building_id')['meter_reading']
# b = pd.concat([X_valid, y_valid], axis=1).groupby('building_id')['meter_reading']

# pd.concat([a.median().iloc[105:], b.median().iloc[105:]], axis=1)

# pd.concat([a.describe().iloc[105:], b.describe().iloc[105:]], axis=1)


# In[39]:


X_train = train_fe.query('20160115 <= timestamp < 20160601 & site_id != 0')
X_valid = train_fe.query('20160901 <= timestamp < 20170101 & site_id != 0')
X_test = test_fe

y_train = target_fe.loc[X_train.index]
y_valid = target_fe.loc[X_valid.index]
# y_train = np.log1p(y_train)
# y_valid = np.log1p(y_valid)

X_train = X_train.drop('timestamp', axis=1)
X_valid = X_valid.drop('timestamp', axis=1)
X_test = X_test.drop('timestamp', axis=1)

print(X_train.shape)


# In[40]:


def meter_fit(meter, X_train, X_valid, y_train, y_valid, n_estimators=20000, verbose=5000, random_state=823, **params):
    model = lgb.LGBMRegressor(random_state=random_state, n_estimators=n_estimators, n_jobs=4, metric='rmse', **params)

    X_train_m = X_train.query('meter == {}'.format(meter)).drop('meter', axis=1)
    X_valid_m = X_valid.query('meter == {}'.format(meter)).drop('meter', axis=1)
    y_train_m = y_train[X_train_m.index]
    y_valid_m = y_valid[X_valid_m.index]
    
    g = X_valid_m.groupby('building_id')
    eval_names = ['train', 'valid']
    eval_set = [(X_train_m, y_train_m), (X_valid_m, y_valid_m)]
#     print(sorted(X_valid_m['building_id'].unique()))
    for i in tqdm(sorted(X_valid_m['building_id'].unique())):
        set_evalX = g.get_group(i)
        eval_set.append((set_evalX, y_valid_m.loc[set_evalX.index]))
        eval_names.append(i)

# building_idを抜いて実験する場合
#     for X, y in eval_set:
#         del X['building_id']
        
    print(X_train_m.shape)
    
    model.fit(X_train_m, y_train_m , eval_set = eval_set,#[(X_train_m, y_train_m), (X_valid_m, y_valid_m)], 
                    eval_names=eval_names, verbose=verbose)#, early_stopping_rounds = 100)
    return model


def meter_fit_all(meter, X_train, y_train, n_estimators, random_state=823, **params):
    print(n_estimators)
    X_train_m = X_train.query('meter == {}'.format(meter)).drop('meter', axis=1)
    y_train_m = y_train[X_train_m.index]
    
    print("meter{}".format(meter), end='')
    model = lgb.LGBMRegressor(random_state=random_state, n_estimators=n_estimators, n_jobs=4, metric='rmse', **params)
    model.fit(X_train_m,y_train_m,
             eval_set = [(X_train_m, y_train_m)], 
                    verbose=10000)
    print(' done')
    return model

# def meter_predict(meter, model, X_test):
#     X_test_m = X_test.query('meter == {}'.format(meter)).drop('meter', axis=1)
#     y_pred = model.predict(X_test_m)
#     return pd.Series(y_pred, index=X_test_m.index)


def meter_predict(meter, model, X_test, best_iteration, iteration_mul=1.5):
    X_test_m = X_test.query('meter == {}'.format(meter)).drop('meter', axis=1)
    g = X_test_m.groupby('building_id')
    
    y_pred = []
    for building_id in tqdm(sorted(X_test_m['building_id'].unique())):
        X_building = g.get_group(building_id)
        y_pred.append(pd.Series(model.predict(X_building, num_iteration=min(models_all[meter].n_estimators, int(best_iteration[meter][building_id]*iteration_mul))), index=X_building.index))
        
    return pd.concat(y_pred).sort_index()


# In[41]:


# meter=0
# StratifiedKFold_params = {
# #                     'boosting_type':'gbdt',
#                     'learning_rate':0.01, #for faster training
# #                     'num_leaves': 2**8,
# #                     'max_depth':-1,
# #                     'colsample_bytree': 0.7,
# #                     'subsample_freq':1,
# #                     'subsample':0.7,
#                 }

# X_train_m = X_train.query('meter == {}'.format(meter)).drop('meter', axis=1)#.reset_index(drop=True)
# X_valid_m = X_valid.query('meter == {}'.format(meter)).drop('meter', axis=1)#.reset_index(drop=True)
# y_train_m = y_train.loc[X_train_m.index]
# y_valid_m = y_valid.loc[X_valid_m.index]

# NFOLDS = 3
# folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, 
#                         random_state=823)

# oof_preds_lgb = np.zeros((len(X_train_m), 1))
# y_pred_lgb = np.zeros((NFOLDS, len(X_valid_m)))
# rmsle_lgb =[]

# # models_f = [lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=4, **params) for _ in range(NFOLDS)]
# models_f = dict()
# for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train_m, X_train_m['building_id'])):
#     print('Fold', fold_n, 'started at', time.ctime())
#     X_train_f, X_valid_f = X_train_m.iloc[train_index, :], X_train_m.iloc[valid_index, :]
#     y_train_f, y_valid_f = y_train_m.loc[X_train_f.index], y_train_m.loc[X_valid_f.index]
    
    
#     models_f[fold_n] = lgb.LGBMRegressor(random_state=823, n_estimators=10000, n_jobs=4, metric='rmse', **StratifiedKFold_params)
#     models_f[fold_n].fit(X_train_f, y_train_f , eval_set = [(X_train_f, y_train_f), (X_valid_f, y_valid_f), (X_valid_m, y_valid_m)], 
#                                  verbose=50, early_stopping_rounds = 100)
# #     meter_fit(i, X_train_f, X_valid_f, y_train_f, y_valid_f, **StratifiedKFold_params)
    
#     val_pred = models_f[fold_n].predict(X_valid_f)#.values
#     oof_preds_lgb[valid_index] = val_pred.reshape((-1, 1))
    
#     y_pred_lgb[fold_n] = models_f[fold_n].predict(X_valid_m)#.values
# # y_pred_lgb = y_pred_lgb.reshape(-1)

# print('total_rmsle = {}'.format(np.sqrt(mean_squared_error(y_valid_m, y_pred_lgb.mean(axis=0)))))


# In[42]:


lgb_params = {
#                     'boosting_type':'gbdt',
                    'learning_rate':0.05, 
                    'num_leaves': 31,
#                     'max_depth':-1,
                    'colsample_bytree': 0.8,
                    'subsample_freq':1,
                    'subsample':0.8,
                }


# In[43]:


with open('../model/model_5_95_hokan_cleaning_50000tree_seed1.pkl', 'rb') as f:
    models = pickle.load(f)


# In[45]:


# # meter type毎に訓練
# models = dict()
# for i in [3,2,1,0]:
#     print('meter {} start at {}'.format(i,time.ctime()))
#     models[i] = meter_fit(i, X_train, X_valid, y_train, y_valid, random_state=args.seed, n_estimators=50000, **lgb_params)
    
# # rmseの計算
# meter_counts = X_valid['meter'].value_counts()
# mse_score = 0
# for i in [0,1,2,3]:
#     mse_score += meter_counts[i] * (models[i].best_score_['valid']['rmse'] ** 2)
# print('total rmse = {}'.format(np.sqrt(mse_score / meter_counts.sum())))


# In[46]:


# with open('../model/model_5_95_hokan_cleaning_50000tree_seed{}.pkl'.format(args.seed), 'wb') as f:
#     pickle.dump(models, f)


# In[47]:


# 各building, meter毎の最良のiteration数
best_iteration = dict()
for meter in [0,1,2,3]:
    best_iteration[meter] = dict()
#     for i in range(1448):
#         best_iteration[meter][i] = 200
    for i in tqdm(sorted(X_valid.query('meter == {}'.format(meter))['building_id'].unique())):
        best_iteration[meter][i] = max(20, np.argmin(np.array(models[meter].evals_result_[i]['rmse'])) + 1)
#         best_iteration[meter][i] = np.argmin(np.array(models[meter].evals_result_[i]['rmse'])) + 1


# In[48]:


# 2つとも抜いた
# 全期間抜いた
# 50000本
# leave31
# lr0.01
best_scores = dict()
for meter in [0,1,2,3]:
    best_scores[meter] = 0
    meter_size = X_valid.query('meter=={}'.format(meter)).groupby('building_id').size()
    meter_size = meter_size[meter_size!=0]
    for buildingID, cnt in meter_size.items():
        best_scores[meter] += cnt * (min(models[meter].evals_result_[buildingID]['rmse']) ** 2)
    best_scores[meter] = np.sqrt(best_scores[meter] / meter_size.sum())
    

# rmseの計算
meter_counts = X_valid['meter'].value_counts()
mse_score = 0
for i in [0,1,2,3]:
    mse_score += meter_counts[i] * (best_scores[i] ** 2)
print('total rmse = {}'.format(np.sqrt(mse_score / meter_counts.sum())))
print(best_scores)
for meter in [0,1,2,3]:
    print('meter{} best_valid_iteration={}'.format(meter,np.argmin(np.array(models[meter].evals_result_['valid']['rmse']))+1))
    print('meter{} best_valid_score={}'.format(meter,np.min(np.array(models[meter].evals_result_['valid']['rmse'])+1)))
    plt.subplot(2,2,meter+1)
    plt.hist(best_iteration[meter].values())
plt.show()


# In[49]:


# for meter in [0,1,2,3]:
#     print(models[meter].best_score_['valid'])


# In[50]:


del_list = [list(), list(), list(), list()]
for meter in [0,1,2,3]:
    for buildingID, itr in best_iteration[meter].items():
        if itr<=20:
            del_list[meter].append(buildingID)
        if itr<=100:
            best_iteration[meter][buildingID] = 100
#         if itr>=int(models[0].n_estimators * 0.98):
#             best_iteration[meter][buildingID] = models[0].n_estimators
            
new_train_fe = train_fe.copy()
new_train_fe['meter_reading'] = target
for meter in [0,1,2,3]:
    new_train_fe = new_train_fe.query('~(meter=={} & building_id == {} & timestamp<20160601)'.format(meter, del_list[meter]))

new_target = new_train_fe['meter_reading']
new_train_fe = new_train_fe.drop('meter_reading', axis=1)


# In[51]:


for meter in [0,1,2,3]:
    for i in range(1448):
        if i not in best_iteration[meter]:
            best_iteration[meter][i] = 200


# In[52]:


# for i in models[3].evals_result_.keys():
#     plt.plot(models[3].evals_result_[i]['rmse'])
#     plt.title(i)
#     plt.show()


# In[53]:


import gc


# In[54]:


gc.collect()


# In[56]:


# meter type毎に訓練(全てのデータを使う)
models_all = dict()
for i in [3, 2, 1, 0]:
    print('meter {} start at {}'.format(i,time.ctime()))
    models_all[i] = meter_fit_all(i, new_train_fe.drop('timestamp', axis=1), new_target, random_state=args.seed, n_estimators=50000, **lgb_params)


# In[57]:


with open('../model/model_all_5_95_hokan_cleaning_50000tree_seed{}.pkl'.format(args.seed), 'wb') as f:
    pickle.dump(models_all, f)


# In[58]:


# with open('../model/models_all_300000tree.pkl', 'wb') as f:
#     pickle.dump(models_all, f)


# In[59]:


# with open('../model/models_all_50000tree.pkl', 'rb') as f:
#     models_all = pickle.load(f)


# In[60]:


# meter type毎のtestの予測    
preds = list()
for i in tqdm([3,2,1,0]):
    preds.append(meter_predict(i, models_all[i], X_test, best_iteration, iteration_mul=1))

y_preds = pd.concat(preds).sort_index()


# In[61]:


# np.sqrt(((0.418693**2 * 3014164) + (1.4395**2 * 1081665) + (1.58482**2 * 701454) + (1.75093**2 * 319097)) / len(X_valid))


# In[62]:


# np.sqrt(((0.453832**2 * 3014164) + (1.47836**2 * 1081665) + (1.58737**2 * 701454) + (1.75538**2 * 319097)) / len(X_valid))


# In[63]:


# model = lgb.LGBMRegressor(learning_rate=0.1, random_state=823, n_estimators=1000, n_jobs=4, metric='rmse')
# print("=" * 50)
# model.fit(X_train,y_train, eval_set = [(X_train,y_train), (X_valid,y_valid)], 
#                 verbose=20, early_stopping_rounds = 50)

# y_preds = model.predict(X_test)
# print(y_preds.max())


# In[64]:


lgb.plot_importance(models[0], importance_type='gain', figsize=(10,20))


# In[65]:


lgb.plot_importance(models_all[0], importance_type='split', figsize=(10,20))


# In[66]:


lgb.plot_importance(models[0], importance_type='split', figsize=(10,25))


# In[67]:


submission = pd.read_csv('../input/sample_submission.csv')
submission['meter_reading'] = (np.expm1(y_preds))
submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0


# In[68]:


submission.to_csv('../output/submission_5_95_hokan_cleaning_100000tree_seed{}.csv'.format(args.seed), index=False)


# In[ ]:


submission

