'''

mean logerror weighted (logerror mean/monthly mean) linear xgb implementation

# version 1 (score 0.0653678)
no outlier removal, no N/A filling

# ideas
1) best feature set to use
2) blank filling
3) prediction model to use

'''

import numpy as np
import pandas as pd
import xgboost as xgb
import gc



train = pd.read_csv('../input/train_2016_v2.csv',parse_dates=['transactiondate'])
prop = pd.read_csv('../input/properties_2016.csv')
sample = pd.read_csv('../input/sample_submission.csv')

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')
df_train = train.merge(prop, how='left', on='parcelid')
df_train['transaction_month'] = df_train['transactiondate'].dt.month

'''set weight'''
weight_dict={}
avg_logerror=df_train['logerror'].mean()
for i in range(1,13):
    weight_dict[i]=avg_logerror/(df_train['logerror'].ix[df_train['transaction_month']==i].mean())
weight_vector=[weight_dict[i] for i in df_train['transaction_month'].values]

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                         'propertycountylandusecode','transaction_month'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 80000
x_train, y_train, x_valid, y_valid, train_weight, valid_weight = \
    x_train[:split], y_train[:split], x_train[split:], y_train[split:], weight_vector[:split], weight_vector[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train, weight=train_weight)
d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=valid_weight)

del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1
#params['sample_type']='weighted'

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

del d_train, d_valid

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')

del prop; gc.collect()

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

del df_test, sample; gc.collect()

d_test = xgb.DMatrix(x_test)

del x_test; gc.collect()

print('Predicting on test ...')

p_test = clf.predict(d_test)

del d_test; gc.collect()

sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

print('Writing csv ...')
sub.to_csv('xgb_starter.csv', index=False, float_format='%.4f')