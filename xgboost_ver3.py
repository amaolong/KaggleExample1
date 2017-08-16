'''

mean logerror weighted (logerror mean/monthly mean) linear xgb implementation

# version 1
no outlier removal

# version 2 (score 0.1493868)
do have N/A filling (with fillna, and maxdepth:4, score 0.1049204)
max depths change to 10 from 4
seems to be overfitting training data, validation is around 0.0678 while training is around 0.0623

# version 3
do have N/A filling

(score 0.1259109)
params['max_depth'] = 6
params['silent'] = 1
params['subsample']=0.8
params['colsample_bytree']=0.8

(0.1049204)
params['max_depth'] = 4
params['silent'] = 1
params['sample_type']='weighted'

() need to go back to origin params to see what's going on
params['max_depth'] = 4
params['silent'] = 1
no weights

one of the problem here is the generalization issue.
maybe the fillna using median is not a good idea
also try to think about the extreme valued data


# ideas
1) handle outliers
2) how weight should be optimized
3) best feature set to use
4) prediction model to use
5) using all training data for prediction

'''

import numpy as np
import pandas as pd
import xgboost as xgb
import gc


''' load data '''
train = pd.read_csv('../input/train_2016_v2.csv',parse_dates=['transactiondate'])
prop = pd.read_csv('../input/properties_2016.csv')
sample = pd.read_csv('../input/sample_submission.csv')

print('Binding to float32')
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')
df_train = train.merge(prop, how='left', on='parcelid')
df_train.fillna(df_train.median(),inplace = True)       # fillna
df_train['transaction_month'] = df_train['transactiondate'].dt.month

''' set upper and lower percentile limit using 0.05 '''
ulimit = np.percentile(df_train .logerror.values, 99.95)    # get percentile value
llimit = np.percentile(df_train .logerror.values, 0.05)     # get percentile value
df_train ['logerror'].ix[df_train ['logerror']>ulimit] = ulimit       # set dataframe.ix[index] = value
df_train ['logerror'].ix[df_train ['logerror']<llimit] = llimit       # set dataframe.ix[index] = value

'''set weight'''
weight_dict={}
avg_logerror=df_train['logerror'].mean()
for i in range(1,13):
    weight_dict[i]=avg_logerror/(df_train['logerror'].ix[df_train['transaction_month']==i].mean())
weight_vector=[weight_dict[i] for i in df_train['transaction_month'].values]

y_train = df_train['logerror'].values
''' drop un-used columns'''
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'transaction_month', 'assessmentyear',
                         'airconditioningtypeid','basementsqft', 'architecturalstyletypeid', 'buildingclasstypeid',
                         'buildingqualitytypeid','calculatedbathnbr', 'decktypeid', 'finishedfloor1squarefeet', 'fips',
                         'heatingorsystemtypeid', 'rawcensustractandblock','numberofstories', 'storytypeid',
                         'threequarterbathnbr', 'typeconstructiontypeid', 'unitcnt', 'censustractandblock',
                         'fireplaceflag', 'taxdelinquencyflag', 'taxdelinquencyyear','fullbathcnt',
                         'finishedsquarefeet6', 'finishedsquarefeet12', 'finishedsquarefeet13',
                         'finishedsquarefeet15', 'finishedsquarefeet50','propertycountylandusecode',
                         'propertylandusetypeid', 'propertyzoningdesc','pooltypeid10', 'pooltypeid2', 'pooltypeid7',
                         'poolsizesum', 'poolcnt','regionidzip', 'regionidneighborhood', 'regionidcity', 'regionidcounty'], axis=1)
print(x_train.shape, y_train.shape)
train_columns = x_train.columns

''' convert to boolean if possible'''
for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

''' train/valid split '''
split = 80000
x_train, y_train, x_valid, y_valid, train_weight, valid_weight = \
    x_train[:split], y_train[:split], x_train[split:], y_train[split:], weight_vector[:split], weight_vector[split:]

print('Building DMatrix...')
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)


del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1


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