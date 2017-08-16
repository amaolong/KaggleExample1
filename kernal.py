''' general import stuff '''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function   # print function in python 3.x
import xlrd # read spreadsheet
import numpy as np
import sys
import os
import argparse
import csv
from random import randint
import pickle

''' skicit learn ML package '''
from sklearn.svm import SVC
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import BaggingRegressor as BR
from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression as LogR
from sklearn.linear_model import Lasso as LS
from sklearn.linear_model import Ridge as RG
from sklearn.linear_model import LinearRegression as LR

from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.model_selection import cross_val_score as CVS
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import KFold as KF
from sklearn.model_selection import GridSearchCV as GSCV

from sklearn.feature_selection import RFECV
from sklearn.metrics import *
import xgboost as xgb

'''visual output'''
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

'''data frame'''
import pandas as pd


'''python version of command line'''
from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))


train_df = pd.read_csv('../input/train_2016_v2.csv', parse_dates=["transactiondate"])      # parse_dates


train_df.shape      # show data frame shapes
train_df.head()     # show top few data line


# plotting
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))    # scatter plot (x, y, ...)
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.show()


'''limit logerror values with top/bottom 1% percentile'''
ulimit = np.percentile(train_df.logerror.values, 99.95)    # get percentile value
llimit = np.percentile(train_df.logerror.values, 0.05)     # get percentile value
train_df['logerror'].ix[train_df['logerror']>ulimit] = ulimit       # set dataframe.ix[index] = value
train_df['logerror'].ix[train_df['logerror']<llimit] = llimit       # set dataframe.ix[index] = value






train_df['transaction_month'] = train_df['transactiondate'].dt.month        # datetime.dt.xyz  return xyz category (year/month/day/time/hour/minute/second/...) of datetime object
train_df['transaction_year']=train_df['transactiondate'].dt.year

'''only usable when 'arse_dates' option is set when reading the corresponding column of the original data file'''
cnt_srs = train_df['transaction_month'].value_counts()      # count the number in each month
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
plt.xticks(rotation='vertical')
plt.xlabel('Month of transaction', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()




plt.figure(figsize=(12,12))
sns.jointplot(x=train_df['logerror'], y=train_df['transaction_month'], size=10)
plt.ylabel('logerror', fontsize=12)
plt.xlabel('month', fontsize=12)
plt.show()



for i in range(1,13):
    print (i, '\t', train_df['logerror'].ix[train_df['transaction_month']==i].mean(),train_df['logerror'].ix[train_df['transaction_month']==i].std())





plt.figure(figsize=(12,8))
sns.violinplot(x='transaction_month', y='logerror', data=train_df)
plt.xlabel('transaction_month', fontsize=12)
plt.ylabel('Log Error', fontsize=12)
plt.show()






