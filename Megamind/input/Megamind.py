'''
@author: Megamind : Project 5, Applied Artificial Intelligence
'''

#------------------------Import all required Packages-------------------#
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import gc


#---------------------------------Data Loading---------------------------#

print('Loading Data.....')
train = pd.read_csv('./train_2016_v2.csv', low_memory=False)
properties = pd.read_csv('./properties_2016.csv', low_memory=False)
test = pd.read_csv('./sample_submission.csv', low_memory=False)

print(" ")
print("Data Loaded successfully.")


#----------------Creating a single matrix of all features------------------------#
train_data = properties
merged_data = train_data.merge(train, how='inner', on='parcelid')


#-----------------Preprocessing the data and keeping only those columns which are un-empty----------------------------#
for o in merged_data.dtypes[merged_data.dtypes==object].index.values:
    merged_data[o] = (merged_data[o]==True)



#----------The number of unreachable objects are retrned-----------------------#
gc.collect()

#----------------Splitting entire data into training and validation data set----------#
train_data, valid_data = train_test_split(merged_data, test_size=0.1, random_state=40)


ytrain = train_data.iloc[:, train_data.shape[1]-2:train_data.shape[1]-1]
xtrain = train_data.iloc[:, 0:train_data.shape[1]-2]



yvalid = valid_data.iloc[:, train_data.shape[1]-2:train_data.shape[1]-1]
xvalid = valid_data.iloc[:, 0:train_data.shape[1]-2]


#-------------------------------------------------------------------------------#
#-------------ESTIMATING OPTIMAL SETTING USING GRIDSEARCH CV----------------------#
#-------------------------------------------------------------------------------#

print(" ")
print("Tuning Hyperparamters using GridSearch CV...")
#------------Its always conventional to tune hyperparamters using development data set---------#

estimator = lgb.LGBMRegressor()

param_grid = {
    'learning_rate': [0.005,0.01,0.1,1],
    'n_estimators': [20, 30, 40],
    'max_bin' : [10],
    'num_leaves' : [50, 100],
    'objective': ['regression']

}

gs=GridSearchCV(estimator, param_grid)

gs.fit(np.array(xvalid), np.array(yvalid.values.ravel()))

print(" ")
print('Best parameters found by grid search are:', gs.best_params_)

lgb_train = lgb.Dataset(xtrain, ytrain['logerror'])


#------Select hyperparamters selected using GRIDSEARCH CV---------#
params={
'objective':gs.best_params_['objective'],
'max_bin':gs.best_params_['max_bin'],
'learning_rate':gs.best_params_['learning_rate'],
'num_leaves':gs.best_params_['num_leaves'],
'n_estimators':gs.best_params_['n_estimators']


}

clf = lgb.train(params,lgb_train)

print(" ")
print("Preparing test data for the prediction ...")

test = pd.read_csv("./sample_submission.csv")
test['parcelid'] = test['ParcelId']
test = test.merge(properties,  how='inner',
                               on='parcelid')

gc.collect()

#-------The num_threads parameter defines the maximum number of worker threads available for scheduler to use.--------#
clf.reset_parameter({"num_threads":4})

gc.collect()


for o in test.dtypes[test.dtypes==object].index.values:
    test[o] = (test[o]==True)

print(" ")
print("Starting prediction ...")
del test['ParcelId']
del test['201610']
del test['201611']
del test['201612']
del test['201710']
del test['201711']
del test['201712']

#---------------Prediction of the test data------------#

p_test = clf.predict(test, num_iteration=clf.best_iteration)

print(p_test)

print(" ")
print("Writing Results to .CSV file...")
#--------------Writing the results to .CSV file----------------------#
sub = pd.read_csv('./sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('Results.csv', index=False, float_format='%.4f')

