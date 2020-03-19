#!/usr/bin/env python
# coding: utf-8

# # Import Data
# # originally this one is a ipynb file
# In[1]:


import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.getcwd()


# In[2]:


#change working directory
os.chdir('C:\\Users\\russe\\Desktop\\ML1\\final project')


# In[2]:


#Read data
#Import three datasets
train_000_01=pd.read_csv("train_000_01.csv")
val_000_01=pd.read_csv("val_000_01.csv") 
test_000_01=pd.read_csv('test_000_01.csv')


# # Data Preparation

# In[3]:


#For the variable hour, which is the date and hour when the ad was displayed.
#We would like to get the information of hour since people tend to click on advertisement on specific time period 
train_000_01.hour=train_000_01.hour % 100
val_000_01.hour=val_000_01.hour % 100
test_000_01.hour=test_000_01.hour % 100


# In[4]:


#Define x-variables and y-variable in train, validation and test data set
feature=range(3,25,1)
X_train=train_000_01.iloc[:,feature]
y_train=train_000_01.iloc[:,2]
X_val=val_000_01.iloc[:,feature]
y_val=val_000_01.iloc[:,2]
X_test=test_000_01.iloc[:,feature]
y_test=test_000_01.iloc[:,2]


# In[5]:


#Feature Hashing
from sklearn.feature_extraction import FeatureHasher
X_train_hash = X_train.copy()
X_val_hash = X_val.copy()
X_test_hash = X_test.copy()
for i in range(X_train_hash.shape[1]):
    X_train_hash.iloc[:,i]=X_train_hash.iloc[:,i].astype('str')
for i in range(X_val_hash.shape[1]):
    X_val_hash.iloc[:,i]=X_val_hash.iloc[:,i].astype('str')
for i in range(X_test_hash.shape[1]):
    X_test_hash.iloc[:,i]=X_test_hash.iloc[:,i].astype('str')

#encoding hashing
h = FeatureHasher(n_features=10000,input_type="string")
X_train_hash = h.transform(X_train_hash.values)
X_val_hash = h.transform(X_val_hash.values)
X_test_hash = h.transform(X_test_hash.values)


# # Modeling

# In[9]:


#Import Neccessary Packages
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
import keras 
import random as rn
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.preprocessing import StandardScaler
import lightgbm


# ## Logsitic Regression

# In[46]:


#We used for loop to do parameter tuning for logistic regression
#using l2 norm penalty
C_param_range = [0.001,0.01,0.1,1,10,100]
for i in C_param_range:
    lr = LogisticRegression(penalty = 'l2', C = i,random_state = 0)
    lr.fit(X_train_hash,y_train)
    y_pred = lr.predict_proba(X_val_hash)
    print(log_loss(y_val,y_pred),i)


# In[47]:


#We used for loop to do parameter tuning for logistic regression
#using l1 norm penalty
C_param_range = [0.001,0.01,0.1,1,10,100]
for i in C_param_range:
    lr = LogisticRegression(penalty = 'l1', C = i,random_state = 0)
    lr.fit(X_train_hash,y_train)
    y_pred = lr.predict_proba(X_val_hash)
    print(log_loss(y_val,y_pred),i)


# In[15]:


#According to the parameter tuning result, the best parameters: C = 0.1, penalty=l1
#Test the generalized performance on the testing set    
l = LogisticRegression(penalty = 'l1', C = 0.1,random_state = 0)
l.fit(X_train_hash,y_train)
y_pred_test_logit = l.predict_proba(X_test_hash)

print(log_loss(y_test,y_pred_test_logit))


# ## Ensemble Method using Logistic Regression

# In[9]:


#Import other three train datasets to train logistic regression model
y_train_01=y_train

train_000_02=pd.read_csv("train_000_02.csv")
train_000_02.hour=train_000_02.hour % 100
X_train_02=train_000_02.iloc[:,feature]
y_train_02=train_000_02.iloc[:,1]

train_000_03=pd.read_csv("train_000_03.csv")
X_train_03=train_000_03.iloc[:,feature]
y_train_03=train_000_03.iloc[:,1]

train_000_04=pd.read_csv("train_000_04.csv")
X_train_04=train_000_04.iloc[:,feature]
y_train_04=train_000_04.iloc[:,1]


# In[10]:


#Repeat hashing for these three new train datasets
X_train_hash_01=X_train_hash

X_train_hash_02 = X_train_02.copy()
for i in range(X_train_hash_02.shape[1]):
    X_train_hash_02.iloc[:,i]=X_train_hash_02.iloc[:,i].astype('str')
X_train_hash_02 = h.transform(X_train_hash_02.values)

X_train_hash_03 = X_train_03.copy()
for i in range(X_train_hash_03.shape[1]):
    X_train_hash_03.iloc[:,i]=X_train_hash_03.iloc[:,i].astype('str')
X_train_hash_03 = h.transform(X_train_hash_03.values)

X_train_hash_04 = X_train_04.copy()
for i in range(X_train_hash_04.shape[1]):
    X_train_hash_04.iloc[:,i]=X_train_hash_04.iloc[:,i].astype('str')
X_train_hash_04 = h.transform(X_train_hash_04.values)


# In[13]:


l = LogisticRegression(penalty = 'l1', C = 0.1,random_state = 0)


# In[ ]:


#Fit these three train datasets and make predictions based on validation dataset
l.fit(X_train_hash_01,y_train_01)
y_pred_01 = l.predict_proba(X_test_hash)
l.fit(X_train_hash_02,y_train_02)
y_pred_02 = l.predict_proba(X_test_hash)
l.fit(X_train_hash_03,y_train_03)
y_pred_03 = l.predict_proba(X_test_hash)
l.fit(X_train_hash_04,y_train_04)
y_pred_04 = l.predict_proba(X_test_hash)


# In[27]:


#Calculate the average performance
y_pred_avg=(y_pred_01+y_pred_02+y_pred_03+y_pred_04)/4

print(log_loss(y_test,y_pred_avg))


# ## Random Forest

# In[10]:


#Trial 1
r1 = RandomForestClassifier(n_estimators=10)
r1.fit(X_train_hash,y_train)
#predicting on the validation set
y_pred_r1 = r1.predict_proba(X_val_hash)
print(log_loss(y_val,y_pred_r1))


# In[ ]:


#Trial 2
r2 = RandomForestClassifier(n_estimators=100)
r2.fit(X_train_hash,y_train)
#predicting on the validation set
y_pred_r2 = r2.predict_proba(X_val_hash)
print(log_loss(y_val,y_pred_r2))


# In[ ]:


#According to the parameter tuning result, the best parameters: n_estimators=
#Test the generalized performance on the testing set    
y_pred_test_forest = r2.predict_proba(X_test_hash)

print(log_loss(y_test,y_pred_test_forest))


# ## Neural Network

# In[ ]:


#Datasets preparation for Neural Network

#Feature Hashing using 500 n_features

h = FeatureHasher(n_features=500,input_type="string")
X_train_hash = h.transform(X_train_hash.values)
X_val_hash = h.transform(X_val_hash.values)
X_test_hash = h.transform(X_test_hash.values)    

#convert into array
YTr = np.array(y_train)
XTr = X_train_hash.toarray()

YVal = np.array(y_val)
XVal = X_val_hash.toarray()

YTest = np.array(y_test)
XTest = X_test_hash.toarray()


# In[ ]:


#rescale
XTrRsc = (XTr - XTr.min(axis=0))/XTr.ptp(axis=0)
XTrRsc.shape
XTrRsc.min(axis=0)
XTrRsc.max(axis=0)

# Note YTr does not need to be rescaled since it is binary

#Rescale Validation and Test data. Really should use Training parameters to rescale.
XValRsc = (XVal - XTr.min(axis=0))/XTr.ptp(axis=0)
XValRsc.shape
XValRsc.min(axis=0)
XValRsc.max(axis=0)

XTestRsc = (XTest - XTr.min(axis=0))/XTr.ptp(axis=0)
XTestRsc.shape
XTestRsc.min(axis=0)
XTestRsc.max(axis=0)


# In[ ]:


BatchSize=250
Optimizer=optimizers.RMSprop(lr=0.01)

def SetTheSeed(Seed):
    np.random.seed(Seed)
    rn.seed(Seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

    tf.set_random_seed(Seed)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


# In[ ]:


#NN MODEL 1 sigmoid activation function

NEpochs = 1000
BCNN = Sequential()

BCNN.add(Dense(units=4,input_shape=(XTrRsc.shape[1],),activation="relu",use_bias=True))
BCNN.add(Dense(units=4,activation="relu",use_bias=True))
BCNN.add(Dense(units=4,activation="relu",use_bias=True))
BCNN.add(Dense(units=4,activation="relu",use_bias=True))
BCNN.add(Dense(units=1,activation="sigmoid",use_bias=True))

BCNN.compile(loss='binary_crossentropy', optimizer=Optimizer,metrics=['binary_crossentropy'])

#%% Fit NN Model

FitHist = BCNN.fit(XTrRsc,YTr,epochs=NEpochs,batch_size=BatchSize,verbose=1)
print("Number of Epochs = "+str(len(FitHist.history['binary_crossentropy'])))
FitHist.history['binary_crossentropy'][-1]
FitHist.history['binary_crossentropy'][-10:-1]

#%% Make Predictions
YHatTr = BCNN.predict(XTrRsc,batch_size=XTrRsc.shape[0]) # Note: Not scaled, so not necessary to undo.
YHatTr = YHatTr.reshape((YHatTr.shape[0]),)

YHatVal = BCNN.predict(XValRsc,batch_size=XValRsc.shape[0])
YHatVal = YHatVal.reshape((YHatVal.shape[0]),)

print(log_loss(y_val,YHatVal))


# In[ ]:


#NN MODEL 2 Now try using softmax activation function

#SetTheSeed(3456)
NEpochs = 10 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

BCNNsm = Sequential()

BCNNsm.add(Dense(units=4,input_shape=(XTrRsc.shape[1],),activation="relu",use_bias=True))
BCNNsm.add(Dense(units=4,activation="relu",use_bias=True))
BCNNsm.add(Dense(units=4,activation="relu",use_bias=True))
BCNNsm.add(Dense(units=4,activation="relu",use_bias=True))
BCNNsm.add(Dense(units=2,activation="softmax",use_bias=True))

BCNNsm.compile(loss='categorical_crossentropy', optimizer=Optimizer,metrics=['categorical_crossentropy'])

#%% Fit NN Model with Softmax

# Need to make YTr an n by 2 matrix

YTr = np.array([1-YTr,YTr]).transpose()

FitHist = BCNNsm.fit(XTrRsc,YTr,epochs=NEpochs,batch_size=BatchSize,verbose=1)
print("Number of Epochs = "+str(len(FitHist.history['categorical_crossentropy'])))
FitHist.history['categorical_crossentropy'][-1]
FitHist.history['categorical_crossentropy'][-10:-1]

#%% Make Predictions
YHatTrSM = BCNNsm.predict(XTrRsc,batch_size=XTrRsc.shape[0]) # Note: Not scaled, so not necessary to undo.
YHatValSM = BCNNsm.predict(XValRsc,batch_size=XValRsc.shape[0]) # Note: Not scaled, so not necessary to undo.

print(log_loss(y_val,YHatValSM))


# In[ ]:


#predict log-loss based on test data
YHatTestSM = BCNNsm.predict(XTestRsc,batch_size=XTestRsc.shape[0]) 
print(log_loss(y_test,YHatTestSM))


# ## LightGBM

# In[ ]:


#Since LightGBM needs two train dataset, we do hashing(n_features=500) on another train dataset
#The other datasets are ready since we made changes in Neural Netowrk
X_train_hash_02 = h.transform(X_train_hash_02.values)

#convert to array first 
x_train_hash_copy=x_train_hash.toarray()
x_train_hash_02=x_train_hash_02.toarray()
x_val_hash_copy=x_val_hash.toarray()
x_test_hash_copy=X_test_hash.toarray()


# In[ ]:


#get series data for LightGBM
ytrain_series=pd.DataFrame(YTr).values
xtrain_series=x_train_hash_copy.copy()

ytrain_series_02=pd.DataFrame(y_train_02).values
xtrain_series_02=x_train_hash_02.copy()

xval_series=x_val_hash_copy.copy()
yval_series=pd.DataFrame(YVal).values

xtest_series=x_test_hash_copy.copy()
ytest_series=pd.DataFrame(y_test).values

#get series for y
ytrain_series=ytrain_series[:,0]
yval_series=yval_series[:,0]
ytest_series=ytest_series[:,0]
ytrain_series_02=ytrain_series_02[:,0]


# In[ ]:


# Feature Scaling 
sc = StandardScaler()
xtrain_series= sc.fit_transform(xtrain_series)
xtrain_series_02= sc.fit_transform(xtrain_series_02)
xval_series=sc.fit_transform(xval_series)


# In[ ]:


train_data = lightgbm.Dataset(xtrain_series, label=ytrain_series)
test_data = lightgbm.Dataset(xtrain_series_02, label=ytrain_series_02,reference=train_data)

# Train the model
# parameter tuning manually

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 8,
    'learning_rate': 0.1,
    'verbose': 0
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=100,
                       early_stopping_rounds=100)
#prediction
y = model.predict(xval_series)

#estimate log loss
print(log_loss(y_val,y))


# In[ ]:


#get generalized performance on testing dataset using the best parameter
train_data = lightgbm.Dataset(xtrain_series, label=ytrain_series)
test_data = lightgbm.Dataset(xtest_series, label=ytest_series,reference=train_data)

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=100,
                       early_stopping_rounds=100)

y = model.predict(xtest_series)


# In[ ]:


#estimate log loss
print(log_loss(y_test,y))


# # Make Prediction

# In[4]:


#change working directory
os.chdir('C:\\Users\\russe\\Desktop\\ML1\\final project\\Project Data\\Project Data')


# In[5]:


#import testing dataset
testing=pd.read_csv("ProjectTestData.csv")
#chage the hour variable
testing.hour=testing.hour % 100


# In[22]:


testing.head(3)


# In[20]:


train_000_01.head(3)


# In[8]:


#import training data
train_000_01=pd.read_csv("train_000_01.csv")


# In[29]:


#Define x-variables and y-variable in train and validation data set
feature_test=range(1,23,1)
X_train=train_000_01.iloc[:,feature]
y_train=train_000_01.iloc[:,2]
X_test=testing.iloc[:,feature_test]
#y_val=val_000_01.iloc[:,2]
#Feature Hashing
from sklearn.feature_extraction import FeatureHasher
X_train_hash = X_train.copy()
X_test_hash = X_test.copy()
for i in range(X_train_hash.shape[1]):
    X_train_hash.iloc[:,i]=X_train_hash.iloc[:,i].astype('str')
for i in range(X_test_hash.shape[1]):
    X_test_hash.iloc[:,i]=X_test_hash.iloc[:,i].astype('str')

#encoding hashing
h = FeatureHasher(n_features=10000,input_type="string")
X_train_hash = h.transform(X_train_hash.values)
X_test_hash = h.transform(X_test_hash.values)


# In[33]:


#fit the model
l1 = LogisticRegression(penalty = 'l1', C = 0.1,random_state = 0)
l1.fit(X_train_hash,y_train)
y_pred_test = l1.predict_proba(X_test_hash)


# In[35]:


#write out the predict outcome
df_predict = pd.DataFrame(y_pred_test)
df_predict.to_csv('predict_outcome.csv',index=False)


# In[39]:


#write in the submission file
submission=pd.read_csv("ProjectSubmission-TeamX.csv")


# In[43]:


#insert the p(click) into submission file
submission.iloc[:,1]=df_predict.iloc[:,1]


# In[45]:


#write out the submission file with preidiction
submission.to_csv('submission_final.csv',index=False)

