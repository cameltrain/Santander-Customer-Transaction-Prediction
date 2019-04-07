#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Santander EDA, PCA and Light GBM Classification Model</font></center></h1>
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Another_new_Santander_bank_-_geograph.org.uk_-_1710962.jpg/640px-Another_new_Santander_bank_-_geograph.org.uk_-_1710962.jpg"></img>
# 
# <br>
# <b>
# In this challenge, Santander invites Kagglers to help them identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data they have available to solve this problem. 
# The data is anonimyzed, each row containing 200 numerical values identified just with a number.</b>
# 
# <b>Inspired by Jiwei Liu's Kernel. I added Data Augmentation Segment to my kernel</b>
# 
# ### I will not be covering EDA in this kernel . I'd keep it short as the data is completely anonimized and all columns are just pure numbers, giving almost no insight . 
# https://www.kaggle.com/roydatascience/eda-pca-lgbm-santander-transactions  You can check for EDA here

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

plt.style.use('seaborn')
sns.set(font_scale=1)


# In[4]:


random_state = 42
np.random.seed(random_state)
df_train = pd.read_csv('../../data/train.csv')
df_test = pd.read_csv('../../data/test.csv')


# In[5]:


df_train.columns


# In[6]:


print("Total values in the dataset : {}".format(df_train['target'].count()))
Ones = df_train.groupby('target')['target'].count()
print("% of 1s in total {}".format(Ones[1]*100.0/200000))


# ## As one can see , there is a class imbalance. 
# ### Now , how do we solve it ? 
# 
# ### Plan 1 : Oversampling / Undersampling -> 
# * In this strategy , we either increase or decrease the number of samples by duplicating the smaller class or removing the majority class elements to make them equal or similar
# * The risk involved is that we may change the original distribution of the data . 
# 
# ### Plan 2 : Follow below ----->

# This is how we filter using masks 

# In[ ]:





# In[7]:


# Using the mask to filter out 1s . 
y = df_train['target']
y.head()
x = df_train[y > 0].copy()
x.head()


# 

# In[ ]:





# In[8]:


def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


# In[17]:


lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_hessian_in_leaf" : 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state,
    'num_threads': 16,
}


# In[18]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
oof = df_train[['ID_code', 'target']]
oof['predict'] = 0
predictions = df_test[['ID_code']]
val_aucs = []
feature_importance_df = pd.DataFrame()


# In[19]:


features = [col for col in df_train.columns if col not in ['target', 'ID_code']]
X_test = df_test[features].values


# In[ ]:


for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):
    X_train, y_train = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx]['target']
    X_valid, y_valid = df_train.iloc[val_idx][features], df_train.iloc[val_idx]['target']
    
    N = 5
    p_valid,yp = 0,0
    for i in range(N):
        X_t, y_t = augment(X_train.values, y_train.values,12)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')
    
        trn_data = lgb.Dataset(X_t, label=y_t)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result
                       )
        p_valid += lgb_clf.predict(X_valid)
        yp += lgb_clf.predict(X_test)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    oof['predict'][val_idx] = p_valid/N
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
    
    predictions['fold{}'.format(fold+1)] = yp/N


# In[ ]:


get_ipython().run_line_magic('precision', '3')


# # Let's see what this augmentation does to our data frame

# In[ ]:


print("Distribution of 1s in original data : {} / {} ".format(np.sum(y_train) , len(y_train)))
print("Percentage of 1s in original data : {}".format(np.sum(y_train)*100.0/len(y_train)))


print("Percentage of 1s in augmented data : {}".format(np.sum(y_t)*100.0/len(y_t)))
print("Distribution of 1s in augmented data : {} / {} ".format(np.sum(y_t) , len(y_t)))


# ## So How the augmentation was done ? 
# 
#     * X : Original  : 200,000
#     * Xs : Ones     ~  20,000
#     * Xn : Zeros    ~ 180,000  
# 
#     ### X_Final = X + 3\*Xs + 2\*Xn
# 

# Proof : ? 
# 
# 1s = 20k(X) + 3\*20k(Xs) = 80k
# Total = 20k(X) + 3\*20k(Xs) + 2\*180k(Xn)
# 
# i.e. , 
#     80k / 520 k = 
#     14.6 % (Approx. , the one we found above)

# ## Hence this technique is more like oversampling , but , here we oversample BOTH classses , rather than just one.      

# In[ ]:


mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))


# In[ ]:


cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# In[ ]:


# submission
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('lgb_all_predictions.csv', index=None)
sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv("../../data/lgb_submission_12.csv", index=False)
oof.to_csv('../../data/lgb_oof_12.csv', index=False)


# In[ ]:



# In[ ]:




