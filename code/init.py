import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time

df = pd.read_csv('../data/train.csv')
df = df.drop(columns=['ID_code'])
X = df.iloc[:,1:201]
y = df.iloc[:,0]

# Create 0.75/0.25 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75,
                                                    random_state=42)

# Specify sufficient boosting iterations to reach a minimum
num_round = 3

# Leave most parameters as default
param = {'objective': 'multi:softmax', # Specify multiclass classification
         'num_class': 8, # Number of possible output classes
         'tree_method': 'gpu_hist' # Use GPU accelerated algorithm
         }

# Convert input data from numpy to XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

gpu_res = {} # Store accuracy result
tmp = time.time()
# Train model
modelXGB = xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))

y_pred = modelXGB.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(modelXGB.score(X_test, y_test)))

df_test= pd.read_csv('../data/test.csv')
df_x_test = df_test.drop(columns = ['ID_code'])
y_pred = modelXGB.predict(df_x_test)
df_y = pd.DataFrame(y_pred)
df_submission = pd.merge(pd.DataFrame(df_test['ID_code']),df_y,left_index=True,right_index=True)


df_submission = df_submission.rename(columns={0: 'target'})

df_submission.to_csv('../data/submit2.csv', encoding='utf-8', index=False)
