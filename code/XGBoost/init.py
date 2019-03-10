import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import time

df = pd.read_csv('../../data/train.csv',engine='python')
df = df.drop(columns=['ID_code'])
X = df.iloc[:,1:201]
y = df.iloc[:,0]

# Create 0.75/0.25 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75,
                                                    random_state=42)

# Specify sufficient boosting iterations to reach a minimum
num_round = 2000

# Leave most parameters as default
param = {'objective': 'multi:softmax', # Specify multiclass classification
         'num_class': 2, # Number of possible output classes
         'bin':64,
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

joblib.dump(modelXGB, '../../models/modelXGB2000.pkl', compress=9)

# load in training data, directly use numpy
#df_test =  pd.read_csv('../data/testedited.csv')

#print ('finish loading from csv ')
#xgmat = xgb.DMatrix( df_test, missing = -999.0 )
#ypred = modelXGB.predict( xgmat )


#np.savetxt("../data/submit3.csv", ypred, delimiter=",")
