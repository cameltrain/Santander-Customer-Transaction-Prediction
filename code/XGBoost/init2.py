import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import time
from sklearn.externals import joblib
modelXGB = joblib.load('../data/modelXGB10.pkl')

# load in training data, directly use numpy


#df_test =  pd.read_csv('../data/testedited.csv')

#print ('finish loading from csv ')

#xgmat = xgb.DMatrix(df_test)
#ypred = modelXGB.predict( xgmat )


df = pd.read_csv('../data/test.csv')
df = df.drop(columns=['ID_code'])


# Create 0.75/0.25 train/test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75,
#                                                    random_state=42)

# Specify sufficient boosting iterations to reach a minimum
#num_round = 10

# Leave most parameters as default
#param = {'objective': 'multi:softmax', # Specify multiclass classification
#         'num_class': 2, # Number of possible output classes
#         'tree_method': 'gpu_hist' # Use GPU accelerated algorithm
#         }

# Convert input data from numpy to XGBoost format
dtest = xgb.DMatrix(df)
#dtest = xgb.DMatrix(X_test, label=y_test)

ypred = modelXGB.predict( dtest )

#np.savetxt("../data/submit3.csv", ypred, delimiter=",")
