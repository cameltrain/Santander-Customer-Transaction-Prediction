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


# path to where the data lies
dpath = '../data'

modelfile = modelXGB
outfile = 'submission2.csv'
# make top 15% as positive
threshold_ratio = 0.15

# load in training data, directly use numpy
df_test =  pd.read_csv('../data/test.csv')
df_x_test = df_test.drop(columns = ['ID_code'])

print ('finish loading from csv ')
xgmat = xgb.DMatrix( df_test, missing = -999.0 )
ypred = modelXGB.predict( xgmat )

res  = [ ( int(idx[i]), ypred[i] ) for i in range(len(ypred)) ]

rorder = {}
for k, v in sorted( res, key = lambda x:-x[1] ):
    rorder[ k ] = len(rorder) + 1

# write out predictions
ntop = int( threshold_ratio * len(rorder ) )
fo = open(outfile, 'w')
nhit = 0
ntot = 0
fo.write('EventId,RankOrder,Class\n')
for k, v in res:
    if rorder[k] <= ntop:
        lb = 's'
        nhit += 1
    else:
        lb = 'b'
    # change output rank order to follow Kaggle convention
    fo.write('%s,%d,%s\n' % ( k,  len(rorder)+1-rorder[k], lb ) )
    ntot += 1
fo.close()

print ('finished writing into prediction file')
