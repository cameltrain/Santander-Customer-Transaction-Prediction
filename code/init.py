# -*- coding: utf-8 -*-
"""
Created on Fri Dec 06 16:01:51 2013
@author: Krszysztof Sopyła
@email: krzysztofsopyla@gmail.com
@license: MIT
"""


'''
Simple usage of classifier
'''


import sys
sys.path.append("../pyKMLib")

import GPUSolvers as gslv
import GPUKernels as gker

import numpy as np
import scipy.sparse as sp
import time

import pylab as pl

from sklearn import datasets


# import some data to play with

#iris = datasets.load_iris()
#X = iris.data
#Y = iris.target

# multiclass 
X, Y = datasets.load_svmlight_file('../data/glass.scale.txt')
#X, Y = datasets.load_svmlight_file('glass.scale_3cls.txt')

#binary
#X, Y = datasets.load_svmlight_file('glass.scale_binary')
#X, Y = datasets.load_svmlight_file('Data/heart_scale')
#X, Y = datasets.load_svmlight_file('Data/w8a')
#X, Y = datasets.load_svmlight_file('toy_2d_16.train')


#set the classifier parameters
C=0.1 #penalty SVM parameter
gamma=1.0 #RBF kernel gamma parameter



svm_solver = gslv.GPUSVM2Col(X,Y,C)
#kernel = Linear()
kernel = gker.GPURBFEll(gamma=gamma)

#init the classifier, mainly it inits the cuda module and transform data into 
#particular format
t0=time.clock()
svm_solver.init(kernel)
t1=time.clock()
print ('\nInit takes',t1-t0)


#start trainning
t0=time.clock()

svm_solver.train()

t1=time.clock()

print ('\nTakes: ', t1-t0)

#one model coresponds to one classifier in All vs All (or One vs One) multiclass approach
#for each model show solution details
for k in xrange(len(svm_solver.models)):
    m=svm_solver.models[k]
    print ('Iter=',m.Iter)
    print ('Obj={} Rho={}'.format(m.Obj,m.Rho))

    print ('nSV=',m.NSV)
    #print m.Alpha

#start prediction
t0=time.clock()
pred2,dec_vals=svm_solver.predict(X)
t1=time.clock()


svm_solver.clean()

print ('\nPredict Takes: ', t1-t0)
#print pred2
acc = (0.0+sum(Y==pred2))/len(Y)

print ('acc=',acc)


#libsvm from sklearn
from sklearn import svm

clf = svm.SVC(C=C,kernel='linear',verbose=True)
clf = svm.SVC(C=C,kernel='rbf',gamma=gamma,verbose=True)
t0=time.clock()
svm_m= clf.fit(X,Y)
t1=time.clock()
#
print ('\nTrains Takes: ', t1-t0)
#print 'alpha\n',clf.dual_coef_.toarray()

#print 'nSV=',clf.n_support_
#print 'sv \n',clf.support_vectors_.toarray()
#print 'sv idx=',clf.support_


t0=time.clock()
pred1 = clf.predict(X)
t1=time.clock()
print ('\nPredict Takes: ', t1-t0)
#print pred1
acc = (0.0+sum(Y==pred1))/len(Y)

print ('acc=',acc)

print ('--------------\n')


#np.random.seed(0)
#n=6
#X = np.random.randn(n, 2)
#Y = np.random.randint(1,4,n)
#X = np.array([ (1,2), (3,4), (5,6), (7,8), (9,0)])
#Y = np.array([4,1,2,1,4])
