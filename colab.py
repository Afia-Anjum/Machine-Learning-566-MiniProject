# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:49:49 2019

@author: Asus
"""
  #param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
  #grid = GridSearchCV(LogisticRegression(), param_grid, cv=3)
  #grid.fit(X_train, ytrain)
  #best_score = grid.best_score_ * 100
  #print("Best cross-validation score for logistic regression : " +str(best_score))
  #print("Best parameters: ", grid.best_params_)
  #print("Best estimator: ", grid.best_estimator_)
  

import math
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression

def loadCsvDataset(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def load_cervical_dataset(samplesize):
    """ A cervical cancer risk factor dataset """
    filename = "kag_risk_factors_cervical_cancer.csv"
    dataset = loadCsvDataset(filename)
    return dataset

def main():

  samplesize=858
  dataset=load_cervical_dataset(samplesize)
  X=dataset[1:, :31]  
  Y=dataset[1:, 32]
  
  #clf1=LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr',penalty='l2',fit_intercept=True,class_weight='balanced')
  clf1=LogisticRegression(random_state=0,multi_class='ovr',fit_intercept=True,class_weight='balanced')

  imp = IterativeImputer(max_iter=100, random_state=0)
  imp.fit(X)

  IterativeImputer(add_indicator=False, estimator=None,
                     imputation_order='random', initial_strategy='constant',
                     max_iter=1000, max_value=None, min_value=None,
                     missing_values=np.nan, n_nearest_features=None,
                     random_state=0, sample_posterior=False, tol=0.001,
                     verbose=0)
  X_imp = imp.transform(X)
  
 # params = [
 #       { random_state=0, solver='lbfgs',multi_class='ovr',penalty='l2',fit_intercept=True,tol=0.0001, max_iter=100,class_weight=None },
 #       { 'remove_stop_words': True, 'remove_unknown_words': False },
 #       { 'remove_stop_words': False, 'remove_unknown_words': True },
 #       { 'remove_stop_words': True, 'remove_unknown_words': True }
 #   ]
  #1000; 0.92073303
  #100;  0.9219162246702026
  #tot=0.001 ; 0.9219162246702026
  #liblinear
  
  #param_grid = {
  #  'select__k': [1, 2],
  # 'model__base_estimator__max_depth': [2, 4, 6, 8]}
  #parameters1 = { 'tol':[0.0001,0.001,0.01,0.1], 'max_iter':[100,1000]}   #correct one 
  parameters = { 'Logistic Regression': {'tol':[0.0001,0.001,0.01,0.1], 'max_iter':[100,1000],'penalty':['l2','l1'],'solver':['liblinear','lbfgs'] }}
  
  #clf1=
  classalgs = {
        'Logistic Regression': clf1,
        #'Neural Network': algs.NeuralNet,
        #'Support Vector Machine': algs.SVM
    }
  
  numalgs = len(classalgs)
  
  #clf=LogisticRegression(random_state=0, solver='liblinear',multi_class='ovr',penalty='l2',fit_intercept=True,tol=0.0001, max_iter=1000,class_weight='balanced')
  #clf=LogisticRegression(fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1.0, multi_class='auto', random_state=None, l1_ratios=None)

  #https://datascience.stackexchange.com/questions/6676/scikit-learn-getting-sgdclassifier-to-predict-as-well-as-a-logistic-regression
  #helpful for our model outline
  #accuracy=np.zeros()
  n_splits=5
  
  errors = {}
  for learnername in classalgs:
      errors[learnername] = np.zeros(n_splits)
    
  count_array=np.zeros((3,n_splits))
  accuracy=[]   
  """
  skf = StratifiedKFold(n_splits=5)
  skf.get_n_splits(X, Y)
  i=0
  for train_index, test_index in skf.split(X, Y):
    Xtrain,Xtest=X_imp[train_index], X_imp[test_index]
    Ytrain,Ytest=Y[train_index],Y[test_index]
    #print("TRAIN:", train_index, "TEST:", test_index)   #original train and original test divide
    #for learnername, Learner in classalgs.items():
    #    params = parameters.get(learnername, [ None ])
    #    grid = GridSearchCV(Learner, params, cv=5)
    
    grid = GridSearchCV(clf1, parameters1, cv=5)
    grid.fit(Xtrain, Ytrain)
    best_score = grid.best_score_ * 100
    print("Best cross-validation score for logistic regression : " +str(best_score))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    predictions=grid.best_estimator_.predict(Xtest)
    score=accuracy_score(predictions,Ytest)
    accuracy.append(score)
    #print(score)
    count_array[0][i]=score
    i+=1
  #print(count_array)
  """
  j=0
  for learnername, Learner in classalgs.items():
        params = parameters.get(learnername, [ None ])
        i=0
        skf = StratifiedKFold(n_splits=5)
        skf.get_n_splits(X, Y)
        for train_index, test_index in skf.split(X, Y):
            Xtrain,Xtest=X_imp[train_index], X_imp[test_index]
            Ytrain,Ytest=Y[train_index],Y[test_index]
            grid = GridSearchCV(Learner, params, cv=5)
            grid.fit(Xtrain, Ytrain)
            #best_score = grid.best_score_ * 100
            #print("Best cross-validation score for logistic regression : " +str(best_score))
            #print("Best parameters: ", grid.best_params_)
            #print("Best estimator: ", grid.best_estimator_)
            predictions=grid.best_estimator_.predict(Xtest)
            score=accuracy_score(predictions,Ytest)
            count_array[j][i]=score
            i+=1
        j+=1
        #print(params)
        #print(Learner)
        #print(learnername)
        #print("\n")
  #print(np.array(accuracy).mean())
  print(count_array)
  """
  accuracy=[]
  skf = StratifiedKFold(n_splits=5)
  skf.get_n_splits(X, Y)
  for train_index, test_index in skf.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X1train,X1validate=X_imp[train_index], X_imp[test_index]
    Y1train,Y1validate=Y[train_index],Y[test_index]
    clf.fit(X1train,Y1train)   
    predictions=clf.predict(X1validate)
    '''
    pred_proba_df = pd.DataFrame(clf.predict_proba(X1validate))
    
    Ypred = pred_proba_df.applymap(lambda x: 1 if x>0.1 else 0)
    #print(Ypred)
    test_accuracy = metrics.accuracy_score(Y1validate,
                                           Ypred.iloc[:,1].as_matrix().reshape(Ypred.iloc[:,1].as_matrix().size,1))
    #print('Our testing accuracy is {}'.format(test_accuracy))
    #score=test_accuracy
    '''
    #score=accuracy_score(Ypred,Y1validate)
    
    score=accuracy_score(predictions,Y1validate)
    accuracy.append(score)
    #tn, fp, fn, tp = confusion_matrix(Y1validate, Ypred.iloc[:,1].as_matrix().reshape(Ypred.iloc[:,1].as_matrix().size,1)).ravel()
    tn, fp, fn, tp = confusion_matrix(Y1validate, predictions).ravel()
    print(tn, fp, fn, tp)
  #print(accuracy)
  print(np.array(accuracy).mean())
  """
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    #thresholding parameters
  """
    pred_proba_df = pd.DataFrame(model.predict_proba(x_test))
    threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,.95,.99]
    for i in threshold_list:
        print ('\n******** For i = {} ******'.format(i))
        Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>i else 0)
        test_accuracy = metrics.accuracy_score(Y_test.as_matrix().reshape(Y_test.as_matrix().size,1),
                                           Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1))
        print('Our testing accuracy is {}'.format(test_accuracy))

        print(confusion_matrix(Y_test.as_matrix().reshape(Y_test.as_matrix().size,1),
                           Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1)))
   """
  #with warnings.catch_warnings():
    # ignore all caught warnings
    #warnings.filterwarnings("ignore")
    
    

main()