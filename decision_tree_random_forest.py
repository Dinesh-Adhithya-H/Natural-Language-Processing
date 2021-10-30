#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:32:07 2021

@author: Tanmay Basu
"""

import csv,sys
import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn import svm 
from sklearn.feature_selection import SelectKBest,chi2 
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from scipy import sparse
from mlxtend.preprocessing import DenseTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import check_array

# Load data
fl=open('./spambase.csv', 'r')  # Source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
reader = list(csv.reader(fl,delimiter='\t'))

data=[]; labels=[];
for item in reader:
    labels.append(item[0])    
    data.append(item[1])

CountVec = CountVectorizer(stop_words='english',ngram_range=(1,1))
Count_data = CountVec.fit_transform(data)
features=CountVec.get_feature_names()

opt1=input('Enter\n\t "a" for Simple Classification \n\t "b" for Classification with Grid Search \n\t "q" to quit \n')

# simple run with no parameter tuning
if opt1=='a': 
    opt2=input('Enter\n\t "d" for Decision Tree \n\t "r" for Random Forest \n\t "q" to quit \n')             
    # TFIDF
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,1))
    tfidf = vectorizer.fit_transform(data)
    tfidf = tfidf.toarray()
    
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(tfidf, labels, test_size=0.20, random_state=42,stratify=labels)   

    if opt2=='d':       # Decision Tree Classifier
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=40,random_state=40,class_weight='balanced')
        path=clf.cost_complexity_pruning_path(trn_data,trn_cat) 
        print("The Alpha Values of Cost Complexity Pruning \n")
        alphas=path['ccp_alphas']
        print(alphas)
    elif opt2=='r':         # Random Forest Classifier
        clf = RandomForestClassifier(criterion='gini',class_weight='balanced') 
    else:
        print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
        sys.exit(0)
        # Classificaion    
    clf.fit(trn_data,trn_cat)
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
# Parameter tuning using grid search    
elif opt1=='b': 
    opt2=input('Enter\n\t "d" for Decision Tree \n\t "r" for Random Forest \n\t "q" to quit \n')
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels)   
    if opt2=='d':       # Decision Tree Classifier   
        clf = DecisionTreeClassifier(random_state=40)
        clf_parameters = {
        'clf__criterion':('gini', 'entropy'), 
        'clf__max_features':('auto', 'sqrt', 'log2'),
        'clf__max_depth':(10,20,30,40,45,50,60,100),
        'clf__ccp_alpha':(0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.05,0.1),
        }  
    elif opt2=='r':         # Random Forest Classifier
#        clf = svm.SVC(kernel='linear', class_weight='balanced')  
#        clf_parameters = {'clf__C':(0.1,0.5,1,2,10,50,100),}       
        clf = RandomForestClassifier(class_weight='balanced') 
        clf_parameters = {
                    'clf__criterion':('gini', 'entropy'), 
                    'clf__max_features':('auto', 'sqrt', 'log2'),   
                    'clf__n_estimators':(30,50,100,200),
                    'clf__max_depth':(10,20,30,40,45,50,60,100),
                    }               
    else:
        print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
        sys.exit(0)
    
# Feature Extraction
    pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('feature_selector', SelectKBest(chi2, k=1000)),     
    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),   
#    ('to_dense',DenseTransformer()),    
    ('to_dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),    
    ('clf', clf),]) 
        
    feature_parameters = {
    'vect__min_df': (2,3),
    'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigrams
    }
    
    #Classificaion
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10) 
    grid.fit(trn_data,trn_cat)  
#    grid.fit(numpy.asarray(trn_data),numpy.asarray(trn_cat))    
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
else:
    print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
    sys.exit(0)

# Evaluation
print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
print('\n Total documents in the test set: '+str(len(tst_data))+'\n')
print ('\n Confusion Matrix \n')  
print (confusion_matrix(tst_cat, predicted))  

pr=precision_score(tst_cat, predicted, average='micro') 
print ('\n Precision:'+str(pr)) 

rl=recall_score(tst_cat, predicted, average='micro') 
print ('\n Recall:'+str(rl))

#fm=f1_score(tst_cat, predicted, average='macro') 
#print ('\n Macro Averaged F1-Score :'+str(fm))
#
fm=f1_score(tst_cat, predicted, average='micro') 
print ('\n Mircro Averaged F1-Score:'+str(fm))


# Run the following modules to install necessary packages to 
# plot the decision tree
    # apt-get install graphviz  (for Debian/Ubuntu Linux OS)
    # pip install graphviz
    # pip install pydotplus    

# Plot
#if opt2=='d':      
#    dot_data = StringIO()
#    export_graphviz(clf, out_file=dot_data,  
#                    filled=True, rounded=True,
#                    special_characters=True, feature_names = features,
#                    class_names=['ham','spam'])
#    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#    graph.write_png('spambase_decision_tree.png')
#    Image(graph.create_png()) 
