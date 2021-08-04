#!/usr/bin/env python
# coding: utf-8

# In[4]:


import warnings

warnings.filterwarnings("ignore")
from collections import Counter
import json, logging, tempfile, sys, codecs, math, io, os, zipfile, arff, time, copy, csv, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from scipy import interp
from scipy.io.arff import loadarff
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,
                                    KMeansSMOTE)
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss,
                                     InstanceHardnessThreshold,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection)
from imblearn.combine import SMOTEENN, SMOTETomek
from os import listdir
from os.path import isfile, join
import typing as t
import numpy as np
import itertools

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score
from scipy.spatial import distance


dataset = 'dataset-name.dat'
datafile = 'link-to-file.zip'


def fscore(classifier, smo_name, seed=0):
    global current_best, best_cls, best_smo, local_log, dataset, X_log, y_log
    k = 5
    cv = StratifiedKFold(n_splits=k, random_state=seed)
    if (classifier == 'SVM'):
        clf = SVC(max_iter=10000, cache_size=700, random_state=seed)
    elif (classifier == 'RF'):
        clf = RandomForestClassifier(random_state=seed)
    elif (classifier == 'KNN'):
        clf = KNeighborsClassifier()
    elif (classifier == 'LR'):
        clf = LogisticRegression(random_state=seed)
    elif (classifier == 'DTC'):
        clf = DecisionTreeClassifier(random_state=seed)
    # resampling parameter

    p_sub_type = smo_name

    if (p_sub_type == 'SMOTE'):
        smo = SMOTE(random_state=seed)
    elif (p_sub_type == 'BorderlineSMOTE'):
        smo = BorderlineSMOTE(random_state=seed)
    elif (p_sub_type == 'SMOTENC'):
        smo = SMOTENC(categorical_features=True, random_state=seed)
    elif (p_sub_type == 'SVMSMOTE'):
        smo = SVMSMOTE(random_state=seed)
    elif (p_sub_type == 'KMeansSMOTE'):
        smo = KMeansSMOTE(random_state=seed)
    elif (p_sub_type == 'ADASYN'):
        smo = ADASYN(random_state=seed)
    elif (p_sub_type == 'RandomOverSampler'):
        smo = RandomOverSampler(random_state=seed)
    # Combine
    elif (p_sub_type == 'SMOTEENN'):
        smo = SMOTEENN(random_state=seed)
    elif (p_sub_type == 'SMOTETomek'):
        smo = SMOTETomek(random_state=seed)
    # Undersampling
    elif (p_sub_type == 'CondensedNearestNeighbour'):
        smo = CondensedNearestNeighbour(random_state=seed)
    elif (p_sub_type == 'EditedNearestNeighbours'):
        smo = EditedNearestNeighbours()
    elif (p_sub_type == 'RepeatedEditedNearestNeighbours'):
        smo = RepeatedEditedNearestNeighbours()
    elif (p_sub_type == 'AllKNN'):
        smo = AllKNN()
    elif (p_sub_type == 'InstanceHardnessThreshold'):
        smo = InstanceHardnessThreshold(random_state=seed)
    elif (p_sub_type == 'NearMiss'):
        smo = NearMiss()
    elif (p_sub_type == 'NeighbourhoodCleaningRule'):
        smo = NeighbourhoodCleaningRule()
    elif (p_sub_type == 'OneSidedSelection'):
        smo = OneSidedSelection(random_state=seed)
    elif (p_sub_type == 'RandomUnderSampler'):
        smo = RandomUnderSampler(random_state=seed)
    elif (p_sub_type == 'TomekLinks'):
        smo = TomekLinks()
    elif (p_sub_type == 'ClusterCentroids'):
        smo = ClusterCentroids(random_state=seed)
    gmean = []
    excMsg, ifError = '', 0
    try:
        for train, test in cv.split(X, y):
            if (p_sub_type != 'NONE'):
                X_smo_train, y_smo_train = smo.fit_sample(X[train], y[train])
            else:
                X_smo_train, y_smo_train = X[train], y[train]
            y_test_pred = clf.fit(X_smo_train, y_smo_train).predict(X[test])
            gm = geometric_mean_score(y[test], y_test_pred, average='binary')
            gmean.append(gm)
        mean_g = np.mean(gmean)
        # print(dataset,randomstate,classifier,p_sub_type,mean_g, mean_auc, np.mean(aucs))
    except Exception as e:
        mean_g = 0
        print(e)
        excMsg= e
        ifError = 1
    if (mean_g > current_best):
        current_best = mean_g
        best_cls = classifier
        best_smo = p_sub_type
    local_log.append([dataset, randomstate, classifier, p_sub_type, mean_g, current_best, excMsg, ifError])
    return mean_g

local_log = []
##If you read data directy from the original KEEL's zip file, use this code:
'''zf = zipfile.ZipFile(datafile)
in_mem_fo = io.TextIOWrapper(io.BytesIO(zf.read(dataset)), encoding='utf-8')
####Convert KEEL format to arff ###
txt = ''
for i in in_mem_fo:
    if ("@input" not in i and '@output' not in i):
        newi = ''
        for e in i.split(','):
            if (newi == ''):
                newi = e.strip()
            else:
                newi = newi + ',' + e.strip()
        txt = txt + newi + '\n'
arff_fm = io.StringIO(txt)
#######LOAD TRAIN DATA######
data, meta = loadarff(arff_fm)
data = pd.DataFrame(data)
enc = LabelEncoder()
### Encoding ###
for a in [col for col, data in data.dtypes.items() if data == object]:
    data[a] = data[a].str.decode('utf-8')
    try:
        #integer value got b (binary)
        data[a] = data[a].astype('int64')
        #interger value sometime get
    except:
        data[a] = enc.fit_transform(data[a])
'''
#######LOAD TRAIN DATA######
data, meta = loadarff(datafile)
data = pd.DataFrame(data)
enc = LabelEncoder()
### Encoding ###
for a in [col for col, data in data.dtypes.items() if data == object]:
    data[a] = enc.fit_transform(data[a])
X = data[data.columns[:-1]].to_numpy()
y = data[data.columns[-1]].to_numpy()
X = StandardScaler().fit_transform(X)
X = np.c_[X]
for randomstate in [9, 18, 27, 29, 36, 59, 79, 90, 109]:
    current_best, best_cls, best_smo = 0, '', ''
    for cls in ["SVM", "RF", 'KNN', "DTC", "LR"]:
        for smo in ['NONE', 'SMOTE', 'BorderlineSMOTE', 'SMOTENC', 'SVMSMOTE',
                    'KMeansSMOTE', 'ADASYN',
                    'CondensedNearestNeighbour', 'EditedNearestNeighbours',
                    'RepeatedEditedNearestNeighbours', 'AllKNN', 'RandomOverSampler',
                    'InstanceHardnessThreshold', 'NearMiss',
                    'NeighbourhoodCleaningRule', 'OneSidedSelection',
                    'RandomUnderSampler', 'TomekLinks', 'SMOTEENN', 'SMOTETomek', 'ClusterCentroids']:
            x = fscore(cls, smo, randomstate)
    print(current_best, best_cls, best_smo)
    ##save log file
    finallog = "gridsearch.csv"
    if (os.path.exists(finallog) == False):
        with open(finallog, "a") as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow(['dataset', 'randomstate', 'gmean', 'best_cls', 'best_smo'])
    with open(finallog, "a") as f:
        wr = csv.writer(f, dialect='excel')
        wr.writerow([dataset, randomstate, current_best, best_cls, best_smo])

local_log = pd.DataFrame(local_log)
header = ['dataset', 'randomstate', 'classifier', 'p_sub_type', 'gmean', 'current_best', 'e', 'ifError']
local_log.columns = header
local_log.to_csv("./grid_log/" + datasetname + "gridsearch_fulllog.csv")
