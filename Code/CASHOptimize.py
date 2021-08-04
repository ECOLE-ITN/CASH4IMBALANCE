#!/usr/bin/env python
# coding: utf-8

# In[1]:

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import pandas as pd
from pandas import Series, DataFrame
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.io.arff import loadarff
from scipy import interp
import json, logging, tempfile, sys, codecs, math, io, os,zipfile, arff, time, copy, csv,pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import roc_curve, auc
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials
from imblearn.over_sampling import (SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, SMOTENC,
                                    KMeansSMOTE, RandomOverSampler)
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss, TomekLinks,
                                     InstanceHardnessThreshold,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import geometric_mean_score
from scipy.spatial import distance


# In[2]:


randomstate = 9
file='../ds/imb_IRlowerThan9/ecoli-0_vs_1/ecoli-0_vs_1.dat'
dataset='ecoli-0_vs_1.dat'
HPOalg = 'TPE'

DataFolder='./logfiles'
HomeFolder='./finalresults'

##====SVM =====
HPOspace = hp.choice('classifier_type', [
    {
        'classifier': hp.choice('classifier',[
            {
                'name': 'SVM',
                'random_state': randomstate,
                'probability': hp.choice("probability", [True, False]),                
                'C': hp.uniform('C', 0.03125 , 200 ),
                'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),                
                "degree": hp.choice("degree", range(2, 5)),
                "gamma": hp.choice('gamma',['auto','value','scale']),
                'gamma_value': hp.uniform('gamma_value', 3.1E-05, 8),
                "coef0": hp.uniform('coef0', -1, 1),
                "shrinking": hp.choice("shrinking", [True, False]),
                "tol": hp.uniform('tol_svm', 1e-05, 1e-01)#NEW
            },
            {
                'name': 'RF',
                'n_estimators': hp.choice("n_estimators", range(1, 150)),
                'criterion': hp.choice('criterion', ["gini", "entropy"]),
                'max_features': hp.choice('max_features_RF', [1, 'sqrt','log2',None]),
                #If �sqrt�, then max_features=sqrt(n_features) (same as �auto�).
                #If �log2�, then max_features=log2(n_features).
                #If None, then max_features=n_features.
                #'max_depth': hp.choice('max_depth', range(10, 200)),                
                'min_samples_split': hp.choice('min_samples_split', range(2, 20)),
                'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 20)),
                'bootstrap': hp.choice('bootstrap',[True, False]),
                'class_weight':hp.choice('class_weight',['balanced','balanced_subsample',None]),
                'random_state': randomstate
            },
            {
                'name': 'KNN',
                'random_state': randomstate,
                'n_neighbors': hp.choice("n_neighbors_knn", range(1, 51)),
                'weights': hp.choice('weights', ["uniform", "distance"]),
                'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                #'leaf_size': hp.choice('leaf_size', range(1, 200)),
                #'p': hp.choice('p', [1,2]),
                'p': hp.choice("p_value", range(0, 20)),
            },
            {
                'name': 'DTC',
                'random_state': randomstate,
                'criterion': hp.choice("criterion_dtc", ["gini", "entropy"]),
                'max_features': hp.choice('max_features_dtc', [1, 'sqrt','log2',None]),
                #If �sqrt�, then max_features=sqrt(n_features) (same as �auto�).
                #If �log2�, then max_features=log2(n_features).
                #If None, then max_features=n_features.
                'max_depth': hp.choice('max_depth_dtc', range(2,20)),
                'min_samples_split': hp.choice('min_samples_split_dtc', range(2,20)),
                'min_samples_leaf': hp.choice('min_samples_leaf_dtc', range(1,20))     
            },
            {
                'name': 'LR',
                'random_state': randomstate,
                'C': hp.uniform('C_lr', 0.03125 , 100 ),
                'penalty_solver': hp.choice("penalty_lr", ["l1+liblinear","l1+saga","l2+newton-cg","l2+lbfgs",
                                                           "l2+liblinear","l2+sag","l2+saga","elasticnet+saga",
                                                           "none+newton-cg","none+lbfgs","none+sag","none+saga"]),
                #'dual': hp.choice("dual_lr", [True, False]),
                'tol': hp.uniform('tol_lr', 1e-05, 1e-01),
                #'solver': hp.choice('solver_lr', ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
                'l1_ratio': hp.uniform('l1_ratio', 1e-09, 1)
            }]),
            
        'sub' : hp.choice('resampling_type',[
            {
                'type':'NO',
                'smo_grp':'NO'
            },

            #Over sampling
            {
                'type': 'SMOTE',
                'smo_grp':'OVER',
                'k_neighbors': hp.choice('k_neighbors_SMOTE',range(1,10)),
                'random_state': randomstate
            },
            {
                'type': 'BorderlineSMOTE',
                'smo_grp':'OVER',
                'k_neighbors': hp.choice('k_neighbors_Borderline',range(1,10)),
                'm_neighbors': hp.choice('m_neighbors_Borderline',range(1,10)),
                'kind' :  hp.choice('kind', ['borderline-1', 'borderline-2']),
                'random_state': randomstate
            },
            {
                'type': 'SMOTENC',
                'smo_grp':'OVER',
                'categorical_features': True,
                'k_neighbors': hp.choice('k_neighbors_SMOTENC',range(1,10)), 
                'random_state': randomstate
            },
            {
                'type': 'SVMSMOTE',  
                'smo_grp':'OVER',
                'k_neighbors': hp.choice('k_neighbors_SVMSMOTE',range(1,10)), 
                'm_neighbors': hp.choice('m_neighbors_SVMSMOTE',range(1,10)),                
                'random_state': randomstate,
                'out_step': hp.uniform('out_step', 0, 1)
            },
            {
                'type': 'KMeansSMOTE',  
                'smo_grp':'OVER',
                'k_neighbors': hp.choice('k_neighbors_KMeansSMOTE',range(1,10)), 
                'cluster_balance_threshold': hp.uniform('cluster_balance_threshold', 1e-2, 1), 
                'random_state': randomstate                
            },
            {
                'type': 'ADASYN',
                'smo_grp':'OVER',
                'n_neighbors' : hp.choice('n_neighbors_ADASYN',range(1,10)),
                'random_state': randomstate
            },    
            {
                'type': 'RandomOverSampler',
                'smo_grp':'OVER',
                'random_state': randomstate
            },

            #COMBINE RESAMPLING
            {
                'type': 'SMOTEENN',
                'smo_grp':'COMBINE',
                'random_state': randomstate 
            },
            {
                'type': 'SMOTETomek',
                'smo_grp':'COMBINE',
                'random_state': randomstate
            },   
             #UNDER RESAMPLING
            {
                'type': 'CondensedNearestNeighbour',
                'smo_grp':'UNDER',
                'n_neighbors' : hp.choice('n_neighbors_CNN',range(1,50)),
                'n_seeds_S' : hp.choice('n_seeds_S_CNN',range(1,50)),
                'random_state': randomstate   
            },
            {
                'type': 'EditedNearestNeighbours',
                'smo_grp':'UNDER',
                'n_neighbors' : hp.choice('n_neighbors_ENN',range(1,20)),
                'kind_sel' : hp.choice('kind_sel_ENN',['all','mode']),
                #'random_state': randomstate
            },
            {
                'type': 'RepeatedEditedNearestNeighbours',
                'smo_grp':'UNDER',
                'n_neighbors' : hp.choice('n_neighbors_RNN',range(1,20)),
                'kind_sel' : hp.choice('kind_sel_RNN',['all','mode']),
            },
            {
                'type': 'AllKNN',
                'smo_grp':'UNDER',
                'n_neighbors' : hp.choice('n_neighbors_AKNN',range(1,20)),
                'kind_sel' : hp.choice('kind_sel_AKNN',['all','mode']),
                'allow_minority' : hp.choice('allow_minority_AKNN', [True, False])
            },
            {
                'type': 'InstanceHardnessThreshold',
                'smo_grp':'UNDER',
                'estimator': hp.choice('estimator_IHTh', ['knn', 'decision-tree', 'adaboost','gradient-boosting','linear-svm', None]),
                'cv' : hp.choice('cv_IHTh',range(2,10)),                
                'random_state': randomstate   
            },
            {
                'type': 'NearMiss',
                'smo_grp':'UNDER',
                'version' : hp.choice('version_NM',range(1,3)),
                'n_neighbors' : hp.choice('n_neighbors_NM',range(1,20)),
                'n_neighbors_ver3' : hp.choice('n_neighbors_ver3_NM',range(1,20))              
            },
            {
                'type': 'NeighbourhoodCleaningRule',
                'smo_grp':'UNDER',
                'n_neighbors' : hp.choice('n_neighbors_NCR',range(1,20)),
                'threshold_cleaning' : hp.uniform('threshold_cleaning_NCR',0,1)
            },
            {
                'type': 'OneSidedSelection',
                'smo_grp':'UNDER',
                'n_neighbors' : hp.choice('n_neighbors_OSS',range(1,20)),
                'n_seeds_S' : hp.choice('n_seeds_S_OSS',range(1,20)),
                'random_state': randomstate
            },
            {
                'type': 'RandomUnderSampler',
                'smo_grp':'UNDER',
                'replacement' : hp.choice('replacement_RUS', [True, False]),                
                'random_state': randomstate
            },
            {
                'type': 'TomekLinks',
                'smo_grp':'UNDER',
                #'random_state': randomstate
            },
            {
                'type': 'ClusterCentroids',
                'smo_grp':'UNDER',
                'estimator': hp.choice('estimator_CL',['KMeans', 'MiniBatchKMeans']),
                'voting' : hp.choice('voting_CL',['hard', 'soft']),
                'random_state': randomstate
            }

           
        ])
    }
])


# In[4]:


def fscore(params_org):
    #print(params_org)
    parambk = copy.deepcopy(params_org)
    ifError =0
    global best, HPOalg,params_best, errorcount
    params= params_org['classifier']
    classifier = params.pop('name')
    p_random_state = params.pop('random_state')
    
    if (classifier == 'SVM'):  
        param_value= params.pop('gamma_value')
        if(params['gamma'] == "value"):
            params['gamma'] = param_value
        else:
            pass   
        clf = SVC(max_iter = 10000, cache_size= 700, random_state = p_random_state,**params)
        #max_iter=10000 and cache_size= 700 https://github.com/EpistasisLab/pennai/issues/223
        #maxvalue https://github.com/hyperopt/hyperopt-sklearn/blob/fd718c44fc440bd6e2718ec1442b1af58cafcb18/hpsklearn/components.py#L262
    elif(classifier == 'RF'):        
        clf = RandomForestClassifier(random_state = p_random_state, **params)
    elif(classifier == 'KNN'):
        p_value = params.pop('p')
        if(p_value==0):
            params['metric'] = "chebyshev"
        elif(p_value==1):
            params['metric'] = "manhattan"
        elif(p_value==2):
            params['metric'] = "euclidean"
        else:
            params['metric'] = "minkowski"
            params['p'] = p_value
        #https://github.com/hyperopt/hyperopt-sklearn/blob/fd718c44fc440bd6e2718ec1442b1af58cafcb18/hpsklearn/components.py#L302
        clf = KNeighborsClassifier(**params)
    elif(classifier == 'DTC'):        
        clf = DecisionTreeClassifier(random_state = p_random_state, **params)
    elif(classifier == 'LR'):        
        penalty_solver = params.pop('penalty_solver')
        params['penalty'] = penalty_solver.split("+")[0]
        params['solver'] = penalty_solver.split("+")[1]
        clf = LogisticRegression(random_state = p_random_state, **params)
    #resampling parameter
    p_sub_params= params_org.pop('sub')
    p_sub_type = p_sub_params.pop('type')
    sampler = p_sub_params.pop('smo_grp')
    gmean = []
    if (p_sub_type == 'SMOTE'):
        smo = SMOTE(**p_sub_params)
    elif (p_sub_type == 'ADASYN'):
        smo = ADASYN(**p_sub_params)
    elif (p_sub_type == 'BorderlineSMOTE'):
        smo = BorderlineSMOTE(**p_sub_params)
    elif (p_sub_type == 'SVMSMOTE'):
        smo = SVMSMOTE(**p_sub_params)
    elif (p_sub_type == 'SMOTENC'):
        smo = SMOTENC(**p_sub_params)
    elif (p_sub_type == 'KMeansSMOTE'):
        smo = KMeansSMOTE(**p_sub_params)
    elif (p_sub_type == 'RandomOverSampler'):
        smo = RandomOverSampler(**p_sub_params)
#Undersampling
    elif (p_sub_type == 'TomekLinks'):
        smo = TomekLinks(**p_sub_params)
    elif (p_sub_type == 'ClusterCentroids'):
        if(p_sub_params['estimator']=='KMeans'):
            p_sub_params['estimator']= KMeans(random_state = p_random_state)
        elif(p_sub_params['estimator']=='MiniBatchKMeans'):
            p_sub_params['estimator']= MiniBatchKMeans(random_state = p_random_state)
        smo = ClusterCentroids(**p_sub_params) 
    elif (p_sub_type == 'RandomUnderSampler'):
        smo = RandomUnderSampler(**p_sub_params)
    elif (p_sub_type == 'NearMiss'):
        smo = NearMiss(**p_sub_params)
    elif (p_sub_type == 'InstanceHardnessThreshold'):
        if(p_sub_params['estimator']=='knn'):
            p_sub_params['estimator']= KNeighborsClassifier()
        elif(p_sub_params['estimator']=='decision-tree'):
            p_sub_params['estimator']=DecisionTreeClassifier()
        elif(p_sub_params['estimator']=='adaboost'):
            p_sub_params['estimator']=AdaBoostClassifier()
        elif(p_sub_params['estimator']=='gradient-boosting'):
            p_sub_params['estimator']=GradientBoostingClassifier()
        elif(p_sub_params['estimator']=='linear-svm'):
            p_sub_params['estimator']=CalibratedClassifierCV(LinearSVC())
        elif(p_sub_params['estimator']=='random-forest'):
            p_sub_params['estimator']=RandomForestClassifier(n_estimators=100)
        smo = InstanceHardnessThreshold(**p_sub_params) 
    elif (p_sub_type == 'CondensedNearestNeighbour'):
        smo = CondensedNearestNeighbour(**p_sub_params)
    elif (p_sub_type == 'EditedNearestNeighbours'):
        smo = EditedNearestNeighbours(**p_sub_params)
    elif (p_sub_type == 'RepeatedEditedNearestNeighbours'):
        smo = RepeatedEditedNearestNeighbours(**p_sub_params) 
    elif (p_sub_type == 'AllKNN'):
        smo = AllKNN(**p_sub_params)
    elif (p_sub_type == 'NeighbourhoodCleaningRule'):
        smo = NeighbourhoodCleaningRule(**p_sub_params) 
    elif (p_sub_type == 'OneSidedSelection'):
        smo = OneSidedSelection(**p_sub_params)
#Combine
    elif (p_sub_type == 'SMOTEENN'):
        smo = SMOTEENN(**p_sub_params)
    elif (p_sub_type == 'SMOTETomek'):
        smo = SMOTETomek(**p_sub_params)
    e=''
    try:        
        for train, test in cv.split(X, y):
            if(p_sub_type=='NO'):
                X_smo_train, y_smo_train = X[train], y[train]
            else:
                X_smo_train, y_smo_train = smo.fit_sample(X[train], y[train])
            y_test_pred = clf.fit(X_smo_train, y_smo_train).predict(X[test])
            gm = geometric_mean_score(y[test], y_test_pred, average='binary')
            gmean.append(gm)
        mean_g=np.mean(gmean)
    except Exception as eec:
        e=eec
        mean_g = 0
        ifError =1 
        errorcount = errorcount+1
    gm_loss = 1 - mean_g
    abc=time.time()-starttime
    if mean_g > best:
        best = mean_g
        params_best = copy.deepcopy(parambk)
    return {'loss': gm_loss,
            'mean': mean_g,
            'status': STATUS_OK,         
            # -- store other results like this
            'run_time': abc,
            'iter': iid,
            'current_best': best,
            'eval_time': time.time(),            
            'SamplingGrp': sampler,
            'SamplingType': p_sub_type,
            'ifError': ifError,
            'Error': e,
            'params' : parambk,
            'attachments':
                {'time_module': pickle.dumps(time.time)}
           }   


# In[5]:


best,params_best = 0,''
cv = []
print('\033[91m',HPOalg,'==Random Seed:',randomstate,'=== START DATASET: ', dataset, '=======', '\033[0m')
##If you read data directy from the original KEEL's zip file, use this code:
'''zf = zipfile.ZipFile(file) 
in_mem_fo = io.TextIOWrapper(io.BytesIO(zf.read(dataset)), encoding='utf-8')
####Convert KEEL format to arff ###
txt=''
for i in in_mem_fo:
    if("@input" not in i and '@output' not in i ):
        newi=''
        for e in i.split(','):
            if(newi==''):
                newi=e.strip()
            else:
                newi=newi+','+e.strip()
        txt=txt+newi+'\n'
abc=io.StringIO(txt)
data,meta=loadarff(abc)
data=pd.DataFrame(data)
enc = LabelEncoder()
#######LOAD TRAIN DATA######
for a in [ col  for col, data in data.dtypes.items() if data == object]:
    #print(a)
    data[a] = data[a].str.decode('utf-8') 
    try:
        data[a]=data[a].astype('int64')
        #print(a)
    except:
        data[a] = enc.fit_transform(data[a])
'''
#######LOAD TRAIN DATA######
data, meta = loadarff(file)
data = pd.DataFrame(data)
enc = LabelEncoder()
### Encoding ###
for a in [col for col, data in data.dtypes.items() if data == object]:
    data[a] = enc.fit_transform(data[a])
X= data[data.columns[:-1]].to_numpy()
y= data[data.columns[-1]].to_numpy()
X = StandardScaler().fit_transform(X)
X = np.c_[X]
k = 5        
cv = StratifiedKFold(n_splits=k, random_state=randomstate)
space = HPOspace           
trials = Trials()
starttime = time.time()
ran_best = 0 
best = 0        
iid = 0
errorcount=0
rstate=np.random.RandomState(randomstate)
if (HPOalg == 'Random'):
    suggest=rand.suggest
else:
    suggest=tpe.suggest
try:
    xOpt= fmin(fscore, space, algo=suggest, max_evals=500, trials=trials, rstate=rstate)
except:
    print('==ERROR: RANDOM-',dataset,'===')
runtime=time.time()-starttime

try:
    ran_results = pd.DataFrame({'current_best': [x['current_best'] for x in trials.results],
                                'run_time':[x['run_time'] for x in trials.results],
                                'SamplingGrp': [x['SamplingGrp'] for x in trials.results],
                                'SamplingType': [x['SamplingType'] for x in trials.results], 
                                'ifError': [x['ifError'] for x in trials.results], 
                                'Error': [x['Error'] for x in trials.results], 
                                'loss': [x['loss'] for x in trials.results], 
                                'mean': [x['mean'] for x in trials.results], 
                                'iteration': trials.idxs_vals[0]['classifier_type'],
                                'params':[x['params'] for x in trials.results]})
    ran_results.to_csv(DataFolder+'/hyperopt_'+HPOalg+'_'+dataset+'_'+str(randomstate)+'.csv', index = True, header=True)
except:
    print('ERROR: No logfile')
finallog= HomeFolder+"/hyperopt_finallog.csv"
if (os.path.exists(finallog)==False):
    with open(finallog, "a") as f:    
        wr = csv.writer(f, dialect='excel')
        wr.writerow(['dataname','HPOalg','random_state','mean', 'params','runtime','errorcount'])
with open(finallog, "a") as f:
    wr = csv.writer(f, dialect='excel')
    wr.writerow([dataset,HPOalg,randomstate,best,params_best,runtime,errorcount])

