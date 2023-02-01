#!/usr/bin/python

import pandas as pd

def excel_to_dataframe(filename,sheet_name=0,titles=None ):
    data=pd.read_excel(filename,sheet_name=sheet_name)  
    if titles!= None:
        data=data[titles];
    return data;


def exel_to_dataframe_y(filename,
                        sheet_name="TX_OUT", 
                        labels=['CLASIFICACION_SPFMV','CLASIFICACION_SPCSV','CLASIFICACION_SPLCV']):
    data_dict=dict();
    for label in labels:
        data=excel_to_dataframe(  filename,
                                  sheet_name=sheet_name,
                                  titles=[label]
                                  );
        data[label]=data[label].astype('category');
        dummies=pd.get_dummies(data, columns=[label],prefix='',prefix_sep='');
        data_dict[label]=dummies;
    
    return data_dict;
    
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

def create_classifier(name='RadomForest'):
    if name=='RandomForest':
        clf = RandomForestClassifier(n_estimators=10000,max_depth=7, random_state=0, mean_impurity_decrease=0.4);
    elif name=='SVM':
        clf = svm.SVC(kernel='rbf');
    elif name=='KNN':
        clf = KNeighborsClassifier(n_neighbors=5);
    else:
        clf = RandomForestClassifier(max_depth=7, random_state=0);
    return clf;
