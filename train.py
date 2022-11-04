#!/usr/bin/python

import sys 
sys.path.append('library')
import extras as mylib
import numpy as np
import pandas as pd

##############################

labels_y=['CLASIFICACION_SPFMV','CLASIFICACION_SPCSV','CLASIFICACION_SPLCV'];
labels_x=[  'Coberture (%)_Avg','Coberture (%)_stdev',
            'BLUE_mean_avg'    ,'BLUE_mean_stdev',
            'GREEN_mean_avg'   ,'GREEN_mean_stdev',
            'NIR_mean_avg'     ,'NIR_mean_stdev',
            'RED_mean_avg'     ,'RED_mean_stdev',
            'RE_mean_avg'      ,'RE_mean_stdev',
            'TEMP_mean_avg'    ,'TEMP_mean_stdev'];

filename='dataset/All_data_Trial_SanRamon_custom.xlsx';
classifier_list=['RadomForest','SVM','KNN'];
category_list=['Resistant','Susceptible','Tolerant','Nothing'];
myseed=42;


##############################
data_dict=mylib.exel_to_dataframe_y(filename,
                                    sheet_name="TX_OUT", 
                                    labels=labels_y);

data_xxx2=mylib.excel_to_dataframe( filename,
                                    sheet_name="TX_Visit2",
                                    titles=labels_x);
data_xxx3=mylib.excel_to_dataframe( filename,
                                    sheet_name="TX_Visit3",
                                    titles=labels_x);
data_xxx4=mylib.excel_to_dataframe( filename,
                                    sheet_name="TX_Visit4",
                                    titles=labels_x);
data_xxx5=mylib.excel_to_dataframe( filename,
                                    sheet_name="TX_Visit5",
                                    titles=labels_x);

#data_xxxx=pd.concat([data_xxx2,data_xxx3,data_xxx4,data_xxx5],axis=1)
data_xxxx=pd.concat([data_xxx2,data_xxx3,data_xxx4],axis=1)

X=data_xxxx.to_numpy();


from sklearn.model_selection import train_test_split
for label in labels_y:
    print('########################################')
    print(label)
    print('########################################')
    for classifier_name in classifier_list:
        print(">>>",classifier_name)
        
        clf=mylib.create_classifier(name=classifier_name);
        
        mean_list=[];
        for category in category_list:
            #print(category)
            y=data_dict[label][category].to_numpy();
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.33, random_state=myseed);
            clf.fit(X_train,y_train);
            acc=clf.score(X_test,y_test);
            
            #print('acc', acc);
            
            mean_list.append(acc);
       
        mean_acc=np.mean(mean_list);
        print('mean_acc', mean_acc);
        
        print('')
        
    print('')

