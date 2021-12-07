#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:26:16 2021

@author: Sukrit Sharma 
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier

from numpy import mean
from numpy import std
from sklearn.feature_selection import RFECV

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import nibabel as nib
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

voxel_data = pd.read_csv('saved_files/All_patient_data/All_patient_metabolite_ratio_only_voxel_data.csv',low_memory=False)
voxel_data=voxel_data.drop(voxel_data[(voxel_data['Segmentation']==2) & (voxel_data['Grade']==2)].index)
# threshold_data=pd.read_csv('Threshold_new.csv',low_memory=False)

voxel_data_copy=voxel_data.copy(deep=True)

IDH_column_dummies=pd.get_dummies(voxel_data_copy['IDH'])
IDH_column_dummies=IDH_column_dummies.drop(['0.0','NAWM0','PT0'],axis=1)
IDH_column=IDH_column_dummies.sum(axis=1)
voxel_data_copy['IDH']=IDH_column
metabolites=voxel_data_copy.columns
number_of_columns=len(metabolites)
metabolites=metabolites[28:number_of_columns+1]


# metabolites=['GlnToNAA','GluToNAA','GlyToNAA','GPC+PChToNAA','InsToCr+PCr','GlyToCr+PCr','GPC+PChToCr+PCr','GluToCr+PCr','GlyToIns']

selected_columns_primary=voxel_data_copy[np.concatenate((['Index','Pat Code','IDH','Grade','High_grade','Segmentation','Region','WM_mask'],metabolites))]

selected_columns_primary=selected_columns_primary.drop(selected_columns_primary[ (selected_columns_primary['Segmentation']>4) | (selected_columns_primary['Segmentation']==0) ].index)
selected_columns_primary.replace(np.nan, 0, inplace=True)


threshold_data=pd.read_csv('Threshold_new.csv',low_memory=False)

upper_threshold=10

cutoff_options=[['0','saved_files/IDH_map_1/RFECV_RF_without_threshold/'],['threshold_data[metabolite_name].min()','saved_files/IDH_map_1/RFECV_RF_hotspot_min_filter/'],['threshold_data[metabolite_name].max()','saved_files/IDH_map_1/RFECV_RF_hotspot_max_filter/'],['threshold_data[metabolite_name].mean()','saved_files/IDH_map_1/RFECV_RF_hotspot_mean_filter/']]
# cutoff_options=[['threshold_data[metabolite_name].min()','saved_files/IDH_map_1/RFECV_RF_hotspot_min_filter/']]

# cutoff_options=[['threshold_data[metabolite_name].min()','saved_files/IDH_map/RFECV_RF_hotspot_min_filter/'],['threshold_data[metabolite_name].max()','saved_files/IDH_map/RFECV_RF_hotspot_max_filter/'],['threshold_data[metabolite_name].mean()','saved_files/IDH_map/RFECV_RF_hotspot_mean_filter/']]
cutoff_metabolites=['GPC+PChToNAA+NAAG','GlnToNAA+NAAG']
for opt in cutoff_options:
    threshold=opt[0]
    save_data_path = opt[1]
    try:
        # Create target Directory
        os.makedirs(save_data_path)
        print("Directory " , save_data_path ,  " created ") 
    except FileExistsError:
        print("Directory " , save_data_path ,  " already exists")
        
    selected_columns=selected_columns_primary.copy(deep=True)
    print("Number of rows before cutoff :" + str(len(selected_columns)))
    for metabolite_name in cutoff_metabolites:
       
        # lower_threshold=threshold_data[metabolite_name].mean()
        lower_threshold=0
        exec('lower_threshold='+threshold)
        print(lower_threshold)
        index_voxel_over_threshold = selected_columns[selected_columns[metabolite_name] >upper_threshold ].index
        index_voxel_below_threshold = selected_columns[selected_columns[metabolite_name] < lower_threshold ].index
        
        selected_columns=selected_columns.drop(index_voxel_over_threshold)
        selected_columns=selected_columns.drop(index_voxel_below_threshold)
        
        print("Number of after cutoff by limit of " + metabolite_name + ":" + str(len(selected_columns)))
        
    
    metabolites=list(metabolites)
    # random.shuffle(metabolites)
    features=selected_columns[metabolites]
    labels=np.array(selected_columns['IDH'])
    
    feature_list=list(features.columns)
    features=np.array(features)
    
    patients=pd.Categorical(selected_columns['Pat Code'])
    patients=patients.categories
    
    patients= ['Tumor_Patient_021','Tumor_Patient_023', 'Tumor_Patient_027', 
            'Tumor_Patient_030', 'Tumor_Patient_031', 'Tumor_Patient_034',
            'Tumor_Patient_035', 'Tumor_Patient_037', 'Tumor_Patient_038',
            'Tumor_Patient_039', 'Tumor_Patient_040', 'Tumor_Patient_046',
            'Tumor_Patient_050', 'Tumor_Patient_054', 'Tumor_Patient_055',
            'Tumor_Patient_056', 'Tumor_Patient_057', 'Tumor_Patient_058',
            'Tumor_Patient_059', 'Tumor_Patient_062', 'Tumor_Patient_063',
            'Tumor_Patient_065', 'Tumor_Patient_066', 'Tumor_Patient_067',
            'Tumor_Patient_068', 'Tumor_Patient_073', 'Tumor_Patient_074',
            'Tumor_Patient_078', 'Tumor_Patient_081', 'Tumor_Patient_084',
            'Tumor_Patient_086', 'Tumor_Patient_091', 'Tumor_Patient_093', 
            'Tumor_Patient_096', 'Tumor_Patient_098', 'Tumor_Patient_100',
            'Tumor_Patient_104'] 
    
    
    # patients=[ 'Tumor_Patient_055', 'Tumor_Patient_040', 'Tumor_Patient_057','Tumor_Patient_066','Tumor_Patient_078' ]
    
    whole_report=pd.DataFrame([],columns=['Patients','Number of Features','Classification Report'])
    
    
    
    training_set=selected_columns.copy(deep=True)
    testing_set=selected_columns.copy(deep=True)    
     
     
    # number_of_voxels=len(selected_columns[metabolite_name])
    # count_voxel_over_threshold=len(index_voxel_over_threshold)
    train_features=training_set[metabolites]
    train_labels=training_set['IDH']
    
    
    # Split the data into training and testing sets
    # train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.01, random_state = 42)
    
    print('RFECV Training Features Shape:', train_features.shape)
    print('RFECV Training Labels Shape:', train_labels.shape)
  


    # define dataset
    X=train_features
    # X=X.drop(['Cr+PCrToNAA+NAAG'],axis=1)
    y=train_labels
    features_list=X.columns

    number_features=len(X.columns)
    
    # cfeature selection
    rfecv = RFECV(estimator=RandomForestClassifier(n_estimators = 10, random_state = 1), step=1, scoring="roc_auc",n_jobs=-1,cv=5, verbose=3)
    
    rfecv_data =rfecv.fit(X,y)
    
    imp_features_list =features_list[rfecv.support_]
    
    imp_feature_df = pd.DataFrame()
    imp_feature_df["feature_imp"]=rfecv.estimator_.feature_importances_
    imp_feature_df["features"]=imp_features_list
    imp_feature_df = imp_feature_df.sort_values(by='feature_imp', ascending=False)
    
    rf = RandomForestClassifier(n_estimators = 100, random_state = 1)
    # n_scores = cross_val_score(rf, rfecv.transform(X), y, scoring='accuracy', cv=4, n_jobs=-1, error_score='raise')
    rf_data=rf.fit(rfecv.transform(X),y)
    
    
    
    for i in range(X.shape[1]):
     	print('Column: %s, Selected %s, Rank: %.3f, CV Score: %.3f' % (features_list[i], rfecv.support_[i], rfecv.ranking_[i], rfecv.grid_scores_[i]))
    print(str(rfecv.n_features_)+ ' are selected.')
    
    feature_selection_plot=plt.figure(figsize=(100,50),dpi=150)
    plt.rcParams.update({'font.size': 50})
    plt.title('RFECV - Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Feature importance')
    plt.bar(imp_feature_df['features'], imp_feature_df['feature_imp'])
    # plt.axvline(x=rfe.n_features_)
    # plt.xticks(range(0, len(rfecv.estimator_.feature_importances_) ), imp_feature_df["features"], rotation='vertical')
    plt.xticks(rotation='vertical')
    plt.show()
    feature_plot_name=save_data_path + 'feature_importance_plot.png'
    # feature_selection_plot.set_size_inches(30, 25)
    feature_selection_plot.savefig(feature_plot_name,bbox_inches='tight')
    
    feature_selection_plot=plt.figure(figsize=(100,50),dpi=150)
    plt.rcParams.update({'font.size': 50})
    plt.title('Feature Selection')
    plt.xlabel('Number of features selected')
    plt.ylabel('Cross validation score (nb of correct classifications)')
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ , linewidth=5)
    # plt.axvline(x=rfe.n_features_)
    plt.xticks(range(1, len(rfecv.grid_scores_) + 1), features_list, rotation='vertical')
    plt.show()
    feature_plot_name=save_data_path + 'feature_cv_plot.png'
    # feature_selection_plot.set_size_inches(30, 25)
    feature_selection_plot.savefig(feature_plot_name,bbox_inches='tight')
    
    metabolites_columns=voxel_data_copy[imp_features_list]

    plt.figure(figsize=(120,120), dpi=150)
    plt.rcParams.update({'font.size': 50})
    correlation_matrix=metabolites_columns.corr()
    sns.set(font_scale=8)
    correlation_matrix=sns.heatmap(correlation_matrix ,annot=True, linewidths=0.8, linecolor='black',annot_kws={"size": 7},cmap= 'Oranges',square=1)
    output_name=save_data_path +'correlation_matrix.png'
    correlation_matrix.figure.savefig(output_name,bbox_inches='tight')
        
    
    for pat in patients:
    
        print(pat)
        training_set=selected_columns.copy(deep=True)
        testing_set=selected_columns.copy(deep=True)
        pat_code=pat
        training_set=training_set.drop(training_set[(training_set['Pat Code']==pat_code)].index)
        # training_set=training_set.drop(training_set[ (training_set['Segmentation']>4) | (training_set['Segmentation']==0) ].index)
        testing_set=testing_set.drop(testing_set[(testing_set['Pat Code']!=pat_code)].index)
        # testing_set=testing_set.drop(testing_set[ (testing_set['Segmentation']>4) | (testing_set['Segmentation']==0) ].index)
        
        train_features=training_set[metabolites]
        train_labels=training_set['IDH']
        test_features=testing_set[metabolites]
        test_labels=testing_set['IDH']
        
        # Split the data into training and testing sets
        # train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.01, random_state = 42)
        
        print('RF Training Features Shape:', rfecv.transform(train_features).shape)
        print('RF Testing Features Shape:', rfecv.transform(test_features).shape)

        # define dataset
        X=train_features
        # X=X.drop(['Cr+PCrToNAA+NAAG'],axis=1)
        y=train_labels
        
    

        # training with selected feature
        rf = RandomForestClassifier(n_estimators = 100, random_state = 1)
        n_scores = cross_val_score(rf, rfecv.transform(X), y, scoring='accuracy', cv=4, n_jobs=-1, error_score='raise')
        rf_data=rf.fit(rfecv.transform(X),y)
        
        # testing with selected feature
        predictions_probability=rf.predict_proba(rfecv.transform(test_features))
        predictions_probability=predictions_probability[:,1]
        predictions=predictions_probability>0.5
        
        #create analysis table
        
        rfecv_df = pd.DataFrame(rfecv.ranking_,index=X.columns,columns=['Rank'])
        
        rfecv_df['Selected']=rfecv.support_
        rfecv_df['Feature Selection CV Score']=rfecv.grid_scores_
        rfecv_df['Model CV Accuracy']=mean(n_scores)
        rfecv_df['Model All AUC Scores']=str(n_scores)
        rfecv_df['Accuracy'] = accuracy_score(test_labels, predictions)
        rfecv_df['Classification Report']=classification_report(test_labels, predictions)
    
        feature_table_filename= save_data_path+pat_code + '_feature_table.csv'
        rfecv_df.to_csv(feature_table_filename, header=True, index=True)
        
        
        print("=== Confusion Matrix ===")
        print(confusion_matrix(test_labels,predictions))
        print('\n')
        print("=== Classification Report ===")
        print(classification_report(test_labels, predictions))
        print('\n')
        print("=== All AUC Scores ===")
        print(n_scores)
        print('\n')
        print("=== Mean AUC Score ===")
        print("Mean AUC Score - Random Forest: ", n_scores.mean())
        
        
        
        index_present=testing_set['Index']
        index_all = pd.Series(range(0,159744))
        index_absent= list(set(index_all)-set(index_present))
        df_absent=pd.DataFrame([])
        df_present=pd.DataFrame([])
        df_absent['Index']=index_absent
        df_absent['Prediction']=np.nan
        df_absent['Segmentation']=np.nan
        df_absent['Probability']=np.nan
        df_absent['Labels']=np.nan
        df_present['Index']=testing_set['Index']
        df_present['Prediction']=predictions
        df_present['Segmentation']=testing_set['Segmentation']
        df_present['Probability']=predictions_probability
        df_present['Labels']=test_labels
        
        
        df_all=df_present.append(df_absent)
        df_all=df_all.sort_values(by='Index')
        
        prediction_table_filename= save_data_path+pat_code + '_prediction_table.csv'
        df_present.to_csv(prediction_table_filename, header=True, index=False)
    
    
        
        
        
        sample_nii=nib.load('sample.nii')
        sample_affine=sample_nii.affine
        
        prediction_column=df_all.loc[:,['Probability']]
        segmentation_column = df_all.loc[:,['Segmentation']]
        prediction_array=prediction_column.to_numpy()
        segmentation_array=segmentation_column.to_numpy()
        prediction_3D=prediction_array.reshape(39,64,64)
        segmentation_3d=segmentation_array.reshape(39,64,64)
        prediction_3D_nii=nib.nifti1.Nifti1Image(prediction_3D,affine=sample_affine,file_map=sample_nii)
        plot_pdf_file_name=save_data_path+'maps_'+pat_code+'.pdf'
        plot_nii_file_name=save_data_path+'maps_'+pat_code+'.nii'
        plot_minc = PdfPages(plot_pdf_file_name)
        number_of_slices=prediction_3D.shape[0]
        for slice in range(0,number_of_slices):
            fig=plt.figure()
            color_map='tab20c'
            # plt.imshow(segmentation_3d[slice,:,:],cmap=color_map)
            plt.imshow(prediction_3D[slice,:,:],cmap=color_map,alpha=1)
            plt.colorbar()
            plt.plot()
            plot_minc.savefig(fig)
            plt.close()
        plot_minc.close()
        prediction_3D_nii.to_filename(plot_nii_file_name)
        
        sum_report=pd.DataFrame()
        sum_report['Patients']=[pat]
        sum_report['Number of Features']=[rfecv_data.n_features_]
        sum_report['Accuracy'] = accuracy_score(test_labels, predictions)
        sum_report['Classification Report']=[classification_report(test_labels, predictions)]
        sum_report['Model CV Accuracy']=mean(n_scores)
        sum_report['Model All AUC Scores']=str(n_scores)
        sum_report['Label']=np.array(testing_set['IDH'])[0]
        sum_report['Predicted Voxels'] = len(predictions)
        sum_report['IDH Positive'] = sum(predictions)
        sum_report['IDH Positive Percent'] = sum(predictions)/len(predictions) * 100
    
        whole_report=pd.concat([whole_report,sum_report])
        
    
    whole_report_filename= save_data_path+'whole_classification_report.csv'
    whole_report.to_csv(whole_report_filename, header=True, index=True)
        
