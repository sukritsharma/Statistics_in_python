#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 23:02:49 2021


@author: local
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

data_paths=[
            ['tCho/tNAA as feature with mean Threshold','saved_files/IDH_map_1/RFECV_RF_single_feature_hotspot_mean_filter/'],
            ['tCho/tNAA as feature with max Threshold','saved_files/IDH_map_1/RFECV_RF_single_feature_hotspot_max_filter/'],
            ['tCho/tNAA as feature with min Threshold','saved_files/IDH_map_1/RFECV_RF_single_feature_hotspot_min_filter/'],
            ['tCho/tNAA as feature without Threshold','saved_files/IDH_map_1/RFECV_RF_single_feature_without_threshold/'],
            
            ['Without Threshold','saved_files/IDH_map_1/RFECV_RF_without_threshold/'],
            ['With Threshold Min','saved_files/IDH_map_1/RFECV_RF_hotspot_min_filter/'],
            ['With Threshold Max','saved_files/IDH_map_1/RFECV_RF_hotspot_max_filter/'],
            ['With Threshold Mean','saved_files/IDH_map_1/RFECV_RF_hotspot_mean_filter/'],
            
            # ['Without Threshold Union','saved_files/IDH_map/Union_with_two_metabolites/RFECV_RF_without_threshold_union/'],
            # ['With Threshold Min Union','saved_files/IDH_map/Union_with_two_metabolites/RFECV_RF_hotspot_min_filter_union/'],
            # ['With Threshold Max Union','saved_files/IDH_map/Union_with_two_metabolites/RFECV_RF_hotspot_max_filter_union/'],
            # ['With Threshold Mean Union','saved_files/IDH_map/Union_with_two_metabolites/RFECV_RF_hotspot_mean_filter_union/'],
            
            # ['Without Threshold Union','saved_files/IDH_map/Union_with_five_metabolites/RFECV_RF_without_threshold_union/'],
            # ['With Threshold Min Union','saved_files/IDH_map/Union_with_five_metabolites/RFECV_RF_hotspot_min_filter_union/'],
            # ['With Threshold Max Union','saved_files/IDH_map/Union_with_five_metabolites/RFECV_RF_hotspot_max_filter_union/'],
            # ['With Threshold Mean Union','saved_files/IDH_map/Union_with_five_metabolites/RFECV_RF_hotspot_mean_filter_union/']
            
            ]
fig_roc_voxelwise=plt.figure(1,figsize=(15, 10), dpi=80)
fig_roc_patientwise=plt.figure(2,figsize=(15, 10), dpi=80)
fig_roc_patientwise=plt.figure(3,figsize=(15, 10), dpi=80)

for data_path in data_paths:
    lab=data_path[0]
    patients_data_path=data_path[1]
    try:
        # Create target Directory
        os.makedirs(patients_data_path)
        print("Directory " , patients_data_path ,  " created ") 
    except FileExistsError:
        print("Directory " , patients_data_path ,  " already exists")
    dir_patients_data_folder = os.listdir(patients_data_path)
    # remaining_patients=len(dir_patients_data_folder)
    combined_data=pd.DataFrame()
    patients_list=[]
    average_prediction=[]
    for file in dir_patients_data_folder:
        if file.startswith('Tumor_Patient_'):
            if file.endswith('_prediction_table.csv'):
                print(file)
                # remaining_patients = remaining_patients-1
                # print('Remaining Patients =' + str(remaining_patients)) 
                csv_file_path=patients_data_path + file
                csv_file=pd.read_csv(csv_file_path)
    #.................................................................................
                splitted=file.split('_')
                number=splitted[2]  
                pat_name='Tumor_Patient_'+str(number)
                csv_file.insert(0,'Pat Code',pat_name)
                # selected_row=csv_file.iloc[[2]]
                combined_data=combined_data.append(csv_file)
                
                patients_list.append(pat_name)
                average_prediction.append(csv_file['Probability'].mean())
                
                
    #................................................................................
    
    mean_prediction=pd.DataFrame()
    mean_prediction['Mean_prediction']=average_prediction
    mean_prediction['Pat Code']=patients_list
    
    test_labels=combined_data['Labels']
    # test_labels=test_labels==test_labels
    
    predictions_probability=combined_data['Probability']
    
    
    fig_roc_voxelwise=plt.figure(1)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(test_labels, predictions_probability)
    AUC=auc(fpr,tpr)
    # plot the roc curve for the model
    
    plt.plot(fpr, tpr, marker='.', label=lab + " : AUC = " + str(round(AUC,2)))
    ####################################################################################
    csv_file_path=patients_data_path + 'whole_classification_report.csv'
    csv_file=pd.read_csv(csv_file_path)
  
    test_labels=csv_file['Label']
    predictions_probability=csv_file['IDH Positive Percent']/100
    AUC=auc(fpr,tpr)
    fig_roc_patientwise=plt.figure(2)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(test_labels, predictions_probability)
    AUC=auc(fpr,tpr)
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.', label=lab + " : AUC = " + str(round(AUC,2)))
  ############################################################################################### 
    test_labels=csv_file['Label']
    # test_labels=test_labels==test_labels
    mean_predictions_probability=mean_prediction['Mean_prediction']
    fig_roc_patientwise_mean=plt.figure(3)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(test_labels, mean_predictions_probability)
    AUC=auc(fpr,tpr)
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.', label=lab + " : AUC = " + str(round(AUC,2)))
# axis labels
fig_roc_voxelwise=plt.figure(1)
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(label='ROC Voxelwise')
# show the legend
plt.legend(loc=4)
# show the plot
fig_roc_voxelwise.show()


fig_roc_patientwise=plt.figure(2)
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(label='ROC Patientwise based on IDH Prediction Probability')
# show the legend
plt.legend(loc=4)
# show the plot
fig_roc_patientwise.show()

fig_roc_patientwise_mean=plt.figure(3)
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(label='ROC Patientwise based on Mean Voxel Prediction')
# show the legend
plt.legend(loc=4)
# show the plot
fig_roc_patientwise_mean.show()

try:
    # Create target Directory
    os.makedirs("saved_files/images")
    print("Directory " , "saved_files/images" ,  " created ") 
except FileExistsError:
    print("Directory " , "saved_files/images" ,  " already exists")


fig_roc_voxelwise.savefig('saved_files/images/fig_roc_voxelwise_v01.png')
fig_roc_patientwise.savefig('saved_files/images/fig_roc_IDH_Prediction_probability_v01.png')
fig_roc_patientwise_mean.savefig('saved_files/images/fig_roc_patientwise_mean_v01.png')

# predictions=predictions_probability>0.5
# Report=classification_report(test_labels, predictions)
# Confusion_matrix=confusion_matrix(test_labels,predictions)
# print(Report)
# print(Confusion_matrix)

# combined_data.to_csv('combined_data_1.csv', header=True, index=False)