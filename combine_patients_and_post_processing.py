# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 18:10:46 2021

@author: Sukrit Sharma ; sukritsharmap@gmail.com

This code combines previously created single patient voxel data (only Cr masked region).
Here WM mask is recreated to avoid Tumor region.
IDH positive is defined for tumor regions in IDH positive patients.
IDH is defined for PT region and NAWM seperately.
It also adds some new columns like regions. 

Cheers!

"""

# Packages and functions
import os
import numpy as np
import pandas as pd

# ****************************************************************************

# read single patient database table of all patients
patients_data_path='saved_files/single_patient_voxel_data/orig_and_ratio/'
dir_patients_data_folder = os.listdir(patients_data_path)

remaining_patients=len(dir_patients_data_folder)                                # define remaining patients
combined_data=pd.DataFrame()                                                    # empty dataframe to insert data later

# combine all single patients into a single database
for file in dir_patients_data_folder:
    if file.startswith('Tumor_Patient_'):
        if file.endswith('.csv'):
            # printing remaining patients
            remaining_patients = remaining_patients-1
            print('Remaining Patients =' + str(remaining_patients)) 
            
            # reading csv
            csv_file_path=patients_data_path + file
            csv_file=pd.read_csv(csv_file_path)
            
            # combining csv
#.................................................................................
            # only those voxels in creatine mask are combined
            data_cr_masked=csv_file[csv_file['Cr_mask']!=0]
            # split file name to remove .csv
            splitted=file.split('.')                                            
            orig_filename=splitted[0]
            
            # save creatine masked single patient data
            # save csv files
            try:
                # Create target Directory
                os.makedirs("saved_files/single_patient_voxel_data/Cr_masked/")
                print("Directory " , "saved_files/single_patient_voxel_data/Cr_masked/" ,  " created ") 
            except FileExistsError:
                print("Directory " , "saved_files/single_patient_voxel_data/Cr_masked/" ,  " already exists")
                
            output_filename= 'saved_files/single_patient_voxel_data/Cr_masked/' + orig_filename + '_Cr_masked.csv'
            data_cr_masked.to_csv(output_filename, header=True, index=False)
            
            # combine all patients 
            combined_data=combined_data.append(data_cr_masked)
#................................................................................


# make copy of combined database so that we do not change the one till here
voxel_data=combined_data.copy(deep=True)    

# create WM mask without tumor region ; should already be the case but to be sure
Segmentation_column_dummies=pd.get_dummies(voxel_data['Segmentation'])
Segmentation_column_dummies=Segmentation_column_dummies.drop([0],axis=1)
segmentation_mask=Segmentation_column_dummies.sum(axis=1)
wm_mask=voxel_data['WM_mask']
wm_mask=wm_mask-segmentation_mask
wm_mask_column=pd.get_dummies(wm_mask)
wm_mask_column=wm_mask_column.drop([-1,0],axis=1)
voxel_data['WM_mask']=wm_mask_column
  
# define IDH mutation for tumor regions only in IDH positive patients
voxel_data['IDH'].fillna(10,inplace=True)  # fill empty space with 10     
IDH_column=pd.get_dummies(voxel_data['IDH'])
IDH_column[10]=10*IDH_column[10]
IDH_column=IDH_column.drop(['no'],axis=1)
IDH_column=IDH_column.sum(axis=1)
IDH_column[IDH_column==10]=np.nan
# IDH_column=IDH_column*segmentation_mask
voxel_data['IDH']=IDH_column

#changing idh column seperating Peritumoral region and NAWM with IDH positive and negative
# voxel_data['IDH'][voxel_data[(voxel_data['Segmentation']==5)].index]=2
voxel_data.loc[(voxel_data['Segmentation'] == 5)  & (voxel_data['IDH'] == 1), ['IDH']] = 'PT1'
voxel_data.loc[(voxel_data['Segmentation'] == 5)  & (voxel_data['IDH'] == 0), ['IDH']] = 'PT0'
voxel_data.loc[(voxel_data['Segmentation'] == 0)  & (voxel_data['IDH'] == 1), ['IDH']] = 'NAWM1'
voxel_data.loc[(voxel_data['Segmentation'] == 0)  & (voxel_data['IDH'] == 0), ['IDH']] = 'NAWM0'

#-----------------------------------------------------------------------------
# IDH_info=voxel_data['IDH']*segmentation_mask
# IDH_info=pd.get_dummies(IDH_info)
# IDH_in_tumor=IDH_info.drop([0,1],axis=1)
# no_IDH_info=IDH_info.drop([2,3,4,5,6],axis=1)
# no_IDH_column=no_IDH_info.sum(axis=1)
# no_IDH_column=10*no_IDH_column

# IDH_in_tumor=IDH_in_tumor.drop([10,11,15],axis=1)
# IDH_in_tumor=IDH_in_tumor.sum(axis=1)
# IDH_in_tumor=IDH_in_tumor+no_IDH_column
# IDH_in_tumor[IDH_in_tumor==10]=np.nan
# voxel_data['IDH']=IDH_in_tumor
#------------------------------------------------------------------------------

# define Tumor VOI, PT VOI and NAWM VOI in regions column as 1, 2 and 3 and add it to database
seg_column=pd.get_dummies(voxel_data['Segmentation'])
tVOI=seg_column.drop([0,5],axis=1)
tVOI=tVOI.sum(axis=1)
pVOI=seg_column.drop([0,1,2,4],axis=1)
pVOI=pVOI.sum(axis=1)
pVOI=2*pVOI
wm_column=3*voxel_data['WM_mask']
regions_column=tVOI+pVOI+wm_column
regions_column[regions_column==1]='Tumor VOI'
regions_column[regions_column==2]='Peritumoral VOI'
regions_column[regions_column==3]='NAWM'
# selected_columns['Region']=regions_column
segmentation_column_index=voxel_data.columns.get_loc('Segmentation')
voxel_data.insert(segmentation_column_index+1,'Region',regions_column)

# add column with HG and LG definition in database
voxel_data=voxel_data.rename({'Easy_Diagnosis':'High_grade'},axis=1)
voxel_data['High_grade'][voxel_data['High_grade']==2.5]=3.0
grade_column_index=voxel_data.columns.get_loc('High_grade')
voxel_data.insert(grade_column_index+1,'Grade',voxel_data['High_grade'])
voxel_data['High_grade'].fillna(10,inplace=True)  # fill empty space with 10  
grade_dummy=pd.get_dummies(voxel_data['High_grade'])
# ----------------------------------------------------------------------------
# grade_two=grade_dummy.drop([2.5,3.0,4.0],axis=1)
# grade_two=2*grade_two
# grade_three=grade_dummy.drop([2.0,4.0],axis=1)
# grade_three=grade_three.sum(axis=1) 
# grade_three=3*grade_three
# grade_four=grade_dummy.drop([2.0,2.5,3.0],axis=1)
#----------------------------------------------------------------------------
grade_dummy=grade_dummy.drop([2.0],axis=1)
grade_column=grade_dummy.sum(axis=1)
voxel_data['High_grade']=grade_column

#Exceptionally for pat 56
voxel_data=voxel_data.drop(voxel_data[(voxel_data['Segmentation']==2) & (voxel_data['Grade']==2)].index)

# save csv files
try:
    # Create target Directory
    os.makedirs("saved_files/All_patient_data/")
    print("Directory " , "saved_files/All_patient_data/" ,  " created ") 
except FileExistsError:
    print("Directory " , "saved_files/All_patient_data/" ,  " already exists")
    
voxel_table_file_name_combi="saved_files/All_patient_data/"+"All_patient_metabolite_orig_and_ratio_voxel_data.csv"
voxel_data.to_csv(voxel_table_file_name_combi, header=True, index=False) 

number_of_columns=len(voxel_data.columns)
voxel_data_only_ratios= pd.concat([voxel_data.iloc[:,0:28],voxel_data.iloc[:,44 :number_of_columns+1]],axis=1)
voxel_table_file_name_ratio="saved_files/All_patient_data/"+"All_patient_metabolite_ratio_only_voxel_data.csv"
voxel_data_only_ratios.to_csv(voxel_table_file_name_ratio, header=True,index=False)
