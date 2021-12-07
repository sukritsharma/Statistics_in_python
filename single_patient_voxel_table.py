
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 18:13:21 2021

@author: Sukrit Sharma ; sukritsharmap@gmail.com

This code deals with creating table out of MRSI maps, for each patients with 
all the imoprtant MRSI maps and some other extra maps.
The code takes data path of patients with MRSI maps and list of patients with 
histology informations as input and gives the databse table with all metabolites 
maps and extra maps in csv format as output.

Cheers!
"""

# Packages and functions
import os
import numpy as np
import pandas as pd
# import nibabel as nib
# import csv
# import matplotlib.pyplot as plt
from functions import map_1D  
from functions import add_peritumoral_segment
from functions import outliers_filter

# ****************************************************************************


# find all the patients for processing

home_path=os.path.expanduser('~')
patients_data_path =home_path+'/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Process_Results/Tumor_Patients/Reprocessed_Stateval_2021/'
dir_patients_data_folder = os.listdir(patients_data_path)

# read csv file with patients list and histology data 
patients_list= pd.read_csv('patients_list_06_2021.csv',sep=',',na_values='.')
col_names_patients_list=patients_list.columns                                   # columns of patient list table
patients_code = patients_list['Pat Code']  
                                     


# patients_code=['Tumor_Patient_021']                                           # For single patient test purpose

# extract all patient numbers from patient code
splitted_patients_number=[]
for pats in patients_code:
    splitted=pats.split('_')
    number=splitted[2]                                                          #taking only numeric part
    number =str(int(number))                                                    #removind zero in front of numeric part   
    splitted_patients_number.append(number)                                     # add all the numbers to splitted_patients_number
    


# find remaining patients while processing    
remaining_patients = len(splitted_patients_number)
voxel_data_table=pd.DataFrame()
row_index_count=0

# compute data table for each patient
for patnum in splitted_patients_number:
    pat_voxel_data_table = pd.DataFrame()                                      # single patients voxel data table
# =============================================================================
#     pat_voxel_data_table = pd.DataFrame(columns=patients_list.columns) 
#     data_patients_list = patients_list.loc[row_index_count]
#     pat_voxel_data_table=pat_voxel_data_table.append(data_patients_list)
# =============================================================================

    # print number of remaining patients to be processed and patient number currently being processed
    print (str(remaining_patients),'Remaining')                                 
    remaining_patients = remaining_patients - 1
    print (patnum)    
    
    # getting maps of individual patients
    
    file_index = 'Tumor_Patient_' + patnum
    
    # define patients folder with maps
    for file in dir_patients_data_folder:                                       
        if file.startswith(file_index):
            patient_data_file=file

    
    
    maps_origin_path = patients_data_path + patient_data_file + '/maps/Orig/'   #define origin map data path
    dir_maps_orig=os.listdir(maps_origin_path)                                  # read all the maps in orig folder
    original_maps=[]                                                            #create space for original maps 
    for maps in dir_maps_orig :                                                 # loop for all maps
        if  maps.endswith('amp_map.mnc'):
            original_maps.append(maps)                                          # filtering minc files
    # Remove some metabolites
    original_maps.remove('Cr_amp_map.mnc')
    original_maps.remove('Scyllo_amp_map.mnc')
    original_maps.remove('GPC_amp_map.mnc')
    original_maps.remove('PCh_amp_map.mnc')
    original_maps.remove('PCr_amp_map.mnc')
    original_maps.remove('TwoHG_amp_map.mnc')
    # original_maps.remove('MM_amp_map.mnc')
    original_maps.remove('Cys_amp_map.mnc')
    
    original_maps.sort()                                                        # sort maps alphabetically

    # data path for some extra maps          
    Cr_mask_path = patients_data_path + patient_data_file + '/maps/Cr_Masked/mask_cr.mnc'
    segmentation = patients_data_path + patient_data_file + "/maps/segmentations/segmentation_resampled.nii"                                  # creatine mask
    WM_mask = patients_data_path + patient_data_file + '/maps/mask/WM_mask.nii'
    WM = patients_data_path + patient_data_file + '/maps/Extra/WM_CSI_map.mnc'
    GM = patients_data_path + patient_data_file + '/maps/Extra/GM_CSI_map.mnc'
    # PET= patients_data_path + patient_data_file + '/maps/ct_pet/ct_pet_resampled.mnc'
    
    
    # add metabolite maps into database 
    for orig_map in original_maps:                                               
         split_name=orig_map.split('_')                                         # extraction metabolites name
         met_name = split_name[0]
         
         orig_met_path=maps_origin_path + orig_map                              
         met_map=map_1D(orig_met_path)                                          #get metabolites map as 1D array
         Cr_mask_map=map_1D(Cr_mask_path)                                       #get creatine mask
         
         met_maps_Cr_masked = met_map * Cr_mask_map
         ######################################################################
         # met_maps_Cr_masked=outliers_filter (met_maps_Cr_masked,std_coefficient=15)
         # print(orig_map)
         ######################################################################
         pat_voxel_data_table[met_name] = met_maps_Cr_masked
    
    # add extra maps into database
    pat_voxel_data_table["Segmentation"] = add_peritumoral_segment(segmentation) # add segmentation with peritumoral region
    pat_voxel_data_table["WM_mask"] = map_1D(WM_mask)                           
    pat_voxel_data_table["WM"] = map_1D(WM)
    pat_voxel_data_table["GM"] = map_1D(GM)
    pat_voxel_data_table["Cr_mask"]=map_1D(Cr_mask_path)
    # pat_voxel_data_table["PET"]=map_1D(PET)
    
    # create coordinates for map so that we can recreate maps later, based on coordinates
    map_shape=(39, 64, 64)                                                      # maps dimension definition
    pat_dim = pd.DataFrame({'coordinate:z':[],'coordinate:y':[],'coordinate:x':[]})  # dimensions column-name 
    for x in range(map_shape[0]):
        for y in range(map_shape[1]):
            for z in range(map_shape[2]):
                new_dim={'coordinate:z':x+1,'coordinate:y':y+1,'coordinate:x':z+1}
                pat_dim =pat_dim.append(new_dim,ignore_index=True)
                
    # add dimensions to database            
    pat_voxel_data_table=pd.concat([pat_dim,pat_voxel_data_table],axis=1)
              
    # add patients info from patients list into database where patients info repeat for each voxel (row) 
    data_patients_list = patients_list.loc[row_index_count]
    pos=0   
    for col_nam in patients_list.columns :                                      # inserting patients list in front of data table
        pat_voxel_data_table.insert(pos,col_nam,data_patients_list[col_nam],True)
        pos= pos + 1   
    row_index_count = row_index_count + 1                                       # defining row of patient list for next patient in loop
   
    voxel_data_table= pat_voxel_data_table
    # =========================================================================
    # used to remove columns from data base which were later removed before adding it to database
    
    # voxel_data_table=voxel_data_table.drop(['GPC'], axis=1)
    # voxel_data_table=voxel_data_table.drop(['PCh'], axis=1)
    # # voxel_data_table=voxel_data_table.drop(['Cr'], axis=1)
    # voxel_data_table=voxel_data_table.drop(['PCr'], axis=1)
    # voxel_data_table=voxel_data_table.drop(['TwoHG'], axis=1)
    # voxel_data_table=voxel_data_table.drop(['MM'], axis=1)
    # voxel_data_table=voxel_data_table.drop(['Cys'], axis=1)
    # ========================================================================
    
    
    
    
# *****************************************************************************
    # Create ratio from voxel table with orignal maps only
    print ("Creating Ratios") 
    
    
    
    metabolites_voxel_table = voxel_data_table                                 # defining database table as metabolite table
    number_of_columns= len(metabolites_voxel_table.columns)
    
    # seperate table into two parts: one with only metabolites to calculate ratios out of them and next with other columns except metabolites
    metabolites_voxel_table_only_data = metabolites_voxel_table.iloc[:,20:number_of_columns-5]
    metabolites_voxel_table_except_data = pd.concat([metabolites_voxel_table.iloc[:,0:20],metabolites_voxel_table.iloc[:,number_of_columns-5:number_of_columns]],axis=1)
   
    # metabolites_voxel_table_only_data ["TwoHG+Gln"] = metabolites_voxel_table_only_data["TwoHG"] + metabolites_voxel_table_only_data["Gln"]
    # metabolites_voxel_table_only_data ["GABA+Glu"] = metabolites_voxel_table_only_data["GABA"] +metabolites_voxel_table_only_data["Glu"]
    
    column_names = metabolites_voxel_table_only_data.columns
    
    # Calculating different ratios
    # Ratio to NAA
    for met in column_names :
        col_name = met + "ToNAA"
        metabolites_voxel_table_only_data [col_name] = metabolites_voxel_table_only_data[met] / metabolites_voxel_table_only_data["NAA"]
    metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['NAAToNAA'], axis=1)
    metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['NAA+NAAGToNAA'], axis=1)
    metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['NAAGToNAA'], axis=1)
    # metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['GPCToNAA'], axis=1)
    # metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['PChToNAA'], axis=1)
    
    # Ratio to NAA+NAAG
    for met in column_names :
        col_name = met + "ToNAA+NAAG"
        metabolites_voxel_table_only_data [col_name] = metabolites_voxel_table_only_data[met] / metabolites_voxel_table_only_data["NAA+NAAG"]
    metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['NAA+NAAGToNAA+NAAG'], axis=1) 
    metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['NAAToNAA+NAAG'], axis=1)
    metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['NAAGToNAA+NAAG'], axis=1)
    # metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['GPCToNAA+NAAG'], axis=1)
    # metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['PChToNAA+NAAG'], axis=1)
    
    # Ratio to Cr+PCr
    for met in column_names :
        col_name = met + "ToCr+PCr"
        metabolites_voxel_table_only_data [col_name] = metabolites_voxel_table_only_data[met] / metabolites_voxel_table_only_data["Cr+PCr"]
    metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['Cr+PCrToCr+PCr'], axis=1)
    # metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['PCrToCr+PCr'], axis=1)    
    # metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['GPCToCr+PCr'], axis=1)
    # metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['PChToCr+PCr'], axis=1)
    
    # Ratio TwoHG to metabolites
    # for met in column_names :
    #     col_name = "TwoHGTo" + met
    #     metabolites_voxel_table_only_data [col_name] = metabolites_voxel_table_only_data["TwoHG"] / metabolites_voxel_table_only_data[met]
    # metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['TwoHGToTwoHG'], axis=1)  
    # # metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['TwoHGToGPC'], axis=1) 
    # # metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['TwoHGToPCh'], axis=1) 
    
    # # Ratio metabolites to GPC+PCh
    for met in column_names :
        col_name = met +"ToGPC+PCh"
        metabolites_voxel_table_only_data [col_name] =  metabolites_voxel_table_only_data[met] / metabolites_voxel_table_only_data["GPC+PCh"]
    metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['GPC+PChToGPC+PCh'], axis=1)
    # metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['GPC+PChToGPC'], axis=1)
    # metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.drop(['GPC+PChToPCh'], axis=1)
    
    # Ratio Gly to Ins
    metabolites_voxel_table_only_data ["GlyToIns"] = metabolites_voxel_table_only_data["Gly"] /metabolites_voxel_table_only_data["Ins"]
     
    # Ratio Gly To Ins+Gly
    metabolites_voxel_table_only_data ["GlyToGly+Ins"] = metabolites_voxel_table_only_data["Gly"] /metabolites_voxel_table_only_data["Ins+Gly"]
     
    # Ratio Gln To Glu+Gln
    metabolites_voxel_table_only_data ["GlnToGlu+Gln"] = metabolites_voxel_table_only_data["Gln"] /metabolites_voxel_table_only_data["Glu+Gln"]
     
    # Ratio Glu To Glu+Gln
    metabolites_voxel_table_only_data ["GluToGlu+Gln"] = metabolites_voxel_table_only_data["Glu"] /metabolites_voxel_table_only_data["Glu+Gln"]
    
    # Replace inf in database with 0
    metabolites_voxel_table_only_data.replace([np.inf, -np.inf], 0, inplace=True)
    # Replace nan in database with 0
    metabolites_voxel_table_only_data.replace(np.nan, 0, inplace=True)
    
    # round data to 4 decimal places
    metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.round(decimals=4)
    
    # merge previously seperated metabolites and non-metabolites data
    metabolites_voxel_table_only_data=metabolites_voxel_table_only_data.convert_dtypes()
    metabolites_voxel_table_except_data=metabolites_voxel_table_except_data.convert_dtypes()
    metabolites_voxel_table_new = pd.concat([metabolites_voxel_table_except_data,metabolites_voxel_table_only_data],axis=1)

    # save data as csv
    print ("Saving Datafile") 
    try:
        # Create target Directory
        os.makedirs("saved_files/single_patient_voxel_data/orig_and_ratio/")
        print("Directory " , "saved_files/single_patient_voxel_data/orig_and_ratio/" ,  " created ") 
    except FileExistsError:
        print("Directory " , "saved_files/single_patient_voxel_data/orig_and_ratio/" ,  " already exists")
        
    voxel_table_file_name="saved_files/single_patient_voxel_data/orig_and_ratio/"+file_index+"_metabolite_ratio_voxel_data.csv"
    metabolites_voxel_table_new.to_csv(voxel_table_file_name, header=True, index=True,index_label="Index") 


