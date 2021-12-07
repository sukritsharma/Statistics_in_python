#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 05:17:52 2021

@author: local
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import the model we are using




voxel_data = pd.read_csv('saved_files/All_patient_data/All_patient_metabolite_ratio_only_voxel_data.csv',low_memory=False)
voxel_data=voxel_data.drop(voxel_data[(voxel_data['Segmentation']==2) & (voxel_data['Grade']==2)].index)
# threshold_data=pd.read_csv('Threshold_new.csv',low_memory=False)

voxel_data_copy=voxel_data.copy(deep=True)

metabolites=voxel_data_copy.columns
number_of_columns=len(metabolites)
metabolites=metabolites[list(range(28,number_of_columns-1))]

metabolites_columns=voxel_data_copy[metabolites]

plt.figure(figsize=(120,120), dpi=150)
plt.rcParams.update({'font.size': 50})
correlation_matrix=metabolites_columns.corr()
sns.set(font_scale=8)
correlation_matrix=sns.heatmap(correlation_matrix ,annot=True, linewidths=0.8, linecolor='black',annot_kws={"size": 7},cmap= 'Oranges',square=1)
output_name='saved_files/' +'correlation_matrix.png'
correlation_matrix.figure.savefig(output_name,bbox_inches='tight')
