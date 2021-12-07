
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 11:10:31 2021

@author: Sukrit Sharma
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


voxel_data = pd.read_csv('saved_files/All_patient_data/All_patient_metabolite_ratio_only_voxel_data.csv',low_memory=False)

#remove PT and NAWM regions
voxel_data.rename(columns={'High_grade': 'Grades'}, inplace=True)

voxel_data=voxel_data.drop(voxel_data[ (voxel_data['Segmentation']>4) | (voxel_data['Segmentation']==0) ].index)

# voxel_data=voxel_data.drop(voxel_data[(voxel_data['Region']=='Peritumoral VOI')].index)
# voxel_data=voxel_data.drop(voxel_data[(voxel_data['Region']=='NAWM')].index)


voxel_data.loc[(voxel_data['IDH'] == '1.0'), ['IDH']] = 'IDH1_Mutation'
voxel_data.loc[(voxel_data['IDH'] == '0.0'), ['IDH']] = 'Wildtype_Mutation'

voxel_data.loc[(voxel_data['Segmentation'] == 1), ['Segmentation']] = 'NCE'

voxel_data.loc[(voxel_data['Segmentation'] == 2), ['Segmentation']] = 'CE'
voxel_data.loc[(voxel_data['Segmentation'] == 4), ['Segmentation']] = 'Nec'

voxel_data.loc[(voxel_data['Grades'] == 1), ['Grades']] = 'HGG'
voxel_data.loc[(voxel_data['Grades'] == 0), ['Grades']] = 'LGG'


threshold_data=pd.read_csv('Threshold_new.csv',low_memory=False)

metabolites=voxel_data.columns
number_of_columns=len(metabolites)
metabolites=metabolites[28:number_of_columns+1]
# metabolites=['GPC+PChToNAA+NAAG','GlnToNAA+NAAG','GlnToCr+PCr','GlyToNAA+NAAG','InsToCr+PCr','GlyToIns','GlnToIns','GlyToCr+PCr','GluToCr+PCr']
# metabolites=['GlnToNAA','GluToNAA','GlyToNAA','GPC+PChToNAA','GlyToIns']
# metabolites=['GlnToNAA','Ins+GlyToNAA', 'GlyToNAA','GPC+PChToNAA']





for metabolite_name in metabolites:
    subplot_rows=3
    subplot_cols=2
    
    upper_threshold=10
    lower_threshold=threshold_data[metabolite_name].min()
    
    
    fig, axs = plt.subplots(nrows=subplot_rows, ncols=subplot_cols,figsize=(30*subplot_cols, 15*subplot_rows), constrained_layout=True,squeeze=False)
    total_plots=[['Segmentation','IDH'],['Grades','IDH'],['Segmentation','Grades'],['Grade','IDH'],['Region','Grades'],['Region','IDH']]

    # total_plots=[['Segmentation','IDH'],['Segmentation','Grade'],['Grades','IDH'],['Segmentation','Grades'],['Grade','IDH'],['Region','Grades'],['Region','IDH'],['Region','Grade']]

    # total_plots=[['Segmentation','IDH']]


    selected_columns=voxel_data[['Pat Code','IDH','Grade','Grades','Segmentation','Region','WM_mask',metabolite_name]]
    index_voxel_over_threshold = selected_columns[selected_columns[metabolite_name] >upper_threshold ].index
    index_voxel_zero = selected_columns[selected_columns[metabolite_name] == 0 ].index
    number_of_voxels=len(selected_columns[metabolite_name])
    count_voxel_over_threshold=len(index_voxel_over_threshold)
    selected_columns=selected_columns.drop(index_voxel_over_threshold)
    selected_columns=selected_columns.drop(index_voxel_zero)
    
     
    # seg_column=pd.get_dummies(voxel_data['Segmentation'])
    # tVOI=seg_column.drop([0,5],axis=1)
    # tVOI=tVOI.sum(axis=1)
    # pVOI=seg_column.drop([0,1,2,4],axis=1)
    # pVOI=pVOI.sum(axis=1)
    # pVOI=2*pVOI
    # wm_column=3*voxel_data['WM_mask']
    
    # regions_column=tVOI+pVOI+wm_column
    
    # regions_column[regions_column==1]='Tumor VOI'
    # regions_column[regions_column==2]='Peritumoral VOI'
    # regions_column[regions_column==3]='NAWM'
    
    # selected_columns['Region']=regions_column
    
    index_seg_zero = selected_columns[selected_columns['Region'] == 0 ].index
    selected_columns=selected_columns.drop(index_seg_zero)
    
    index_voxel_unter_threshold = selected_columns[(selected_columns[metabolite_name] <=lower_threshold) & (selected_columns['Segmentation']!=5)  ].index
    selected_columns=selected_columns.drop(index_voxel_unter_threshold)
    
    selected_columns=selected_columns.sort_values(by=['IDH'])
    
    # max_value=round(selected_columns[metabolite_name].max()+0.5)
    
    # q1=selected_columns[metabolite_name].quantile(0.25)
    # q3=selected_columns[metabolite_name].quantile(0.75)
    # IQR=q3-q1
    # max_value=q3+1.5*IQR
    # max_value=round(max_value+0.5)
    flag_ylim=True
    for i,plot_type in enumerate(total_plots):
        
        row=i//subplot_cols
        col= i % subplot_cols
        seperation=plot_type[0]
        comparison=plot_type[1]
        print(seperation + "seperated" + comparison+ "compared !")
        sns.set(font_scale=4)
        boxplot_met1=sns.boxplot(x=seperation,y=metabolite_name,hue=comparison,data=selected_columns,showfliers = False,ax=axs[row][col]  ) #, palette="Set1" )
        # boxplot_met1.legend_.remove()
        
        seperation_labels=pd.Categorical(voxel_data[seperation]).categories
        comparison_labels=pd.Categorical(voxel_data[comparison]).categories         #get labels of comparison for statistical test
        
        
        box_pairs_stat_test=list()      
        for sep_lab in seperation_labels:
            box_pairs=()
            for com_lab in comparison_labels:
                box_pairs=box_pairs+((sep_lab,com_lab),)
            box_pairs_stat_test=box_pairs_stat_test+[box_pairs]
            
        from statannot import add_stat_annotation
        # box_pairs_stat_test=[("Wildtype_Mutation", "IDH1_Mutation")]
        # # box_pairs_stat_test=[(("Segmentation", "IDH1 Mutation"), ("Segmentation", "Wildtype Mutation"))]
        
        add_stat_annotation(boxplot_met1,x=seperation,y=metabolite_name,hue=comparison,data=selected_columns,box_pairs=box_pairs_stat_test,test='Mann-Whitney',loc='outside',verbose=2,text_format='star')
        
        # boxplot_met1=sns.swarmplot(x='Segmentation',y=metabolite_name,hue='Grade',data=selected_columns,ax=axs[row][col])
        if flag_ylim==True:
            ylim_max=boxplot_met1.get_ylim()[1]
            ylim_max=round(ylim_max+0.5)
            flag_ylim=False
    
        boxplot_met1.set(ylim=(0,ylim_max))
    # caption=('ns: 5.00e-02 < p <= 1.00e+00 \n' 
    #         +'*: 1.00e-02 < p <= 5.00e-02\n' 
    #         +'**: 1.00e-03 < p <= 1.00e-02\n'  
    #         +'***: 1.00e-04 < p <= 1.00e-03\n' 
    #         +'****: p <= 1.00e-04 ')
    # boxplot_met1=boxplot_met1.figure.text(0.05, -0.05, caption, ha='left', size='xx-large')
    try:
        # Create target Directory
        os.makedirs("saved_files/Boxplots/")
        print("Directory " , "saved_files/Boxplots/" ,  " created ") 
    except FileExistsError:
        print("Directory " , "saved_files/Boxplots/" ,  " already exists")
        
    output_name='saved_files/Boxplots/' +metabolite_name + '_Boxplots'
    boxplot_met1.figure.savefig(output_name)