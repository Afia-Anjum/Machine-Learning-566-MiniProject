# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:35:14 2019

@author: Asus
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import warnings
warnings.filterwarnings('ignore')



def main():

    df = pd.read_csv("D:\ALBERTA\Fall 2019\Intro to ML\Mini Project\kag_risk_factors_cervical_cancer.csv")
    #print(df)
    df_nan = df.replace("?", np.nan)
    df1 = df_nan.convert_objects(convert_numeric=True)
    
    print(df1.isnull().sum())  #number of NAN's in each of the features
    print(df1)                 #preprocessed dataframe
    
    #filling the NAN values with the median values of all the samples for a particular feature
    
    #the features having more than 100 NAN values are not filled with median values
    #beacuase it would affect our learning
    
    df1['First sexual intercourse'].fillna(df1['First sexual intercourse'].median(), inplace = True)
    df1['Num of pregnancies'].fillna(df1['Num of pregnancies'].median(), inplace = True)
    df1['First sexual intercourse'].fillna(df1['First sexual intercourse'].median(), inplace = True)
    df1['Smokes'].fillna(0,inplace = True)
    df1['Number of sexual partners'].fillna(df1['Number of sexual partners'].median(), inplace = True)
    l = (df1['Smokes']==1)
    df1.loc[l,'Smokes (years)'] = df1.loc[l,'Smokes (years)'].fillna(df1.loc[l,'Smokes (years)'].median())
    l = (df1['Smokes']==0)
    df1.loc[l,'Smokes (years)'] = df1.loc[l,'Smokes (years)'].fillna(0)
    l = (df1['Smokes']==1)
    df1.loc[l,'Smokes (packs/year)'] = df1.loc[l,'Smokes (packs/year)'].fillna(df1.loc[l,'Smokes (packs/year)'].median())
    l = (df1['Smokes']==0)
    df1.loc[l,'Smokes (packs/year)'] = df1.loc[l,'Smokes (packs/year)'].fillna(0)
    df2 = df1.drop(['Hinselmann','Schiller','Citology','Biopsy'], axis = 1)
    #print(df2)
    
    corrmat = df2.corr()
    #print(corrmat)
    
    k = 15 #number of variables for heatmap
    cols = corrmat.nlargest(k, 'Hormonal Contraceptives')['Hormonal Contraceptives'].index
    print(cols)
    '''
    output of cols:
        Index(['Hormonal Contraceptives', 'Hormonal Contraceptives (years)',
       'Num of pregnancies', 'STDs: Time since last diagnosis',
       'STDs: Time since first diagnosis', 'Age', 'STDs:HPV', 'Dx:HPV', 'IUD',
       'STDs:genital herpes', 'STDs:pelvic inflammatory disease', 'Dx:Cancer',
       'First sexual intercourse', 'Number of sexual partners',
       'Smokes (packs/year)']
    '''
    #out of the correlated features, these two have very big values(787) for NAN, it will affect our learning, we will drop
    #these two column
    
    cols =cols.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])
    cm = df2[cols].corr()

    plt.figure(figsize=(8,8))
    
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, cmap='Set1' ,annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels = cols.values, xticklabels = cols.values)
    
    plt.show()
    
    # If patient is older than sample mean or number of pregnancies is lower than mean then patient may take Hormonal 
    # Contraceptives
    l = (df2['Age']>df2['Age'].mean())
    df2.loc[l,'Hormonal Contraceptives'] = df2.loc[l,'Hormonal Contraceptives'].fillna(1)
    l = (df2['Num of pregnancies']<df2['Num of pregnancies'].mean())
    df2.loc[l,'Hormonal Contraceptives'] = df2.loc[l,'Hormonal Contraceptives'].fillna(1)
    df2['Hormonal Contraceptives'].fillna(0,inplace = True)

    #print(df2['Hormonal Contraceptives'].isnull().sum())
    
    #For HC(years) NaN values we can fill with median values by using HC feature
    l = (df2['Hormonal Contraceptives'] == 1)
    df2.loc[l,'Hormonal Contraceptives (years)'] = df2.loc[l,'Hormonal Contraceptives (years)'].fillna(df2['Hormonal Contraceptives (years)'].median())
    l = (df2['Hormonal Contraceptives'] == 0)
    df2.loc[l,'Hormonal Contraceptives (years)'] = df2.loc[l,'Hormonal Contraceptives (years)'].fillna(0)
    
    #Also we need to check relationship between HC and HC (years)
    #print(len(df2[(df2['Hormonal Contraceptives'] == 1) & (df2['Hormonal Contraceptives (years)'] == 0) ]))
    
    #print(len(df2[(df2['Hormonal Contraceptives'] == 0) & (df2['Hormonal Contraceptives (years)'] != 0) ]))
    
    #print(df2)
    #print(df2['Hormonal Contraceptives (years)'].isnull().sum())
    
    #Using pearson correlation we can determine which feature is effect 'IUD'.
    corrmat = df2.corr()
    k = 15 #number of variables for heatmap
    cols = corrmat.nlargest(k, 'IUD')['IUD'].index

    cols =cols.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])

    cm = df2[cols].corr()

    plt.figure(figsize=(9,9))

    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm,cmap = 'rainbow', cbar=True, annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels = cols.values, xticklabels = cols.values)
    plt.show()
    
    len(df2[(df2['Age']>df2['Age'].mean())&(df2['IUD']==1)])
    
    len(df2[df2['IUD']==1])
    df2['IUD'].fillna(0, inplace = True)
    
    
    

main()
