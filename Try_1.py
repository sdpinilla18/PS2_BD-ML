# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import pyreadr as pyr
import sklearn as sk
import matplotlib.pyplot as plt
import scipy as sc
import os
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *

os.chdir("C:/Users/hp/OneDrive - Universidad de los Andes/Documentos/Docs/Universidad/2022-2/Big Data/Taller 2/Repo/PS2_BD-ML") #Working directory. 

####Train #########
tr_p=pyr.read_r("train_personas.Rds") 
tr_h=pyr.read_r("train_hogares.Rds")
print(tr_p.keys())
print(tr_h.keys())
df_trp=tr_p[None] #train Data frame (individuals). 
df_trh=tr_h[None] #Train Data frame (households). 

df=pd.merge(df_trp, df_trh, on="id") #Train master data frame (merge by unique identificator key). 

df.columns

#Test ############3
te_p=pyr.read_r("test_personas.Rds") 
te_h=pyr.read_r("test_hogares.Rds")
print(te_p.keys())
print(te_h.keys())
df_tep=te_p[None] #test Data frame (individuals). 
df_teh=te_h[None] #test Data frame (households). 

df_test=pd.merge(df_tep, df_teh, on="id") #Train master data frame (merge by unique identificator key). 

#Missing values count/share in train.
df.isnull().sum() 
df.isnull().sum()/len(df) 

#Train ######################################
#Convert categorical variables to dummy variables:
estrato1_d=pd.get_dummies(df["Estrato1"], prefix="estrato") 
maxeduc_d=pd.get_dummies(df["P6210"], prefix="educ") 

#Merge dummy's variables data frame with master data frame:
df=pd.merge(df, estrato1_d, left_index=True, right_index=True)
df=pd.merge(df, maxeduc_d, left_index=True, right_index=True)

#Test ######################33
maxeduc_d=pd.get_dummies(df_test["P6210"], prefix="educ") 

#Merge dummy's variables data frame with master data frame:
df_test=pd.merge(df_test, estrato1_d, left_index=True, right_index=True)
df_test=pd.merge(df_test, maxeduc_d, left_index=True, right_index=True)

#Recode variables train ###########
df["Ingtotugarrp"]=df["Ingtotugarr"]/df["Nper"] #Mean of household income. 
df["P6020"]=np.where(df["P6020"]==1, 0, 1) #Recode sex, woman=1 
df["P6585s3"]=np.where(((df["P6585s3"]==9) & (df["P6585s3"]==2)), 0, 1) #Recode sex, woman=1 

#Test ##################
df_test["P6020"]=np.where(df_test["P6020"]==1, 0, 1) #Recode sex, woman=1 
df_test["P6585s3"]=np.where(((df_test["P6585s3"]==9) & (df_test["P6585s3"]==2)), 0, 1) #Recode sex, woman=1 

#Train (Household) #####################
df_trh["Ingtotugarr_m"]=df_trh["Ingtotugarr"]/df_trh["Nper"]

#Generate descriptive statistics of train dataset
ds1=(df[["Ingtot", "P6040", "Pobre", "P6020", "estrato_1.0", "estrato_2.0", "estrato_3.0", 
"estrato_4.0", "estrato_5.0", "estrato_6.0", "educ_1.0", "educ_2.0", "educ_3.0", "educ_4.0", "educ_5.0", "educ_6.0", "P6585s3", "Oc"]].describe(include="all"))
ds1=ds1.T
ds1=ds1[["count", "mean", "std", "min", "50%", "max"]]
ds1=ds1.round(2)
ds2=(df_trh[["Ingtotugarr_m","Nper"]].describe(include="all"))
ds2=ds2.T
ds2=ds2[["count", "mean", "std", "min", "50%", "max"]]
ds2=ds2.round(2)
ds=pd.concat([ds2, ds1])
print(ds.to_latex())

#Histogram  of household per_capita income:
df_trh["Ingtotugarr_mM"]=df_trh["Ingtotugarr"]/df_trh["Nper"]/1000000
plt.hist(df_trh["Ingtotugarr_mM"], bins=1000, color = (0.17, 0.44, 0.69, 0.9))
plt.xlim(0,6)
#plt.ylim(0,10000)
plt.xticks([i for i in range(7)])
plt.ylabel("N° hogares")
plt.xlabel("COP millones (pesos colombianos)")
plt.savefig("histy_ipch.jpg", bbox_inches="tight")
plt.show()

df_trh["Ingtotugarr_mM"].describe()








#Revisar 
#df["Ingtotugarr_m"]=df["Ingtotugarr"]/1000000
#Histogram of total household income:
#plt.hist(df["Ingtotugarr_m"], bins=450, color = (0.17, 0.44, 0.69, 0.9))
#plt.xlim(0,12)
#plt.ylim(0,10000)
#plt.xticks([i for i in range(11)])
#plt.ylabel("Personas")
#plt.xlabel("COP millones (pesos)")
#plt.savefig("histy.jpg", bbox_inches="tight")
#plt.show()

#df["Ingtotugarrp_m"]=df["Ingtotugarrp"]/1000000
#Histogram of average household income:
#plt.hist(df["Ingtotugarrp_m"], bins=450, color = (0.17, 0.44, 0.69, 0.9))
#plt.xlim(0,5)
#plt.ylim(0,10000)
#plt.xticks([i for i in range(5)])
#plt.ylabel("Personas")
#plt.xlabel("COP millones (pesos)")
#plt.savefig("histy_avg.jpg", bbox_inches="tight")
#plt.show()


###Split train and test using training database. 
#Train sub test database using PSM to reproduce test 

#df_test["test"]=1
#df["test"]=0
#c=list(df_test.columns)
#df_2=df[c]
#df_tt=df_2.append(df_test, ignore_index=True)
#df_tt=df_tt.reset_index()
#df_tt["index"]=pd.to_numeric(df_tt["index"])
#df_tt["test"]=pd.to_numeric(df_tt["test"])











