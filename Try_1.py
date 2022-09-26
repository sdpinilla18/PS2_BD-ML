# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Packages:
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
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

os.chdir("C:/Users/hp/OneDrive - Universidad de los Andes/Documentos/Docs/Universidad/2022-2/Big Data/Taller 2/Repo/PS2_BD-ML") #Working directory. 

####Train #########
tr_p=pyr.read_r("train_personas.Rds") 
tr_h=pyr.read_r("train_hogares.Rds")
print(tr_p.keys())
print(tr_h.keys())
df_trp=tr_p[None] #train Data frame (individuals). 
df_trh=tr_h[None] #Train Data frame (households). 

df=pd.merge(df_trp, df_trh, on="id") #Train master data frame (merge by unique identificator key). 
df.rename(columns={"Clase_x": "clase"})
df.rename(columns={"Dominio_x": "Dominio"})

#Test ############3
te_p=pyr.read_r("test_personas.Rds") 
te_h=pyr.read_r("test_hogares.Rds")
print(te_p.keys())
print(te_h.keys())
df_tep=te_p[None] #test Data frame (individuals). 
df_teh=te_h[None] #test Data frame (households). 

df_test=pd.merge(df_tep, df_teh, on="id") #Train master data frame (merge by unique identificator key). 
df_test.rename(columns={"Clase_x": "clase"})
df_test.rename(columns={"Dominio_x": "Dominio"})

#Missing values count/share in train.
df.isnull().sum() 
df.isnull().sum()/len(df) 

df_test.columns

######################################Train #############################################################################

#Convert categorical variables to dummy variables:
estrato1_d=pd.get_dummies(df["Estrato1"], prefix="estrato") 
maxeduc_d=pd.get_dummies(df["P6210"], prefix="educ") 
dominio_d=pd.get_dummies(df["Dominio_x"], prefix="dominio")
departamento_d=pd.get_dummies(df["Depto_x"], prefix="depto")
salud_d=pd.get_dummies(df["P6100"], prefix="salud")
trabajo_d=pd.get_dummies(df["P6430"], prefix="trabajo")
actividad_d=pd.get_dummies(df["P6240"], prefix="act")
numper_d=pd.get_dummies(df["P6870"], prefix="numper")
ocseg_d=pd.get_dummies(df["P7050"], prefix="ocseg")
trabdeso_d=pd.get_dummies(df["P7350"], prefix="trabdeso")
tipovivienda_d=pd.get_dummies(df["P5090"], prefix="tipoviv")
oficio_d=pd.get_dummies(df["Oficio"], prefix="oficio")


#Merge dummy's variables data frame with master data frame:
df=pd.merge(df, estrato1_d, left_index=True, right_index=True)
df=pd.merge(df, maxeduc_d, left_index=True, right_index=True)
df=pd.merge(df, dominio_d, left_index=True, right_index=True)
df=pd.merge(df, departamento_d, left_index=True, right_index=True)
df=pd.merge(df, trabajo_d, left_index=True, right_index=True)
df=pd.merge(df, salud_d, left_index=True, right_index=True)
df=pd.merge(df, actividad_d, left_index=True, right_index=True)
df=pd.merge(df, numper_d, left_index=True, right_index=True)
df=pd.merge(df, ocseg_d, left_index=True, right_index=True)
df=pd.merge(df, trabdeso_d, left_index=True, right_index=True)
df=pd.merge(df, tipovivienda_d, left_index=True, right_index=True)
df=pd.merge(df, oficio_d, left_index=True, right_index=True)

##################Test ###############################################################################################

maxeduc_d=pd.get_dummies(df_test["P6210"], prefix="educ") 
dominio_d=pd.get_dummies(df_test["Dominio_x"], prefix="dominio")
departamento_d=pd.get_dummies(df_test["Depto_x"], prefix="depto")
salud_d=pd.get_dummies(df_test["P6100"], prefix="salud")
trabajo_d=pd.get_dummies(df_test["P6430"], prefix="trabajo")
actividad_d=pd.get_dummies(df_test["P6240"], prefix="act")
numper_d=pd.get_dummies(df_test["P6870"], prefix="numper")
ocseg_d=pd.get_dummies(df_test["P7050"], prefix="ocseg")
trabdeso_d=pd.get_dummies(df_test["P7350"], prefix="trabdeso")
tipovivienda_d=pd.get_dummies(df_test["P5090"], prefix="tipoviv")
oficio_d=pd.get_dummies(df_test["Oficio"], prefix="oficio")

#Merge dummy's variables data frame with master data frame:
df_test=pd.merge(df_test, maxeduc_d, left_index=True, right_index=True)
df_test=pd.merge(df_test, dominio_d, left_index=True, right_index=True)
df_test=pd.merge(df_test, departamento_d, left_index=True, right_index=True)
df_test=pd.merge(df_test, salud_d, left_index=True, right_index=True)
df_test=pd.merge(df_test, trabajo_d, left_index=True, right_index=True)
df_test=pd.merge(df_test, actividad_d, left_index=True, right_index=True)
df_test=pd.merge(df_test, numper_d, left_index=True, right_index=True)
df_test=pd.merge(df_test, ocseg_d, left_index=True, right_index=True)
df_test=pd.merge(df_test, trabdeso_d, left_index=True, right_index=True)
df_test=pd.merge(df_test, tipovivienda_d, left_index=True, right_index=True)
df_test=pd.merge(df_test, oficio_d, left_index=True, right_index=True)

#Recode variables of train ###########
var=(["P6020", "P6090", "P6510", "P6545", "P6580", "P6585s1", "P6585s2", "P6585s3", "P6585s4", "P6590", "P6600", "P6610", "P6620"
 ,"P6630s1", "P6630s2", "P6630s3", "P6630s4" ,"P6630s6" , "P6920","P7040","P7090","P7110","P7120","P7150", "P7160", "P7310",
 "P7422", "P7472", "P7495", "P7500s2", "P7500s3","P7505", "P7510s1", "P7510s2","P7510s3", "P7510s5", "P7510s6", "P7510s7"])

ceros=["Oc", "Des", "Ina"]


for i in var:
    df[i]=np.where(df[i]==9, np.nan, df[i])
    df[i]=np.where(df[i]==1, 1, 0*df[i])

##Replace Na=0 in Ocupados, Desocupados and Inactivos
for i in ceros:
    df[i]=np.where(df[i]==1, 1, 0)

#Recode variables of test ###########
for i in var:
    df_test[i]=np.where(df_test[i]==9, np.nan, df_test[i])
    df_test[i]=np.where(df_test[i]==1, 1, 0*df_test[i])

for i in ceros:
    df_test[i]=np.where(df_test[i]==1, 1, 0)

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
plt.xlabel("Millones COP")
plt.savefig("histy_ipch.jpg", bbox_inches="tight")
plt.show()

df_trh["Ingtotugarr_mM"].describe()

###Split train and test using training database. 
#Train sub test database using PSM to reproduce test 

elim=(["id", "Orden", "Fex_c_y", "Clase_y", "Dominio_y", "Depto_y", "Dominio_x", "Depto_x",  "Fex_dpto_y", "Fex_dpto_x", "Fex_c_x" , "Depto_x", 
"P6050" , "P6210s1", "P6210s1" , "educ_1.0" , "dominio_ARMENIA" , "depto_05", "salud_1.0", "salud_9.0" , "trabajo_9.0", "act_1.0", 
"numper_1.0", "ocseg_1.0", "trabdeso_1.0", "tipoviv_1.0", "oficio_1.0", "P6100", "P6210", "P6240" , "Oficio", "P6430", "P6870", "P7050", 
"P7350" , "P5090"])

df["test"]=0
df_test["test"]=1
c=[i for i in df_test.columns if i not in elim]


####Dealing with nans
imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
x1=df_test[c]
x1=imp_mean.fit_transform(x1)

x2=df[c]
x2=imp_mean.fit_transform(x2)

#Create a X matrix of covariates
x11=pd.DataFrame(x1, columns=c)
x22=pd.DataFrame(x2, columns=c)
X=x22.append(x11, ignore_index=True)

#Create matrix Y
Y=X["test"]
Y=Y.astype('int')

## Generate PSM of test
lr = LogisticRegression(random_state=911, class_weight="balanced")
result=lr.fit(X,Y)
PSM=result.predict_proba(X)

PSM=pd.DataFrame(PSM, columns=["no", "si"])
PSM
X=pd.merge(X,PSM, left_index=True, right_index=True) 

grouped = X.groupby(X.test)
X_train = grouped.get_group(0)
X_test = grouped.get_group(1)

plt.hist([X_test.si,X_train.si], bins=10, label=['test', 'train'], color=["r", "b"] )
plt.legend(loc='upper right')
plt.xticks([0,0.2,0.4,0.6,0.8,1])
plt.ylabel("Frecuencia")
plt.xlabel("Probabilidad de pertenecer a test")
plt.savefig("histy_psm.jpg", bbox_inches="tight")
plt.show()

len(X_train[X_train.si>=0.5])/len(X_train["Clase_x"])























