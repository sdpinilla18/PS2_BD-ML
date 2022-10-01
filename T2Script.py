#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Problem-Set #2, BD&MLfAE
# David Santiago Carballo Candela, 201813007
# Sergio David Pinilla Padilla, 201814755
# Juan Diego Valencia Romero, 201815561


# In[7]:


#Packages:
import pandas as pd
import numpy as np
import pyreadr as pyr
import sklearn as sk
import matplotlib.pyplot as plt
import scipy as sc
#import imblearn as ib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from collections import Counter
#from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler
from psmpy.plotting import *
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
import scipy as sc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import d2_pinball_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[18]:


#Set directory
os.chdir("path")

####Train ################################################
tr_p=pyr.read_r("train_personas.Rds") 
tr_h=pyr.read_r("train_hogares.Rds")
print(tr_p.keys())
print(tr_h.keys())
df_trp=tr_p[None] #train Data frame (individuals). 
df_trh=tr_h[None] #Train Data frame (households).


# In[ ]:


df=pd.merge(df_trp, df_trh, on="id") #Train master data frame (merge by unique identificator key). 
df.rename(columns={"Clase_x": "clase"})
df.rename(columns={"Dominio_x": "Dominio"})


# In[ ]:


#Test ############################################################################################
te_p=pyr.read_r("test_personas.Rds") 
te_h=pyr.read_r("test_hogares.Rds")
print(te_p.keys())
print(te_h.keys())
df_tep=te_p[None] #test Data frame (individuals). 
df_teh=te_h[None] #test Data frame (households). 


# In[ ]:


df_test=pd.merge(df_tep, df_teh, on="id") #Train master data frame (merge by unique identificator key). 
df_test.rename(columns={"Clase_x": "clase"})
df_test.rename(columns={"Dominio_x": "Dominio"})


# In[ ]:


#Missing values count/share in train.
df.isnull().sum() 
df.isnull().sum()/len(df) 


# In[ ]:


df_test.columns


# In[ ]:


c0=[i for i in df_test.columns]


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


#Generate descriptive statistics of train dataset
ds=(df[["Ingtotugarr", "Ingtot", "P6040", "Nper", "Pobre", "P6020", "estrato_1.0", "estrato_2.0", "estrato_3.0", 
"estrato_4.0", "estrato_5.0", "estrato_6.0", "educ_1.0", "educ_2.0", "educ_3.0", "educ_4.0", "educ_5.0", "educ_6.0", "P6585s3", "Oc"]].describe(include="all"))
ds=ds.T
ds=ds[["count", "mean", "std", "min", "50%", "max"]]
ds=ds.round(2)
print(ds.to_latex())


# In[ ]:


#Histogram of ingtotug
#Histogram of household per_capita income:
df_trh["Ingtotugarr_mM"]=df_trh["Ingtotugarr"]/df_trh["Nper"]/1000000
plt.hist(df_trh["Ingtotugarr_mM"], bins=1000, color = (0.17, 0.44, 0.69, 0.9))
plt.xlim(0,6)
#plt.ylim(0,10000)
plt.xticks([i for i in range(7)])
plt.ylabel("Número de hogares")
plt.xlabel("COP (millones)")
plt.savefig("histy_ipch.jpg", bbox_inches="tight")
plt.show()


# In[ ]:


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

imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

x1c=df_test[c0]
x1c=imp_mean.fit_transform(x1c)


x2c=df[c0]
x2c=imp_mean.fit_transform(x2c)


# In[ ]:


x111c=pd.DataFrame(x2c, columns=c0)
x111c=x111c[x111c["P6050"]==1]


# In[ ]:


#Create a X matrix of covariates
x11=pd.DataFrame(x1, columns=c)
x22=pd.DataFrame(x2, columns=c)
X=x22.append(x11, ignore_index=True)

#Create matrix Y
Y=X["test"]
Y=Y.astype('int')


# In[ ]:


## Generate PSM of test
lr = LogisticRegression(random_state=911, class_weight="balanced")
result=lr.fit(X,Y)
PSM=result.predict_proba(X)


# In[ ]:


PSM=pd.DataFrame(PSM, columns=["no", "si"])
PSM
X=pd.merge(X,PSM, left_index=True, right_index=True) 
X


# In[ ]:


grouped = X.groupby(X.test)
X_train = grouped.get_group(0)
X_test = grouped.get_group(1)


# In[ ]:


plt.hist([X_test.si,X_train.si], bins=10, label=["test", "train"], color=["r", "b"] )
plt.legend(loc="upper right")
plt.xticks([0,0.2,0.4,0.6,0.8,1])
plt.ylabel("Frecuencia")
plt.xlabel("Probabilidad")
plt.savefig("histy_psm.jpg", bbox_inches="tight")
plt.show()


# In[ ]:


x11c=pd.DataFrame(x1c, columns=c0)
x22c=pd.DataFrame(x2c, columns=c0)
x11c["test"]=1
x22c["test"]=0

Xc=x22c.append(x11c, ignore_index=True)

Xc=pd.merge(Xc, X.si, left_index=True, right_index=True)

grouped = Xc.groupby(Xc.test)
X_trainc = grouped.get_group(0)
X_testc = grouped.get_group(1)


# In[ ]:


###Dummy 1 if propensity score >0.5
X_trainc["PSM"]=np.where(X_trainc["si"]>=0.5, 1,0)

###Keep only jefe hogar
X_traincol = X_trainc[X_trainc["P6050"]==1]

X_testcol = X_testc[X_testc["P6050"]==1]

########Split X_trainc in X_train2 and X_ttest

psm=pd.DataFrame(X_trainc["PSM"].groupby(X_trainc["id"]).mean()).reset_index()
psm=psm.rename(columns={"PSM": "PSMc"})
X_traincol=pd.merge(X_traincol,psm)


# In[ ]:


############################## Collapse data base at household level #############################
####Keep only parentesco=jefe hogar

###########collapse train ##################

psm=X_trainc["P6040"].groupby(X_trainc["id"]).apply(np.mean).reset_index()
psm=psm.rename(columns={"P6040": "promedad"})
X_traincol=pd.merge(X_traincol,psm)

listg=["P6800", "P7045","Pet", "Oc"]

for i in listg:
    psm=pd.DataFrame(X_trainc[i].groupby(X_trainc["id"]).sum()).reset_index()
    psm=psm.rename(columns={i: i+"col"})
    X_traincol=pd.merge(X_traincol,psm)


#############collapse test####################################3
####Keep only parentesco=jefe hogar

###########collapse test ##################
psm=X_testc["P6040"].groupby(X_testc["id"]).apply(np.mean).reset_index()
psm=psm.rename(columns={"P6040": "promedad"})
X_testcol=pd.merge(X_testcol,psm)

listg=["P6800", "P7045","Pet", "Oc"]

for i in listg:
    psm=pd.DataFrame(X_testc[i].groupby(X_testc["id"]).sum()).reset_index()
    psm=psm.rename(columns={i: i+"col"})
    X_testcol=pd.merge(X_testcol,psm)


# In[ ]:


########Merge dependent variables in X########
df_ycol=df[df["P6050"]==1]
X_traincol=pd.merge(X_traincol,df_ycol[["Ingtotugarr", "Pobre", "id"]]) 


# In[ ]:


####Recode variables #################################
##################Train ######################################################
X_traincol["saluds"]=np.where(X_traincol["P6100"]==3, 1,0)
X_traincol["univ"]=np.where(X_traincol["P6210"]==6, 1,0)
X_traincol["bajeduc"]=np.where( (X_traincol["P6210"]==1) | (X_traincol["P6210"]==2) | (X_traincol["P6210"]==3) , 1,0)
X_traincol["traba"]=np.where(X_traincol["P6240"]==1, 1,0)
X_traincol["busctrab"]=np.where(X_traincol["P6240"]==2, 1,0)
X_traincol["microemp"]=np.where( (X_traincol["P6870"]==8) | (X_traincol["P6870"]==9) ,0,1)

###Hacinamiento
X_traincol["personaxhab"]=X_traincol["Nper"]/X_traincol["P5010"]
X_traincol["hacinamiento"]=np.where(X_traincol["personaxhab"]>3, 1,0)

X_traincol["vivpropia"]=np.where( (X_traincol["P5090"]==1) | (X_traincol["P5090"]==1) ,1,0)


###################################Test #################################
X_testcol["saluds"]=np.where(X_testcol["P6100"]==3, 1,0)
X_testcol["univ"]=np.where(X_testcol["P6210"]==6, 1,0)
X_testcol["bajeduc"]=np.where( (X_testcol["P6210"]==1) | (X_testcol["P6210"]==2) | (X_testcol["P6210"]==3) , 1,0)
X_testcol["traba"]=np.where(X_testcol["P6240"]==1, 1,0)
X_testcol["busctrab"]=np.where(X_testcol["P6240"]==2, 1,0)
X_testcol["microemp"]=np.where( (X_testcol["P6870"]==8) | (X_testcol["P6870"]==9) ,0,1)

###Hacinamiento
X_testcol["personaxhab"]=X_testcol["Nper"]/X_testcol["P5010"]
X_testcol["hacinamiento"]=np.where(X_testcol["personaxhab"]>3, 1,0)

X_testcol["vivpropia"]=np.where( (X_testcol["P5090"]==1) | (X_testcol["P5090"]==1) ,1,0)


# In[ ]:


X_traincol["arriendo"]=X_traincol["P5130"]+X_traincol["P5140"]

X_traincol["horastotal"]=X_traincol["P6800"]+X_traincol["P7045"]


# In[ ]:


#####Get dummies dominio
dominio_d1=pd.get_dummies(X_traincol["Dominio_x"], prefix="dominio")
X_traincol=pd.merge(X_traincol, dominio_d, left_index=True, right_index=True)

dominio_d2=pd.get_dummies(X_testcol["Dominio_x"], prefix="dominio")
X_testcol=pd.merge(X_testcol, dominio_d, left_index=True, right_index=True)


# In[ ]:


###Select a subset of variables 

variables1=([ 'Clase_x', 'P6020', 'P6040', 'P6090', 'P6426', 'P6545', 'P6590', 'P6610', 'P6920',  'P7422', 'P7495', 'P7505', 'P5100' ,
'promedad', 'P6800col', 'P7045col', 'Petcol', 'Occol',
       'saluds', 'univ', 'bajeduc', 'traba', 'busctrab', 'microemp',
       'personaxhab', 'hacinamiento', 'vivpropia', 'PSMc',"Ingtotugarr", "Pobre", "Lp", "id", "Li" , "Npersug", "arriendo", "horastotal"])


variables2=([ 'Clase_x', 'P6020', 'P6040', 'P6090', 'P6426', 'P6545', 'P6590', 'P6610', 'P6920',  'P7422', 'P7495', 'P7505', 'P5100' ,
'promedad', 'P6800col', 'P7045col', 'Petcol', 'Occol',
       'saluds', 'univ', 'bajeduc', 'traba', 'busctrab', 'microemp',
       'personaxhab', 'hacinamiento', 'vivpropia', 'id', "Npersug"])


dominio_d12=[i for i in list(dominio_d1.columns) if i!="dominio_ARMENIA" and i!="dominio_BOGOTA"]
dominio_d22=[i for i in list(dominio_d2.columns) if i!="dominio_ARMENIA" and i!="dominio_BOGOTA"]


variables1=variables1+dominio_d12
variables2=variables2+dominio_d22


X_traincol=X_traincol[variables1]
X_testcol=X_testcol[variables2]


# In[ ]:


#Split X train based on psm score

X_ttest=X_traincol[X_traincol["PSMc"]>0.5]
X_train2=X_traincol[X_traincol["PSMc"]<=0.5]

dependientes=["Ingtotugarr", "Pobre", "Lp", "id", "Li", "Npersug", "PSMc"]
var1=[i for i in X_ttest.columns if i not in dependientes]
var2=[i for i in X_train2.columns if i not in dependientes]
var3=[i for i in X_testcol.columns if i not in dependientes]

###Create dependent variables
y_ttest=X_ttest[dependientes]
y_train2=X_train2[dependientes]

##Create independent variables
X_ttest=X_ttest[var1]
X_train2=X_train2[var2]
X_testcol=X_testcol[var3]


# In[ ]:


len(y_train2[y_train2.Pobre==1])/len(y_train2.Pobre)


# In[ ]:


len(y_ttest[y_ttest.Pobre==1])/len(y_ttest.Pobre)


# In[ ]:


##########################Regresion Models ######################################
###Standardize continue variables and winsorize dependent variable

sc.stats.mstats.winsorize(y_train2["Ingtotugarr"], limits=[0,0.05], inplace=True)

y_train2["Ingtotugarrest"]=(y_train2["Ingtotugarr"] - y_train2["Ingtotugarr"].mean()) / y_train2["Ingtotugarr"].std()

cont=["P6040", "P6426", "P5100", "arriendo", "horastotal"]


for x in X_train2:
    if x in cont:
        X_train2[x]=(X_train2[x] - X_train2[x].mean()) / X_train2[x].std()

for x in X_ttest:
    if x in cont:
        X_ttest[x]=( X_ttest[x] - X_ttest[x].mean() ) / X_ttest[x].std()

for x in X_testcol:
    if x in cont:
        X_testcol[x]=( X_testcol[x] - X_testcol[x].mean() ) / X_testcol[x].std()


# In[ ]:


###########Resampling X_train y_train ###################
###Resampling for pobre
sm = SMOTE(random_state=911)
X_train2resp, y_train2resp = sm.fit_resample(X_train2, y_train2[["Pobre"]])


# In[ ]:


y_train2["Ingtotugarr"].max()


# In[ ]:


np.shape(X_train2)


# In[ ]:


##Lasso Regresion
lasso = LassoCV(cv=10, random_state=911, n_jobs=-1, fit_intercept=False,n_alphas=100,selection="random", max_iter=1000)
y_train2["Ingtotugarr"]=y_train2["Ingtotugarr"].astype('int')
resultslasso=lasso.fit(X_train2,y_train2["Ingtotugarrest"])


# In[ ]:


vars=resultslasso.feature_names_in_
print(resultslasso.alpha_)
print(resultslasso.n_features_in_)


# In[ ]:


##Ridge regression
a=np.linspace(0.00001,100,num=100)
ridge2 = RidgeCV( alphas=a,cv=10, fit_intercept=False)
#n_jobs=-1, random_state=911
resultsridge2=ridge2.fit(X_train2,y_train2["Ingtotugarrest"])


# In[ ]:


resultsridge2.alpha_


# In[ ]:


#Elastic Net regression
l1=np.linspace(0.0001,1,num=100)
Elastic2 = ElasticNetCV(cv=10, random_state=911, n_jobs=-1, fit_intercept=False,n_alphas=100,selection="random", max_iter=1000, l1_ratio=l1, precompute=True)
resultselastic2=Elastic2.fit(X_train2,y_train2["Ingtotugarrest"])


# In[ ]:


print(resultselastic2.alpha_)
print(resultselastic2.l1_ratio_)
print(resultselastic2.n_features_in_)


# In[ ]:


#Lasso regression with pinball penalty
a3=np.linspace(0.000000001,100,num=100)
d2_pinball=make_scorer(d2_pinball_score, alpha=0.25)

parameters = {'alpha': a3}
Lqr25= ElasticNet(l1_ratio=1, fit_intercept=False, max_iter=100, random_state=911, selection="random", precompute=True)

Lqr25cv = GridSearchCV(Lqr25, parameters, scoring=d2_pinball, n_jobs=-1)

resultslqr25=Lqr25cv.fit(X_train2,y_train2["Ingtotugarrest"])


# In[ ]:


print(resultslqr25.best_params_)
print(resultslqr25.n_features_in_)


# In[ ]:


##Ridge regression with pinball penalty
a=np.linspace(0.000000001,10,num=100)
d2_pinball=make_scorer(d2_pinball_score, alpha=0.25)

ridge = RidgeCV( alphas=a,cv=10, fit_intercept=False, scoring=d2_pinball)
#n_jobs=-1, random_state=911
resultsridge=ridge.fit(X_train2,y_train2["Ingtotugarrest"])


# In[ ]:


resultsridge.alpha_


# In[ ]:


#Elastic Net regression with pinball penalty
l1=np.linspace(0,1,num=10)
a2=np.linspace(0,100,num=100)
d2_pinball=make_scorer(d2_pinball_score, alpha=0.25)

parameters = {'alpha': a2, "l1_ratio": l1}
Elastic= ElasticNet(fit_intercept=False, max_iter=100, random_state=911, selection="random", precompute=True)
Elasticcv = GridSearchCV(Elastic, parameters, scoring=d2_pinball, n_jobs=-1)
resultselastic=Elasticcv.fit(X_train2,y_train2["Ingtotugarrest"])


# In[ ]:


print(resultselastic.best_params_)
print(resultselastic.n_features_in_)


# In[ ]:


# Quantile Regression (sklearn)
#a=1, b=3, d=1/4
qr25 = QuantileRegressor(quantile=0.25,alpha=0,solver="highs")
resultsqr25=qr25.fit(X_train2,y_train2["Ingtotugarrest"])


# In[ ]:


############### Evaluate regression models ##########################
def dummyscore(y_true,y_pred):
    y_pred1=y_pred*y_ttest["Ingtotugarr"].std()+y_ttest["Ingtotugarr"].mean()
    y_pred2=np.where(y_pred1<y_ttest["Lp"]*y_ttest["Npersug"], 1,0)
    CM = confusion_matrix(y_true, y_pred2)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    # False negative rate
    FNR = FN/(TP+FN)
    #False positive rate
    FPR = FP/(FP+TN)
    WMIR=FNR*0.75+0.25*FPR
    return y_pred2, CM, FNR, FPR, WMIR

liss=[resultslasso, resultsridge2, resultselastic2, resultslqr25, resultsridge, resultselastic, resultsqr25]

X_ttest_reg = np.array(X_ttest)
Y_ttest_reg = np.array(y_ttest["Pobre"])


# In[ ]:


YPrresultslasso,CMresultslasso,FNR1,FPR1,WMIRresultslasso=dummyscore(Y_ttest_reg,resultslasso.predict(X_ttest_reg))
YPrresultsridge2,CMresultsridge2,FNR2,FPR2,WMIRresultsridge2=dummyscore(Y_ttest_reg,resultsridge2.predict(X_ttest_reg))
YPrresultselastic2,CMresultselastic2,FNR3,FPR3,WMIRresultselastic2=dummyscore(Y_ttest_reg,resultselastic2.predict(X_ttest_reg))
YPrresultslqr25,CMresultslqr25,FNR4,FPR4,WMIRresultslqr25=dummyscore(Y_ttest_reg,resultslqr25.predict(X_ttest_reg))
YPrresultsridge,CMresultsridge,FNR5,FPR5,WMIRresultsridge=dummyscore(Y_ttest_reg,resultsridge.predict(X_ttest_reg))
YPrresultselastic,CMresultselastic,FNR6,FPR6,WMIRresultselastic=dummyscore(Y_ttest_reg,resultselastic.predict(X_ttest_reg))
YPrresultsqr25,CMresultsqr25,FNR7,FPR7,WMIRresultsqr25=dummyscore(Y_ttest_reg,resultsqr25.predict(X_ttest_reg))


# In[ ]:


print(WMIRresultslasso,WMIRresultsridge2,WMIRresultselastic2,WMIRresultslqr25,WMIRresultsridge,WMIRresultselastic,WMIRresultsqr25)


# In[ ]:


print(FNR1,FNR2,FNR3,FNR4,FNR5,FNR6,FNR7)


# In[ ]:


print(FPR1,FPR2,FPR3,FPR4,FPR5,FPR6,FPR7)


# In[ ]:


############### Reduced sample regs ################

xyr=pd.merge(X_train2,y_train2["Ingtotugarrest"], left_index=True, right_index=True)
corrxyr=xyr.corr()
ingcmat=corrxyr["Ingtotugarrest"]
ingcmat
corrxyr["Ingtotugarrest"].to_latex(index=True)
xredin=["saluds" ,"univ", "bajeduc" ,"microemp","vivpropia", "hacinamiento"]


# In[ ]:


X_train3=X_train2[xredin]
X_train3


# In[ ]:


#Ridge regression (small)
a=np.linspace(0.00001,100,num=100)
RidgeSmall = RidgeCV( alphas=a,cv=10, fit_intercept=False)
#n_jobs=-1, random_state=911
resultsridgesmall=RidgeSmall.fit(X_train3,y_train2["Ingtotugarrest"])


# In[ ]:


print(resultsridgesmall.alpha_)


# In[ ]:


# Quantile Regression (sklearn)
#a=1, b=3, d=1/4
qr25 = QuantileRegressor(quantile=0.25,alpha=0,solver="highs")
resultsqr25small=qr25.fit(X_train3,y_train2["Ingtotugarrest"])


# In[ ]:


X_ttestin=X_ttest[xredin]
X_ttestin_reg = np.array(X_ttestin)


# In[ ]:


YPrresultsrs,CMresultsrs,FNR8,FPR8,WMIRresultsrs=dummyscore(Y_ttest_reg,resultsridgesmall.predict(X_ttestin_reg))
YPrresultsqs,CMresultsqs,FNR9,FPR9,WMIRresultsqs=dummyscore(Y_ttest_reg,resultsqr25small.predict(X_ttestin_reg))


# In[ ]:


print(WMIRresultsrs,WMIRresultsqs)
print(FNR8,FNR9)
print(FPR8,FPR9)


# In[ ]:


######################################## Clasification Models ####################################################################
##Define penalty #######
def dummyscore2(y,ypred):
    CM = confusion_matrix(y, ypred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    # False negative rate
    FNR = FN/(TP+FN)
    #False positive rate
    FPR = FP/(FP+TN)
    penalty=FNR*0.75+0.25*FPR
    return penalty, CM,FNR,FPR


# In[ ]:


### Quadratic discriminant function
r1=np.linspace(0.0001,1,num=100)
parameters={'reg_param': r1}
qda=QuadraticDiscriminantAnalysis()
qdacv= GridSearchCV(qda, parameters, scoring=dummyscore)
resultsqda=qdacv.fit(X_train2resp,y_train2resp)


# In[ ]:


bpqda=resultsqda.best_params_
bsqda=resultsqda.best_score_
bpqda


# In[ ]:


predqda=resultsqda.predict(X_ttest)
y_true=y_ttest["Pobre"]
performqda, CMqda,FNRqda,FPRqda =dummyscore2( y_true, predqda)
performqda


# In[ ]:


###Linear discriminant function (SVD)
lda=LinearDiscriminantAnalysis()
resultslda=lda.fit(X_train2resp, y_train2resp)


# In[ ]:


predlda=resultslda.predict(X_ttest)
y_true=y_ttest["Pobre"]
performlda, CMlda,FNRlda,FPRlda =dummyscore2( y_true, predlda)
performlda


# In[ ]:


###Linear discriminant analysis (MCO)
sh=np.linspace(0.0001,1, num=100)
parameters={'shrinkage': sh, 'solver' : [ 'lsqr', 'eigen']}
lda=LinearDiscriminantAnalysis()
ldacv= GridSearchCV(lda, parameters, scoring=dummyscore)
resultsldacv=ldacv.fit(X_train2resp,y_train2resp)


# In[ ]:


bpldacv=resultsldacv.best_params_
bsldacv=resultsldacv.best_score_
bpldacv


# In[ ]:


predldacv=resultsldacv.predict(X_ttest)
y_true=y_ttest["Pobre"]
performldacv, CMldacv,FNRldacv,FPRldacv =dummyscore2( y_true, predldacv)
performldacv


# In[ ]:


######Logit #######
c=np.linspace(0.0001,1,num=100)
parameters={'penalty': ['l1', 'l2', 'elasticnet'], 'C': c}
logistica=LogisticRegression(fit_intercept=False, random_state=911, max_iter=1000)
logcv=GridSearchCV(logistica, parameters, scoring=dummyscore)
resultslogistic=logcv.fit(X_train2resp, y_train2resp)


# In[ ]:


bplogistic=resultslogistic.best_params_
bslogistic=resultslogistic.best_score_
bplogistic


# In[ ]:


predlog=resultslogistic.predict(X_ttest)
y_true=y_ttest["Pobre"]
performlog, CMlog,FNRlog,FPRlog =dummyscore2( y_true, predlog)
performlog


# In[ ]:


##Naive Bayes Bernoulli ###
c=np.linspace(0.0001,1,num=100)
parameters={'fit_prior': ['True', 'False'], 'alpha': c}

BNB=BernoulliNB()
BNBcv=GridSearchCV(BNB, parameters, scoring=dummyscore)

continuas=["P6040", "P6426", "P5100", "arriendo", "horastotal"]

nocontinuas=[i for i in list(X_train2resp.columns) if i not in continuas]

X_train2respB=X_train2resp[nocontinuas]

resultsBNB=BNBcv.fit(X_train2respB, y_train2resp)


# In[ ]:


bpBNB=resultsBNB.best_params_
bsBNB=resultsBNB.best_score_
bpBNB


# In[ ]:


y_true=y_ttest["Pobre"]
X_ttestrespB=X_ttest[nocontinuas]
predBNB=resultsBNB.predict(X_ttestrespB)
performBNB, CMBNB,FNRBNB,FPRBNB =dummyscore2( y_true, predBNB)
performBNB


# In[ ]:


##Naive Bayes Gaussian ###
c=np.linspace(0.0001,1,num=10)
parameters={'var_smoothing': [0,1e-10,1e-9,2e-9,3e-9,9e-9,10e-9]}
continuas=["P6040", "P6426", "P5100", "arriendo", "horastotal"]
X_train2respG=X_train2resp[continuas]

GNB=GaussianNB()
GNBcv=GridSearchCV(GNB, parameters, scoring=dummyscore)

resultsGNB=GNBcv.fit(X_train2respG, y_train2resp)


# In[ ]:


bpGNB=resultsGNB.best_params_
bsGNB=resultsGNB.best_score_
bpGNB


# In[ ]:


y_true=y_ttest["Pobre"]
X_ttestrespG=X_ttest[continuas]
predGNB=resultsGNB.predict(X_ttestrespG)
performGNB, CMGNB,FNRGNB,FPRGNB =dummyscore2( y_true, predGNB)
performGNB


# In[ ]:


#### KNN ############
a=[5,10,15,20]
parameters={'n_neighbors': a}
Knn = KNeighborsClassifier(n_jobs=-1)
Knncv=GridSearchCV(Knn, parameters, scoring=dummyscore)
resultsknncv=Knncv.fit(X_train2resp,y_train2resp)


# In[ ]:


bpknn=resultsknncv.best_params_
bsknn=resultsknncv.best_score_
bpknn


# In[ ]:


y_true=y_ttest["Pobre"]
predknn=resultsknncv.predict(X_ttest)
performknn, CMknn,FNRknn,FPRknn =dummyscore2( y_true, predknn)
performknn


# In[ ]:


xred=["Occol", "saluds" ,"univ", "bajeduc" ,"microemp", "P7045col"]

###QDACV#################
r1=np.linspace(0.0001,1,num=100)
parameters={'reg_param': r1}
qda=QuadraticDiscriminantAnalysis()
qdacv= GridSearchCV(qda, parameters, scoring=dummyscore)
X_train2respres=X_train2resp[xred]
resultsqdares=qdacv.fit(X_train2respres,y_train2resp)


# In[ ]:


y_true=y_ttest["Pobre"]
X_ttestres=X_ttest[xred]
predqdares=resultsqdares.predict(X_ttestres)
performqdares, CMqdares,FNRqdares,FPRqdares=dummyscore2( y_true, predqdares)


# In[ ]:


print(performqdares,FNRqdares,FPRqdares)


# In[ ]:


###Linear discriminant analysis ###
xred=["Occol", "saluds" ,"univ", "bajeduc" ,"microemp"]


sh=np.linspace(0.0001,1, num=100)
parameters={'shrinkage': sh, 'solver' : [ 'lsqr', 'eigen']}
lda=LinearDiscriminantAnalysis()
ldacv= GridSearchCV(lda, parameters, scoring=dummyscore)
X_train2respres=X_train2resp[xred]
resultsldacvres=ldacv.fit(X_train2respres,y_train2resp)


# In[ ]:


y_true=y_ttest["Pobre"]
X_ttestres=X_ttest[xred]
predldacvres=resultsldacvres.predict(X_ttestres)
performldares, CMldares,FNRldares,FPRldares=dummyscore2( y_true, predldacvres)


# In[ ]:


print(performldares,FNRldares,FPRldares)


# In[ ]:


X_testcol = X_testc[X_testc["P6050"]==1]


################################### Test (True) #################################
X_testcol["saluds"]=np.where(X_testcol["P6100"]==3, 1,0)
X_testcol["univ"]=np.where(X_testcol["P6210"]==6, 1,0)
X_testcol["bajeduc"]=np.where( (X_testcol["P6210"]==1) | (X_testcol["P6210"]==2) | (X_testcol["P6210"]==3) , 1,0)
X_testcol["traba"]=np.where(X_testcol["P6240"]==1, 1,0)
X_testcol["busctrab"]=np.where(X_testcol["P6240"]==2, 1,0)
X_testcol["microemp"]=np.where( (X_testcol["P6870"]==8) | (X_testcol["P6870"]==9) ,0,1)

###Hacinamiento
X_testcol["personaxhab"]=X_testcol["Nper"]/X_testcol["P5010"]
X_testcol["hacinamiento"]=np.where(X_testcol["personaxhab"]>3, 1,0)

X_testcol["vivpropia"]=np.where( (X_testcol["P5090"]==1) | (X_testcol["P5090"]==1) ,1,0)
X_testcol["arriendo"]=X_testcol["P5130"]+X_testcol["P5140"]

X_testcol["horastotal"]=X_testcol["P6800"]+X_testcol["P7045"]


listg=["P6800", "P7045","Pet", "Oc"]


for x in X_testcol:
    if x in listg:
        psm=pd.DataFrame(X_testc[i].groupby(X_testc["id"]).sum()).reset_index()
        psm=psm.rename(columns={i: i+"col"})
        X_testcol=pd.merge(X_testcol,psm)


cont=["P6040", "P6426", "P5100", "arriendo", "horastotal"]

for x in X_testcol:
    if x in cont:
       X_testcol[x]=( X_testcol[x] - X_testcol[x].mean() ) / X_testcol[x].std()


# In[ ]:


##Best clasification model
idtest=X_testcol["id"]
ee=['dominio_QUIBDO', 'dominio_MEDELLIN', 'dominio_IBAGUE', 'dominio_POPAYAN', 'dominio_PASTO', 'dominio_NEIVA', 'dominio_PEREIRA', 'dominio_CUCUTA', 'dominio_BUCARAMANGA', 'dominio_CALI', 'promedad', 'dominio_TUNJA', 'dominio_VALLEDUPAR', 'dominio_SINCELEJO', 'dominio_SANTA MARTA', 'dominio_BARRANQUILLA', 'dominio_CARTAGENA', 'Petcol', 'dominio_RESTO URBANO', 'dominio_FLORENCIA', 
'dominio_MONTERIA', 'dominio_VILLAVICENCIO', 'P6800col', 'dominio_RURAL', 'Occol', 'P7045col', 'dominio_RIOHACHA', 'dominio_MANIZALES']
for i in ee:
    X_testcol[i]=0

varsf=X_ttest.columns
X_testcolvarf=X_testcol[varsf]
predicctestqda=resultsqda.predict(X_testcolvarf)
predicctestqda=pd.DataFrame(predicctestqda)
csvclas=pd.merge(idtest,predicctestqda, left_index=True, right_index=True)
csvclas=csvclas.rename({0: 'classification_model'}, axis='columns')
csvclas
prediccionbest=csvclas.to_csv('csvcla.csv',index=False)


# In[ ]:


predicctestqr25=resultsqr25.predict(X_testcolvarf)
rqr25=pd.DataFrame(predicctestqr25)
#rqr25=np.where(predicctestqda>)
rqr25=pd.merge(rqr25,X_testcol[["Lp","Npersug"]], left_index=True, right_index=True)
#rqr25=pd.merge(rqr25,y_ttest["Ingtotugarr"], left_index=True, right_index=True)

rqr25[0]=rqr25[0]*y_ttest["Ingtotugarr"].std()+y_ttest["Ingtotugarr"].mean()

rqr25[0]=np.where(rqr25[0]<rqr25["Lp"],1,0)
rqr25


idtest=X_testcol["id"]
ee=['dominio_QUIBDO', 'dominio_MEDELLIN', 'dominio_IBAGUE', 'dominio_POPAYAN', 'dominio_PASTO', 'dominio_NEIVA', 'dominio_PEREIRA', 'dominio_CUCUTA', 'dominio_BUCARAMANGA', 'dominio_CALI', 'promedad', 'dominio_TUNJA', 'dominio_VALLEDUPAR', 'dominio_SINCELEJO', 'dominio_SANTA MARTA', 'dominio_BARRANQUILLA', 'dominio_CARTAGENA', 'Petcol', 'dominio_RESTO URBANO', 'dominio_FLORENCIA', 
'dominio_MONTERIA', 'dominio_VILLAVICENCIO', 'P6800col', 'dominio_RURAL', 'Occol', 'P7045col', 'dominio_RIOHACHA', 'dominio_MANIZALES']
for i in ee:
    X_testcol[i]=0

varsf=X_ttest.columns
X_testcolvarf=X_testcol[varsf]
predicctestqda=resultsqda.predict(X_testcolvarf)
predicctestqda=pd.DataFrame(predicctestqda)
predicctestqda


csv=pd.DataFrame(idtest)
csv=pd.merge(csv,predicctestqda[0], left_index=True, right_index=True)
csv=pd.merge(csv,rqr25[0], left_index=True, right_index=True)
csv=csv.rename(columns={ "0_x":'"classification_model"', "0_y": '"regression_model"' })
csv['"classification_model"']=csv['"classification_model"'].astype(int)
prediccionbest=csv.to_csv('C:/Users/juand/Desktop/Big Data/Taller 2/PS2_BD-ML/predictions_caraballo_pinilla_valencia_c23_r23.csv',index=False)

