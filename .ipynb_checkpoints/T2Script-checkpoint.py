{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Problem-Set #2, BD&MLfAE\n",
    "# David Santiago Carballo Candela, 201813007\n",
    "# Sergio David Pinilla Padilla, 201814755\n",
    "# Juan Diego Valencia Romero, 201815561"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Packages:\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pyreadr as pyr\n",
    "import sklearn as sk\n",
    "import matplotlib.pyplot as plt\n",
    "import scipy as sc\n",
    "#import imblearn as ib\n",
    "import os\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.impute import KNNImputer\n",
    "from collections import Counter\n",
    "#from imblearn.over_sampling import SMOTE\n",
    "#from imblearn.over_sampling import SMOTENC\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from psmpy.plotting import *\n",
    "from sklearn import linear_model\n",
    "from sklearn.linear_model import LassoCV\n",
    "from sklearn.linear_model import RidgeCV\n",
    "from sklearn.linear_model import ElasticNetCV\n",
    "import scipy as sc\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "#from sklearn.metrics import d2_pinball_score, make_scorer\n",
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.linear_model import ElasticNet\n",
    "from sklearn.linear_model import QuantileRegressor\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.naive_bayes import BernoulliNB\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.naive_bayes import GaussianNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "odict_keys([None])\n",
      "odict_keys([None])\n"
     ]
    }
   ],
   "source": [
    "#Set directory\n",
    "os.chdir(\"path\")\n",
    "\n",
    "####Train ################################################\n",
    "tr_p=pyr.read_r(\"train_personas.Rds\") \n",
    "tr_h=pyr.read_r(\"train_hogares.Rds\")\n",
    "print(tr_p.keys())\n",
    "print(tr_h.keys())\n",
    "df_trp=tr_p[None] #train Data frame (individuals). \n",
    "df_trh=tr_h[None] #Train Data frame (households)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.merge(df_trp, df_trh, on=\"id\") #Train master data frame (merge by unique identificator key). \n",
    "df.rename(columns={\"Clase_x\": \"clase\"})\n",
    "df.rename(columns={\"Dominio_x\": \"Dominio\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Test ############################################################################################\n",
    "te_p=pyr.read_r(\"test_personas.Rds\") \n",
    "te_h=pyr.read_r(\"test_hogares.Rds\")\n",
    "print(te_p.keys())\n",
    "print(te_h.keys())\n",
    "df_tep=te_p[None] #test Data frame (individuals). \n",
    "df_teh=te_h[None] #test Data frame (households). "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test=pd.merge(df_tep, df_teh, on=\"id\") #Train master data frame (merge by unique identificator key). \n",
    "df_test.rename(columns={\"Clase_x\": \"clase\"})\n",
    "df_test.rename(columns={\"Dominio_x\": \"Dominio\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Missing values count/share in train.\n",
    "df.isnull().sum() \n",
    "df.isnull().sum()/len(df) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "c0=[i for i in df_test.columns]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "######################################Train #############################################################################\n",
    "\n",
    "#Convert categorical variables to dummy variables:\n",
    "estrato1_d=pd.get_dummies(df[\"Estrato1\"], prefix=\"estrato\") \n",
    "maxeduc_d=pd.get_dummies(df[\"P6210\"], prefix=\"educ\") \n",
    "dominio_d=pd.get_dummies(df[\"Dominio_x\"], prefix=\"dominio\")\n",
    "departamento_d=pd.get_dummies(df[\"Depto_x\"], prefix=\"depto\")\n",
    "salud_d=pd.get_dummies(df[\"P6100\"], prefix=\"salud\")\n",
    "trabajo_d=pd.get_dummies(df[\"P6430\"], prefix=\"trabajo\")\n",
    "actividad_d=pd.get_dummies(df[\"P6240\"], prefix=\"act\")\n",
    "numper_d=pd.get_dummies(df[\"P6870\"], prefix=\"numper\")\n",
    "ocseg_d=pd.get_dummies(df[\"P7050\"], prefix=\"ocseg\")\n",
    "trabdeso_d=pd.get_dummies(df[\"P7350\"], prefix=\"trabdeso\")\n",
    "tipovivienda_d=pd.get_dummies(df[\"P5090\"], prefix=\"tipoviv\")\n",
    "oficio_d=pd.get_dummies(df[\"Oficio\"], prefix=\"oficio\")\n",
    "\n",
    "\n",
    "#Merge dummy's variables data frame with master data frame:\n",
    "df=pd.merge(df, estrato1_d, left_index=True, right_index=True)\n",
    "df=pd.merge(df, maxeduc_d, left_index=True, right_index=True)\n",
    "df=pd.merge(df, dominio_d, left_index=True, right_index=True)\n",
    "df=pd.merge(df, departamento_d, left_index=True, right_index=True)\n",
    "df=pd.merge(df, trabajo_d, left_index=True, right_index=True)\n",
    "df=pd.merge(df, salud_d, left_index=True, right_index=True)\n",
    "df=pd.merge(df, actividad_d, left_index=True, right_index=True)\n",
    "df=pd.merge(df, numper_d, left_index=True, right_index=True)\n",
    "df=pd.merge(df, ocseg_d, left_index=True, right_index=True)\n",
    "df=pd.merge(df, trabdeso_d, left_index=True, right_index=True)\n",
    "df=pd.merge(df, tipovivienda_d, left_index=True, right_index=True)\n",
    "df=pd.merge(df, oficio_d, left_index=True, right_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##################Test ###############################################################################################\n",
    "\n",
    "maxeduc_d=pd.get_dummies(df_test[\"P6210\"], prefix=\"educ\") \n",
    "dominio_d=pd.get_dummies(df_test[\"Dominio_x\"], prefix=\"dominio\")\n",
    "departamento_d=pd.get_dummies(df_test[\"Depto_x\"], prefix=\"depto\")\n",
    "salud_d=pd.get_dummies(df_test[\"P6100\"], prefix=\"salud\")\n",
    "trabajo_d=pd.get_dummies(df_test[\"P6430\"], prefix=\"trabajo\")\n",
    "actividad_d=pd.get_dummies(df_test[\"P6240\"], prefix=\"act\")\n",
    "numper_d=pd.get_dummies(df_test[\"P6870\"], prefix=\"numper\")\n",
    "ocseg_d=pd.get_dummies(df_test[\"P7050\"], prefix=\"ocseg\")\n",
    "trabdeso_d=pd.get_dummies(df_test[\"P7350\"], prefix=\"trabdeso\")\n",
    "tipovivienda_d=pd.get_dummies(df_test[\"P5090\"], prefix=\"tipoviv\")\n",
    "oficio_d=pd.get_dummies(df_test[\"Oficio\"], prefix=\"oficio\")\n",
    "\n",
    "#Merge dummy's variables data frame with master data frame:\n",
    "df_test=pd.merge(df_test, maxeduc_d, left_index=True, right_index=True)\n",
    "df_test=pd.merge(df_test, dominio_d, left_index=True, right_index=True)\n",
    "df_test=pd.merge(df_test, departamento_d, left_index=True, right_index=True)\n",
    "df_test=pd.merge(df_test, salud_d, left_index=True, right_index=True)\n",
    "df_test=pd.merge(df_test, trabajo_d, left_index=True, right_index=True)\n",
    "df_test=pd.merge(df_test, actividad_d, left_index=True, right_index=True)\n",
    "df_test=pd.merge(df_test, numper_d, left_index=True, right_index=True)\n",
    "df_test=pd.merge(df_test, ocseg_d, left_index=True, right_index=True)\n",
    "df_test=pd.merge(df_test, trabdeso_d, left_index=True, right_index=True)\n",
    "df_test=pd.merge(df_test, tipovivienda_d, left_index=True, right_index=True)\n",
    "df_test=pd.merge(df_test, oficio_d, left_index=True, right_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Recode variables of train ###########\n",
    "var=([\"P6020\", \"P6090\", \"P6510\", \"P6545\", \"P6580\", \"P6585s1\", \"P6585s2\", \"P6585s3\", \"P6585s4\", \"P6590\", \"P6600\", \"P6610\", \"P6620\"\n",
    " ,\"P6630s1\", \"P6630s2\", \"P6630s3\", \"P6630s4\" ,\"P6630s6\" , \"P6920\",\"P7040\",\"P7090\",\"P7110\",\"P7120\",\"P7150\", \"P7160\", \"P7310\",\n",
    " \"P7422\", \"P7472\", \"P7495\", \"P7500s2\", \"P7500s3\",\"P7505\", \"P7510s1\", \"P7510s2\",\"P7510s3\", \"P7510s5\", \"P7510s6\", \"P7510s7\"])\n",
    "\n",
    "ceros=[\"Oc\", \"Des\", \"Ina\"]\n",
    "\n",
    "\n",
    "for i in var:\n",
    "    df[i]=np.where(df[i]==9, np.nan, df[i])\n",
    "    df[i]=np.where(df[i]==1, 1, 0*df[i])\n",
    "\n",
    "##Replace Na=0 in Ocupados, Desocupados and Inactivos\n",
    "for i in ceros:\n",
    "    df[i]=np.where(df[i]==1, 1, 0)\n",
    "\n",
    "#Recode variables of test ###########\n",
    "for i in var:\n",
    "    df_test[i]=np.where(df_test[i]==9, np.nan, df_test[i])\n",
    "    df_test[i]=np.where(df_test[i]==1, 1, 0*df_test[i])\n",
    "\n",
    "for i in ceros:\n",
    "    df_test[i]=np.where(df_test[i]==1, 1, 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Generate descriptive statistics of train dataset\n",
    "ds=(df[[\"Ingtotugarr\", \"Ingtot\", \"P6040\", \"Nper\", \"Pobre\", \"P6020\", \"estrato_1.0\", \"estrato_2.0\", \"estrato_3.0\", \n",
    "\"estrato_4.0\", \"estrato_5.0\", \"estrato_6.0\", \"educ_1.0\", \"educ_2.0\", \"educ_3.0\", \"educ_4.0\", \"educ_5.0\", \"educ_6.0\", \"P6585s3\", \"Oc\"]].describe(include=\"all\"))\n",
    "ds=ds.T\n",
    "ds=ds[[\"count\", \"mean\", \"std\", \"min\", \"50%\", \"max\"]]\n",
    "ds=ds.round(2)\n",
    "print(ds.to_latex())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Histogram of ingtotug\n",
    "#Histogram of household per_capita income:\n",
    "df_trh[\"Ingtotugarr_mM\"]=df_trh[\"Ingtotugarr\"]/df_trh[\"Nper\"]/1000000\n",
    "plt.hist(df_trh[\"Ingtotugarr_mM\"], bins=1000, color = (0.17, 0.44, 0.69, 0.9))\n",
    "plt.xlim(0,6)\n",
    "#plt.ylim(0,10000)\n",
    "plt.xticks([i for i in range(7)])\n",
    "plt.ylabel(\"Número de hogares\")\n",
    "plt.xlabel(\"COP (millones)\")\n",
    "plt.savefig(\"histy_ipch.jpg\", bbox_inches=\"tight\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "###Split train and test using training database. \n",
    "#Train sub test database using PSM to reproduce test \n",
    "\n",
    "elim=([\"id\", \"Orden\", \"Fex_c_y\", \"Clase_y\", \"Dominio_y\", \"Depto_y\", \"Dominio_x\", \"Depto_x\",  \"Fex_dpto_y\", \"Fex_dpto_x\", \"Fex_c_x\" , \"Depto_x\", \n",
    "\"P6050\" , \"P6210s1\", \"P6210s1\" , \"educ_1.0\" , \"dominio_ARMENIA\" , \"depto_05\", \"salud_1.0\", \"salud_9.0\" , \"trabajo_9.0\", \"act_1.0\", \n",
    "\"numper_1.0\", \"ocseg_1.0\", \"trabdeso_1.0\", \"tipoviv_1.0\", \"oficio_1.0\", \"P6100\", \"P6210\", \"P6240\" , \"Oficio\", \"P6430\", \"P6870\", \"P7050\", \n",
    "\"P7350\" , \"P5090\"])\n",
    "\n",
    "df[\"test\"]=0\n",
    "df_test[\"test\"]=1\n",
    "c=[i for i in df_test.columns if i not in elim]\n",
    "\n",
    "####Dealing with nans\n",
    "imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')\n",
    "x1=df_test[c]\n",
    "x1=imp_mean.fit_transform(x1)\n",
    "\n",
    "x2=df[c]\n",
    "x2=imp_mean.fit_transform(x2)\n",
    "\n",
    "imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')\n",
    "\n",
    "x1c=df_test[c0]\n",
    "x1c=imp_mean.fit_transform(x1c)\n",
    "\n",
    "\n",
    "x2c=df[c0]\n",
    "x2c=imp_mean.fit_transform(x2c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x111c=pd.DataFrame(x2c, columns=c0)\n",
    "x111c=x111c[x111c[\"P6050\"]==1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a X matrix of covariates\n",
    "x11=pd.DataFrame(x1, columns=c)\n",
    "x22=pd.DataFrame(x2, columns=c)\n",
    "X=x22.append(x11, ignore_index=True)\n",
    "\n",
    "#Create matrix Y\n",
    "Y=X[\"test\"]\n",
    "Y=Y.astype('int')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Generate PSM of test\n",
    "lr = LogisticRegression(random_state=911, class_weight=\"balanced\")\n",
    "result=lr.fit(X,Y)\n",
    "PSM=result.predict_proba(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "PSM=pd.DataFrame(PSM, columns=[\"no\", \"si\"])\n",
    "PSM\n",
    "X=pd.merge(X,PSM, left_index=True, right_index=True) \n",
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "grouped = X.groupby(X.test)\n",
    "X_train = grouped.get_group(0)\n",
    "X_test = grouped.get_group(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.hist([X_test.si,X_train.si], bins=10, label=[\"test\", \"train\"], color=[\"r\", \"b\"] )\n",
    "plt.legend(loc=\"upper right\")\n",
    "plt.xticks([0,0.2,0.4,0.6,0.8,1])\n",
    "plt.ylabel(\"Frecuencia\")\n",
    "plt.xlabel(\"Probabilidad\")\n",
    "plt.savefig(\"histy_psm.jpg\", bbox_inches=\"tight\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x11c=pd.DataFrame(x1c, columns=c0)\n",
    "x22c=pd.DataFrame(x2c, columns=c0)\n",
    "x11c[\"test\"]=1\n",
    "x22c[\"test\"]=0\n",
    "\n",
    "Xc=x22c.append(x11c, ignore_index=True)\n",
    "\n",
    "Xc=pd.merge(Xc, X.si, left_index=True, right_index=True)\n",
    "\n",
    "grouped = Xc.groupby(Xc.test)\n",
    "X_trainc = grouped.get_group(0)\n",
    "X_testc = grouped.get_group(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "###Dummy 1 if propensity score >0.5\n",
    "X_trainc[\"PSM\"]=np.where(X_trainc[\"si\"]>=0.5, 1,0)\n",
    "\n",
    "###Keep only jefe hogar\n",
    "X_traincol = X_trainc[X_trainc[\"P6050\"]==1]\n",
    "\n",
    "X_testcol = X_testc[X_testc[\"P6050\"]==1]\n",
    "\n",
    "########Split X_trainc in X_train2 and X_ttest\n",
    "\n",
    "psm=pd.DataFrame(X_trainc[\"PSM\"].groupby(X_trainc[\"id\"]).mean()).reset_index()\n",
    "psm=psm.rename(columns={\"PSM\": \"PSMc\"})\n",
    "X_traincol=pd.merge(X_traincol,psm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "############################## Collapse data base at household level #############################\n",
    "####Keep only parentesco=jefe hogar\n",
    "\n",
    "###########collapse train ##################\n",
    "\n",
    "psm=X_trainc[\"P6040\"].groupby(X_trainc[\"id\"]).apply(np.mean).reset_index()\n",
    "psm=psm.rename(columns={\"P6040\": \"promedad\"})\n",
    "X_traincol=pd.merge(X_traincol,psm)\n",
    "\n",
    "listg=[\"P6800\", \"P7045\",\"Pet\", \"Oc\"]\n",
    "\n",
    "for i in listg:\n",
    "    psm=pd.DataFrame(X_trainc[i].groupby(X_trainc[\"id\"]).sum()).reset_index()\n",
    "    psm=psm.rename(columns={i: i+\"col\"})\n",
    "    X_traincol=pd.merge(X_traincol,psm)\n",
    "\n",
    "\n",
    "#############collapse test####################################3\n",
    "####Keep only parentesco=jefe hogar\n",
    "\n",
    "###########collapse test ##################\n",
    "psm=X_testc[\"P6040\"].groupby(X_testc[\"id\"]).apply(np.mean).reset_index()\n",
    "psm=psm.rename(columns={\"P6040\": \"promedad\"})\n",
    "X_testcol=pd.merge(X_testcol,psm)\n",
    "\n",
    "listg=[\"P6800\", \"P7045\",\"Pet\", \"Oc\"]\n",
    "\n",
    "for i in listg:\n",
    "    psm=pd.DataFrame(X_testc[i].groupby(X_testc[\"id\"]).sum()).reset_index()\n",
    "    psm=psm.rename(columns={i: i+\"col\"})\n",
    "    X_testcol=pd.merge(X_testcol,psm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "########Merge dependent variables in X########\n",
    "df_ycol=df[df[\"P6050\"]==1]\n",
    "X_traincol=pd.merge(X_traincol,df_ycol[[\"Ingtotugarr\", \"Pobre\", \"id\"]]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "####Recode variables #################################\n",
    "##################Train ######################################################\n",
    "X_traincol[\"saluds\"]=np.where(X_traincol[\"P6100\"]==3, 1,0)\n",
    "X_traincol[\"univ\"]=np.where(X_traincol[\"P6210\"]==6, 1,0)\n",
    "X_traincol[\"bajeduc\"]=np.where( (X_traincol[\"P6210\"]==1) | (X_traincol[\"P6210\"]==2) | (X_traincol[\"P6210\"]==3) , 1,0)\n",
    "X_traincol[\"traba\"]=np.where(X_traincol[\"P6240\"]==1, 1,0)\n",
    "X_traincol[\"busctrab\"]=np.where(X_traincol[\"P6240\"]==2, 1,0)\n",
    "X_traincol[\"microemp\"]=np.where( (X_traincol[\"P6870\"]==8) | (X_traincol[\"P6870\"]==9) ,0,1)\n",
    "\n",
    "###Hacinamiento\n",
    "X_traincol[\"personaxhab\"]=X_traincol[\"Nper\"]/X_traincol[\"P5010\"]\n",
    "X_traincol[\"hacinamiento\"]=np.where(X_traincol[\"personaxhab\"]>3, 1,0)\n",
    "\n",
    "X_traincol[\"vivpropia\"]=np.where( (X_traincol[\"P5090\"]==1) | (X_traincol[\"P5090\"]==1) ,1,0)\n",
    "\n",
    "\n",
    "###################################Test #################################\n",
    "X_testcol[\"saluds\"]=np.where(X_testcol[\"P6100\"]==3, 1,0)\n",
    "X_testcol[\"univ\"]=np.where(X_testcol[\"P6210\"]==6, 1,0)\n",
    "X_testcol[\"bajeduc\"]=np.where( (X_testcol[\"P6210\"]==1) | (X_testcol[\"P6210\"]==2) | (X_testcol[\"P6210\"]==3) , 1,0)\n",
    "X_testcol[\"traba\"]=np.where(X_testcol[\"P6240\"]==1, 1,0)\n",
    "X_testcol[\"busctrab\"]=np.where(X_testcol[\"P6240\"]==2, 1,0)\n",
    "X_testcol[\"microemp\"]=np.where( (X_testcol[\"P6870\"]==8) | (X_testcol[\"P6870\"]==9) ,0,1)\n",
    "\n",
    "###Hacinamiento\n",
    "X_testcol[\"personaxhab\"]=X_testcol[\"Nper\"]/X_testcol[\"P5010\"]\n",
    "X_testcol[\"hacinamiento\"]=np.where(X_testcol[\"personaxhab\"]>3, 1,0)\n",
    "\n",
    "X_testcol[\"vivpropia\"]=np.where( (X_testcol[\"P5090\"]==1) | (X_testcol[\"P5090\"]==1) ,1,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_traincol[\"arriendo\"]=X_traincol[\"P5130\"]+X_traincol[\"P5140\"]\n",
    "\n",
    "X_traincol[\"horastotal\"]=X_traincol[\"P6800\"]+X_traincol[\"P7045\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#####Get dummies dominio\n",
    "dominio_d1=pd.get_dummies(X_traincol[\"Dominio_x\"], prefix=\"dominio\")\n",
    "X_traincol=pd.merge(X_traincol, dominio_d, left_index=True, right_index=True)\n",
    "\n",
    "dominio_d2=pd.get_dummies(X_testcol[\"Dominio_x\"], prefix=\"dominio\")\n",
    "X_testcol=pd.merge(X_testcol, dominio_d, left_index=True, right_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "###Select a subset of variables \n",
    "\n",
    "variables1=([ 'Clase_x', 'P6020', 'P6040', 'P6090', 'P6426', 'P6545', 'P6590', 'P6610', 'P6920',  'P7422', 'P7495', 'P7505', 'P5100' ,\n",
    "'promedad', 'P6800col', 'P7045col', 'Petcol', 'Occol',\n",
    "       'saluds', 'univ', 'bajeduc', 'traba', 'busctrab', 'microemp',\n",
    "       'personaxhab', 'hacinamiento', 'vivpropia', 'PSMc',\"Ingtotugarr\", \"Pobre\", \"Lp\", \"id\", \"Li\" , \"Npersug\", \"arriendo\", \"horastotal\"])\n",
    "\n",
    "\n",
    "variables2=([ 'Clase_x', 'P6020', 'P6040', 'P6090', 'P6426', 'P6545', 'P6590', 'P6610', 'P6920',  'P7422', 'P7495', 'P7505', 'P5100' ,\n",
    "'promedad', 'P6800col', 'P7045col', 'Petcol', 'Occol',\n",
    "       'saluds', 'univ', 'bajeduc', 'traba', 'busctrab', 'microemp',\n",
    "       'personaxhab', 'hacinamiento', 'vivpropia', 'id', \"Npersug\"])\n",
    "\n",
    "\n",
    "dominio_d12=[i for i in list(dominio_d1.columns) if i!=\"dominio_ARMENIA\" and i!=\"dominio_BOGOTA\"]\n",
    "dominio_d22=[i for i in list(dominio_d2.columns) if i!=\"dominio_ARMENIA\" and i!=\"dominio_BOGOTA\"]\n",
    "\n",
    "\n",
    "variables1=variables1+dominio_d12\n",
    "variables2=variables2+dominio_d22\n",
    "\n",
    "\n",
    "X_traincol=X_traincol[variables1]\n",
    "X_testcol=X_testcol[variables2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Split X train based on psm score\n",
    "\n",
    "X_ttest=X_traincol[X_traincol[\"PSMc\"]>0.5]\n",
    "X_train2=X_traincol[X_traincol[\"PSMc\"]<=0.5]\n",
    "\n",
    "dependientes=[\"Ingtotugarr\", \"Pobre\", \"Lp\", \"id\", \"Li\", \"Npersug\", \"PSMc\"]\n",
    "var1=[i for i in X_ttest.columns if i not in dependientes]\n",
    "var2=[i for i in X_train2.columns if i not in dependientes]\n",
    "var3=[i for i in X_testcol.columns if i not in dependientes]\n",
    "\n",
    "###Create dependent variables\n",
    "y_ttest=X_ttest[dependientes]\n",
    "y_train2=X_train2[dependientes]\n",
    "\n",
    "##Create independent variables\n",
    "X_ttest=X_ttest[var1]\n",
    "X_train2=X_train2[var2]\n",
    "X_testcol=X_testcol[var3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "len(y_train2[y_train2.Pobre==1])/len(y_train2.Pobre)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "len(y_ttest[y_ttest.Pobre==1])/len(y_ttest.Pobre)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##########################Regresion Models ######################################\n",
    "###Standardize continue variables and winsorize dependent variable\n",
    "\n",
    "sc.stats.mstats.winsorize(y_train2[\"Ingtotugarr\"], limits=[0,0.05], inplace=True)\n",
    "\n",
    "y_train2[\"Ingtotugarrest\"]=(y_train2[\"Ingtotugarr\"] - y_train2[\"Ingtotugarr\"].mean()) / y_train2[\"Ingtotugarr\"].std()\n",
    "\n",
    "cont=[\"P6040\", \"P6426\", \"P5100\", \"arriendo\", \"horastotal\"]\n",
    "\n",
    "\n",
    "for x in X_train2:\n",
    "    if x in cont:\n",
    "        X_train2[x]=(X_train2[x] - X_train2[x].mean()) / X_train2[x].std()\n",
    "\n",
    "for x in X_ttest:\n",
    "    if x in cont:\n",
    "        X_ttest[x]=( X_ttest[x] - X_ttest[x].mean() ) / X_ttest[x].std()\n",
    "\n",
    "for x in X_testcol:\n",
    "    if x in cont:\n",
    "        X_testcol[x]=( X_testcol[x] - X_testcol[x].mean() ) / X_testcol[x].std()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "###########Resampling X_train y_train ###################\n",
    "###Resampling for pobre\n",
    "sm = SMOTE(random_state=911)\n",
    "X_train2resp, y_train2resp = sm.fit_resample(X_train2, y_train2[[\"Pobre\"]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train2[\"Ingtotugarr\"].max()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.shape(X_train2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Lasso Regresion\n",
    "lasso = LassoCV(cv=10, random_state=911, n_jobs=-1, fit_intercept=False,n_alphas=100,selection=\"random\", max_iter=1000)\n",
    "y_train2[\"Ingtotugarr\"]=y_train2[\"Ingtotugarr\"].astype('int')\n",
    "resultslasso=lasso.fit(X_train2,y_train2[\"Ingtotugarrest\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "vars=resultslasso.feature_names_in_\n",
    "print(resultslasso.alpha_)\n",
    "print(resultslasso.n_features_in_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Ridge regression\n",
    "a=np.linspace(0.00001,100,num=100)\n",
    "ridge2 = RidgeCV( alphas=a,cv=10, fit_intercept=False)\n",
    "#n_jobs=-1, random_state=911\n",
    "resultsridge2=ridge2.fit(X_train2,y_train2[\"Ingtotugarrest\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "resultsridge2.alpha_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Elastic Net regression\n",
    "l1=np.linspace(0.0001,1,num=100)\n",
    "Elastic2 = ElasticNetCV(cv=10, random_state=911, n_jobs=-1, fit_intercept=False,n_alphas=100,selection=\"random\", max_iter=1000, l1_ratio=l1, precompute=True)\n",
    "resultselastic2=Elastic2.fit(X_train2,y_train2[\"Ingtotugarrest\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(resultselastic2.alpha_)\n",
    "print(resultselastic2.l1_ratio_)\n",
    "print(resultselastic2.n_features_in_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Lasso regression with pinball penalty\n",
    "a3=np.linspace(0.000000001,100,num=100)\n",
    "d2_pinball=make_scorer(d2_pinball_score, alpha=0.25)\n",
    "\n",
    "parameters = {'alpha': a3}\n",
    "Lqr25= ElasticNet(l1_ratio=1, fit_intercept=False, max_iter=100, random_state=911, selection=\"random\", precompute=True)\n",
    "\n",
    "Lqr25cv = GridSearchCV(Lqr25, parameters, scoring=d2_pinball, n_jobs=-1)\n",
    "\n",
    "resultslqr25=Lqr25cv.fit(X_train2,y_train2[\"Ingtotugarrest\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(resultslqr25.best_params_)\n",
    "print(resultslqr25.n_features_in_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Ridge regression with pinball penalty\n",
    "a=np.linspace(0.000000001,10,num=100)\n",
    "d2_pinball=make_scorer(d2_pinball_score, alpha=0.25)\n",
    "\n",
    "ridge = RidgeCV( alphas=a,cv=10, fit_intercept=False, scoring=d2_pinball)\n",
    "#n_jobs=-1, random_state=911\n",
    "resultsridge=ridge.fit(X_train2,y_train2[\"Ingtotugarrest\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "resultsridge.alpha_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Elastic Net regression with pinball penalty\n",
    "l1=np.linspace(0,1,num=10)\n",
    "a2=np.linspace(0,100,num=100)\n",
    "d2_pinball=make_scorer(d2_pinball_score, alpha=0.25)\n",
    "\n",
    "parameters = {'alpha': a2, \"l1_ratio\": l1}\n",
    "Elastic= ElasticNet(fit_intercept=False, max_iter=100, random_state=911, selection=\"random\", precompute=True)\n",
    "Elasticcv = GridSearchCV(Elastic, parameters, scoring=d2_pinball, n_jobs=-1)\n",
    "resultselastic=Elasticcv.fit(X_train2,y_train2[\"Ingtotugarrest\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(resultselastic.best_params_)\n",
    "print(resultselastic.n_features_in_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Quantile Regression (sklearn)\n",
    "#a=1, b=3, d=1/4\n",
    "qr25 = QuantileRegressor(quantile=0.25,alpha=0,solver=\"highs\")\n",
    "resultsqr25=qr25.fit(X_train2,y_train2[\"Ingtotugarrest\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "############### Evaluate regression models ##########################\n",
    "def dummyscore(y_true,y_pred):\n",
    "    y_pred1=y_pred*y_ttest[\"Ingtotugarr\"].std()+y_ttest[\"Ingtotugarr\"].mean()\n",
    "    y_pred2=np.where(y_pred1<y_ttest[\"Lp\"]*y_ttest[\"Npersug\"], 1,0)\n",
    "    CM = confusion_matrix(y_true, y_pred2)\n",
    "    TN = CM[0][0]\n",
    "    FN = CM[1][0]\n",
    "    TP = CM[1][1]\n",
    "    FP = CM[0][1]\n",
    "    # False negative rate\n",
    "    FNR = FN/(TP+FN)\n",
    "    #False positive rate\n",
    "    FPR = FP/(FP+TN)\n",
    "    WMIR=FNR*0.75+0.25*FPR\n",
    "    return y_pred2, CM, FNR, FPR, WMIR\n",
    "\n",
    "liss=[resultslasso, resultsridge2, resultselastic2, resultslqr25, resultsridge, resultselastic, resultsqr25]\n",
    "\n",
    "X_ttest_reg = np.array(X_ttest)\n",
    "Y_ttest_reg = np.array(y_ttest[\"Pobre\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "YPrresultslasso,CMresultslasso,FNR1,FPR1,WMIRresultslasso=dummyscore(Y_ttest_reg,resultslasso.predict(X_ttest_reg))\n",
    "YPrresultsridge2,CMresultsridge2,FNR2,FPR2,WMIRresultsridge2=dummyscore(Y_ttest_reg,resultsridge2.predict(X_ttest_reg))\n",
    "YPrresultselastic2,CMresultselastic2,FNR3,FPR3,WMIRresultselastic2=dummyscore(Y_ttest_reg,resultselastic2.predict(X_ttest_reg))\n",
    "YPrresultslqr25,CMresultslqr25,FNR4,FPR4,WMIRresultslqr25=dummyscore(Y_ttest_reg,resultslqr25.predict(X_ttest_reg))\n",
    "YPrresultsridge,CMresultsridge,FNR5,FPR5,WMIRresultsridge=dummyscore(Y_ttest_reg,resultsridge.predict(X_ttest_reg))\n",
    "YPrresultselastic,CMresultselastic,FNR6,FPR6,WMIRresultselastic=dummyscore(Y_ttest_reg,resultselastic.predict(X_ttest_reg))\n",
    "YPrresultsqr25,CMresultsqr25,FNR7,FPR7,WMIRresultsqr25=dummyscore(Y_ttest_reg,resultsqr25.predict(X_ttest_reg))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(WMIRresultslasso,WMIRresultsridge2,WMIRresultselastic2,WMIRresultslqr25,WMIRresultsridge,WMIRresultselastic,WMIRresultsqr25)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(FNR1,FNR2,FNR3,FNR4,FNR5,FNR6,FNR7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(FPR1,FPR2,FPR3,FPR4,FPR5,FPR6,FPR7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "############### Reduced sample regs ################\n",
    "\n",
    "xyr=pd.merge(X_train2,y_train2[\"Ingtotugarrest\"], left_index=True, right_index=True)\n",
    "corrxyr=xyr.corr()\n",
    "ingcmat=corrxyr[\"Ingtotugarrest\"]\n",
    "ingcmat\n",
    "corrxyr[\"Ingtotugarrest\"].to_latex(index=True)\n",
    "xredin=[\"saluds\" ,\"univ\", \"bajeduc\" ,\"microemp\",\"vivpropia\", \"hacinamiento\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train3=X_train2[xredin]\n",
    "X_train3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Ridge regression (small)\n",
    "a=np.linspace(0.00001,100,num=100)\n",
    "RidgeSmall = RidgeCV( alphas=a,cv=10, fit_intercept=False)\n",
    "#n_jobs=-1, random_state=911\n",
    "resultsridgesmall=RidgeSmall.fit(X_train3,y_train2[\"Ingtotugarrest\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(resultsridgesmall.alpha_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Quantile Regression (sklearn)\n",
    "#a=1, b=3, d=1/4\n",
    "qr25 = QuantileRegressor(quantile=0.25,alpha=0,solver=\"highs\")\n",
    "resultsqr25small=qr25.fit(X_train3,y_train2[\"Ingtotugarrest\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_ttestin=X_ttest[xredin]\n",
    "X_ttestin_reg = np.array(X_ttestin)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "YPrresultsrs,CMresultsrs,FNR8,FPR8,WMIRresultsrs=dummyscore(Y_ttest_reg,resultsridgesmall.predict(X_ttestin_reg))\n",
    "YPrresultsqs,CMresultsqs,FNR9,FPR9,WMIRresultsqs=dummyscore(Y_ttest_reg,resultsqr25small.predict(X_ttestin_reg))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(WMIRresultsrs,WMIRresultsqs)\n",
    "print(FNR8,FNR9)\n",
    "print(FPR8,FPR9)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "######################################## Clasification Models ####################################################################\n",
    "##Define penalty #######\n",
    "def dummyscore2(y,ypred):\n",
    "    CM = confusion_matrix(y, ypred)\n",
    "    TN = CM[0][0]\n",
    "    FN = CM[1][0]\n",
    "    TP = CM[1][1]\n",
    "    FP = CM[0][1]\n",
    "    # False negative rate\n",
    "    FNR = FN/(TP+FN)\n",
    "    #False positive rate\n",
    "    FPR = FP/(FP+TN)\n",
    "    penalty=FNR*0.75+0.25*FPR\n",
    "    return penalty, CM,FNR,FPR"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Quadratic discriminant function\n",
    "r1=np.linspace(0.0001,1,num=100)\n",
    "parameters={'reg_param': r1}\n",
    "qda=QuadraticDiscriminantAnalysis()\n",
    "qdacv= GridSearchCV(qda, parameters, scoring=dummyscore)\n",
    "resultsqda=qdacv.fit(X_train2resp,y_train2resp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "bpqda=resultsqda.best_params_\n",
    "bsqda=resultsqda.best_score_\n",
    "bpqda"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predqda=resultsqda.predict(X_ttest)\n",
    "y_true=y_ttest[\"Pobre\"]\n",
    "performqda, CMqda,FNRqda,FPRqda =dummyscore2( y_true, predqda)\n",
    "performqda"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "###Linear discriminant function (SVD)\n",
    "lda=LinearDiscriminantAnalysis()\n",
    "resultslda=lda.fit(X_train2resp, y_train2resp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predlda=resultslda.predict(X_ttest)\n",
    "y_true=y_ttest[\"Pobre\"]\n",
    "performlda, CMlda,FNRlda,FPRlda =dummyscore2( y_true, predlda)\n",
    "performlda"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "###Linear discriminant analysis (MCO)\n",
    "sh=np.linspace(0.0001,1, num=100)\n",
    "parameters={'shrinkage': sh, 'solver' : [ 'lsqr', 'eigen']}\n",
    "lda=LinearDiscriminantAnalysis()\n",
    "ldacv= GridSearchCV(lda, parameters, scoring=dummyscore)\n",
    "resultsldacv=ldacv.fit(X_train2resp,y_train2resp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "bpldacv=resultsldacv.best_params_\n",
    "bsldacv=resultsldacv.best_score_\n",
    "bpldacv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predldacv=resultsldacv.predict(X_ttest)\n",
    "y_true=y_ttest[\"Pobre\"]\n",
    "performldacv, CMldacv,FNRldacv,FPRldacv =dummyscore2( y_true, predldacv)\n",
    "performldacv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "######Logit #######\n",
    "c=np.linspace(0.0001,1,num=100)\n",
    "parameters={'penalty': ['l1', 'l2', 'elasticnet'], 'C': c}\n",
    "logistica=LogisticRegression(fit_intercept=False, random_state=911, max_iter=1000)\n",
    "logcv=GridSearchCV(logistica, parameters, scoring=dummyscore)\n",
    "resultslogistic=logcv.fit(X_train2resp, y_train2resp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "bplogistic=resultslogistic.best_params_\n",
    "bslogistic=resultslogistic.best_score_\n",
    "bplogistic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predlog=resultslogistic.predict(X_ttest)\n",
    "y_true=y_ttest[\"Pobre\"]\n",
    "performlog, CMlog,FNRlog,FPRlog =dummyscore2( y_true, predlog)\n",
    "performlog"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Naive Bayes Bernoulli ###\n",
    "c=np.linspace(0.0001,1,num=100)\n",
    "parameters={'fit_prior': ['True', 'False'], 'alpha': c}\n",
    "\n",
    "BNB=BernoulliNB()\n",
    "BNBcv=GridSearchCV(BNB, parameters, scoring=dummyscore)\n",
    "\n",
    "continuas=[\"P6040\", \"P6426\", \"P5100\", \"arriendo\", \"horastotal\"]\n",
    "\n",
    "nocontinuas=[i for i in list(X_train2resp.columns) if i not in continuas]\n",
    "\n",
    "X_train2respB=X_train2resp[nocontinuas]\n",
    "\n",
    "resultsBNB=BNBcv.fit(X_train2respB, y_train2resp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "bpBNB=resultsBNB.best_params_\n",
    "bsBNB=resultsBNB.best_score_\n",
    "bpBNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_true=y_ttest[\"Pobre\"]\n",
    "X_ttestrespB=X_ttest[nocontinuas]\n",
    "predBNB=resultsBNB.predict(X_ttestrespB)\n",
    "performBNB, CMBNB,FNRBNB,FPRBNB =dummyscore2( y_true, predBNB)\n",
    "performBNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Naive Bayes Gaussian ###\n",
    "c=np.linspace(0.0001,1,num=10)\n",
    "parameters={'var_smoothing': [0,1e-10,1e-9,2e-9,3e-9,9e-9,10e-9]}\n",
    "continuas=[\"P6040\", \"P6426\", \"P5100\", \"arriendo\", \"horastotal\"]\n",
    "X_train2respG=X_train2resp[continuas]\n",
    "\n",
    "GNB=GaussianNB()\n",
    "GNBcv=GridSearchCV(GNB, parameters, scoring=dummyscore)\n",
    "\n",
    "resultsGNB=GNBcv.fit(X_train2respG, y_train2resp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "bpGNB=resultsGNB.best_params_\n",
    "bsGNB=resultsGNB.best_score_\n",
    "bpGNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_true=y_ttest[\"Pobre\"]\n",
    "X_ttestrespG=X_ttest[continuas]\n",
    "predGNB=resultsGNB.predict(X_ttestrespG)\n",
    "performGNB, CMGNB,FNRGNB,FPRGNB =dummyscore2( y_true, predGNB)\n",
    "performGNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#### KNN ############\n",
    "a=[5,10,15,20]\n",
    "parameters={'n_neighbors': a}\n",
    "Knn = KNeighborsClassifier(n_jobs=-1)\n",
    "Knncv=GridSearchCV(Knn, parameters, scoring=dummyscore)\n",
    "resultsknncv=Knncv.fit(X_train2resp,y_train2resp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "bpknn=resultsknncv.best_params_\n",
    "bsknn=resultsknncv.best_score_\n",
    "bpknn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_true=y_ttest[\"Pobre\"]\n",
    "predknn=resultsknncv.predict(X_ttest)\n",
    "performknn, CMknn,FNRknn,FPRknn =dummyscore2( y_true, predknn)\n",
    "performknn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "xred=[\"Occol\", \"saluds\" ,\"univ\", \"bajeduc\" ,\"microemp\", \"P7045col\"]\n",
    "\n",
    "###QDACV#################\n",
    "r1=np.linspace(0.0001,1,num=100)\n",
    "parameters={'reg_param': r1}\n",
    "qda=QuadraticDiscriminantAnalysis()\n",
    "qdacv= GridSearchCV(qda, parameters, scoring=dummyscore)\n",
    "X_train2respres=X_train2resp[xred]\n",
    "resultsqdares=qdacv.fit(X_train2respres,y_train2resp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_true=y_ttest[\"Pobre\"]\n",
    "X_ttestres=X_ttest[xred]\n",
    "predqdares=resultsqdares.predict(X_ttestres)\n",
    "performqdares, CMqdares,FNRqdares,FPRqdares=dummyscore2( y_true, predqdares)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(performqdares,FNRqdares,FPRqdares)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "###Linear discriminant analysis ###\n",
    "xred=[\"Occol\", \"saluds\" ,\"univ\", \"bajeduc\" ,\"microemp\"]\n",
    "\n",
    "\n",
    "sh=np.linspace(0.0001,1, num=100)\n",
    "parameters={'shrinkage': sh, 'solver' : [ 'lsqr', 'eigen']}\n",
    "lda=LinearDiscriminantAnalysis()\n",
    "ldacv= GridSearchCV(lda, parameters, scoring=dummyscore)\n",
    "X_train2respres=X_train2resp[xred]\n",
    "resultsldacvres=ldacv.fit(X_train2respres,y_train2resp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_true=y_ttest[\"Pobre\"]\n",
    "X_ttestres=X_ttest[xred]\n",
    "predldacvres=resultsldacvres.predict(X_ttestres)\n",
    "performldares, CMldares,FNRldares,FPRldares=dummyscore2( y_true, predldacvres)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(performldares,FNRldares,FPRldares)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_testcol = X_testc[X_testc[\"P6050\"]==1]\n",
    "\n",
    "\n",
    "################################### Test (True) #################################\n",
    "X_testcol[\"saluds\"]=np.where(X_testcol[\"P6100\"]==3, 1,0)\n",
    "X_testcol[\"univ\"]=np.where(X_testcol[\"P6210\"]==6, 1,0)\n",
    "X_testcol[\"bajeduc\"]=np.where( (X_testcol[\"P6210\"]==1) | (X_testcol[\"P6210\"]==2) | (X_testcol[\"P6210\"]==3) , 1,0)\n",
    "X_testcol[\"traba\"]=np.where(X_testcol[\"P6240\"]==1, 1,0)\n",
    "X_testcol[\"busctrab\"]=np.where(X_testcol[\"P6240\"]==2, 1,0)\n",
    "X_testcol[\"microemp\"]=np.where( (X_testcol[\"P6870\"]==8) | (X_testcol[\"P6870\"]==9) ,0,1)\n",
    "\n",
    "###Hacinamiento\n",
    "X_testcol[\"personaxhab\"]=X_testcol[\"Nper\"]/X_testcol[\"P5010\"]\n",
    "X_testcol[\"hacinamiento\"]=np.where(X_testcol[\"personaxhab\"]>3, 1,0)\n",
    "\n",
    "X_testcol[\"vivpropia\"]=np.where( (X_testcol[\"P5090\"]==1) | (X_testcol[\"P5090\"]==1) ,1,0)\n",
    "X_testcol[\"arriendo\"]=X_testcol[\"P5130\"]+X_testcol[\"P5140\"]\n",
    "\n",
    "X_testcol[\"horastotal\"]=X_testcol[\"P6800\"]+X_testcol[\"P7045\"]\n",
    "\n",
    "\n",
    "listg=[\"P6800\", \"P7045\",\"Pet\", \"Oc\"]\n",
    "\n",
    "\n",
    "for x in X_testcol:\n",
    "    if x in listg:\n",
    "        psm=pd.DataFrame(X_testc[i].groupby(X_testc[\"id\"]).sum()).reset_index()\n",
    "        psm=psm.rename(columns={i: i+\"col\"})\n",
    "        X_testcol=pd.merge(X_testcol,psm)\n",
    "\n",
    "\n",
    "cont=[\"P6040\", \"P6426\", \"P5100\", \"arriendo\", \"horastotal\"]\n",
    "\n",
    "for x in X_testcol:\n",
    "    if x in cont:\n",
    "       X_testcol[x]=( X_testcol[x] - X_testcol[x].mean() ) / X_testcol[x].std()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Best clasification model\n",
    "idtest=X_testcol[\"id\"]\n",
    "ee=['dominio_QUIBDO', 'dominio_MEDELLIN', 'dominio_IBAGUE', 'dominio_POPAYAN', 'dominio_PASTO', 'dominio_NEIVA', 'dominio_PEREIRA', 'dominio_CUCUTA', 'dominio_BUCARAMANGA', 'dominio_CALI', 'promedad', 'dominio_TUNJA', 'dominio_VALLEDUPAR', 'dominio_SINCELEJO', 'dominio_SANTA MARTA', 'dominio_BARRANQUILLA', 'dominio_CARTAGENA', 'Petcol', 'dominio_RESTO URBANO', 'dominio_FLORENCIA', \n",
    "'dominio_MONTERIA', 'dominio_VILLAVICENCIO', 'P6800col', 'dominio_RURAL', 'Occol', 'P7045col', 'dominio_RIOHACHA', 'dominio_MANIZALES']\n",
    "for i in ee:\n",
    "    X_testcol[i]=0\n",
    "\n",
    "varsf=X_ttest.columns\n",
    "X_testcolvarf=X_testcol[varsf]\n",
    "predicctestqda=resultsqda.predict(X_testcolvarf)\n",
    "predicctestqda=pd.DataFrame(predicctestqda)\n",
    "csvclas=pd.merge(idtest,predicctestqda, left_index=True, right_index=True)\n",
    "csvclas=csvclas.rename({0: 'classification_model'}, axis='columns')\n",
    "csvclas\n",
    "prediccionbest=csvclas.to_csv('csvcla.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predicctestqr25=resultsqr25.predict(X_testcolvarf)\n",
    "rqr25=pd.DataFrame(predicctestqr25)\n",
    "#rqr25=np.where(predicctestqda>)\n",
    "rqr25=pd.merge(rqr25,X_testcol[[\"Lp\",\"Npersug\"]], left_index=True, right_index=True)\n",
    "#rqr25=pd.merge(rqr25,y_ttest[\"Ingtotugarr\"], left_index=True, right_index=True)\n",
    "\n",
    "rqr25[0]=rqr25[0]*y_ttest[\"Ingtotugarr\"].std()+y_ttest[\"Ingtotugarr\"].mean()\n",
    "\n",
    "rqr25[0]=np.where(rqr25[0]<rqr25[\"Lp\"],1,0)\n",
    "rqr25\n",
    "\n",
    "\n",
    "idtest=X_testcol[\"id\"]\n",
    "ee=['dominio_QUIBDO', 'dominio_MEDELLIN', 'dominio_IBAGUE', 'dominio_POPAYAN', 'dominio_PASTO', 'dominio_NEIVA', 'dominio_PEREIRA', 'dominio_CUCUTA', 'dominio_BUCARAMANGA', 'dominio_CALI', 'promedad', 'dominio_TUNJA', 'dominio_VALLEDUPAR', 'dominio_SINCELEJO', 'dominio_SANTA MARTA', 'dominio_BARRANQUILLA', 'dominio_CARTAGENA', 'Petcol', 'dominio_RESTO URBANO', 'dominio_FLORENCIA', \n",
    "'dominio_MONTERIA', 'dominio_VILLAVICENCIO', 'P6800col', 'dominio_RURAL', 'Occol', 'P7045col', 'dominio_RIOHACHA', 'dominio_MANIZALES']\n",
    "for i in ee:\n",
    "    X_testcol[i]=0\n",
    "\n",
    "varsf=X_ttest.columns\n",
    "X_testcolvarf=X_testcol[varsf]\n",
    "predicctestqda=resultsqda.predict(X_testcolvarf)\n",
    "predicctestqda=pd.DataFrame(predicctestqda)\n",
    "predicctestqda\n",
    "\n",
    "\n",
    "csv=pd.DataFrame(idtest)\n",
    "csv=pd.merge(csv,predicctestqda[0], left_index=True, right_index=True)\n",
    "csv=pd.merge(csv,rqr25[0], left_index=True, right_index=True)\n",
    "csv=csv.rename(columns={ \"0_x\":'\"classification_model\"', \"0_y\": '\"regression_model\"' })\n",
    "csv['\"classification_model\"']=csv['\"classification_model\"'].astype(int)\n",
    "prediccionbest=csv.to_csv('C:/Users/juand/Desktop/Big Data/Taller 2/PS2_BD-ML/predictions_caraballo_pinilla_valencia_c23_r23.csv',index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  },
  "vscode": {
   "interpreter": {
    "hash": "79d71d161e7943240a345005223b4b57f09b9732a24e4917a9c0467b3aef16ea"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
