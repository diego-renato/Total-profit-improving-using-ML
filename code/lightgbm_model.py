import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

link = '/kaggle/input/interbank-internacional-2019/'
train = pd.read_csv(link+"ib_base_inicial_train/ib_base_inicial_train.csv")
test = pd.read_csv(link+'ib_base_inicial_test/ib_base_inicial_test.csv')

train["prediction_id"] = train.id_persona.astype(str)+"_"+train.codmes.astype(str)

## new target setting
train['target'] = (train['margen']>0).astype(int)
print("Se definió la variable target")

# some new features 
train["linea_x_ingreso"]=train.linea_ofrecida/train.ingreso_neto
train["cem_x_linea"]=train.cem/train.linea_ofrecida
test["linea_x_ingreso"]=test.linea_ofrecida/test.ingreso_neto
test["cem_x_linea"]=test.cem/test.linea_ofrecida


# join datasets by id_persona
train = train.join(complementos, on=["id_persona", "codmes"])
train = train.join(digitall, on=["id_persona", "codmes"])
test = test.join(complementos, on=["id_persona", "codmes"])
test = test.join(digitall, on=["id_persona", "codmes"])
del complementos, digitall, camp_canal, camp_prod, camp_canal_pro,camp_pres
del digital


train = train.set_index("id_persona").join(vehicular1).join(vehicular2).join(reniec_var2).join(reniec_var3).join(reniec_var4).join(reniec_var5).join(reniec_var6).join(sunat_meses_alta).join(sunat_act_años).join(sunat_activ_econo_one_hot).join(vehicular1_max).join(vehicular2_max).join(vehicular1_min).join(vehicular2_min).join(sunat_act_años_max).join(sunat_act_años_min).join(sunat_act_años_var)
test = test.set_index("id_persona").join(vehicular1).join(vehicular2).join(reniec_var2).join(reniec_var3).join(reniec_var4).join(reniec_var5).join(reniec_var6).join(sunat_meses_alta).join(sunat_act_años).join(sunat_activ_econo_one_hot).join(vehicular1_max).join(vehicular2_max).join(vehicular1_min).join(vehicular2_min).join(sunat_act_años_max).join(sunat_act_años_min).join(sunat_act_años_var)
train = train.reset_index().set_index("prediction_id")
test = test.reset_index().set_index("prediction_id")
del vehicular1, vehicular2, reniec, sunat, sunat_años_alta, reniec_var2, reniec_var3, reniec_var4, reniec_var5, reniec_var6,sunat_act_años,sunat_activ_econo_one_hot,vehicular1_max,vehicular2_max,vehicular1_min,vehicular2_min,sunat_act_años_max,sunat_act_años_min,sunat_act_años_var

gc.collect()
train = train.reset_index()
train = train.join(rcc_tc, on=["id_persona", "codmes"])
gc.collect()
test = test.reset_index()
test = test.join(rcc_tc, on=["id_persona", "codmes"])
train = train.reset_index().set_index("prediction_id").astype("float32")
test = test.reset_index().set_index("prediction_id").astype("float32")
del  rcc_banco_mto_saldo, rcc_rango_mto_saldo, rcc_producto_mto_saldo,rcc_NaN_clasif_mto_saldo, rcc_rango_mora, rcc_nro_producto, rcc_nro_banco



non_ascii = train.columns[[not all(ord(c) < 128 for c in str(s)) for s in train.columns]].tolist()
non_ascii
for i, c in enumerate(non_ascii):
    train["non_ascii_" + str(i)] = train[c]
    train = train.drop(c, axis= 1)
    test["non_ascii_" + str(i)] = test[c]
    test = test.drop(c, axis= 1)
train = train.drop("index",axis=1)
test = test.drop("index",axis=1)

train.id_persona=train.id_persona.astype("int32")
train.codmes=train.codmes.astype("int32")
test.id_persona=test.id_persona.astype("int32")
test.codmes=test.codmes.astype("int32")

y_train = train[['codmes', 'id_persona', 'margen',"codtarget"]].copy()
y_train["prediction_id"] = y_train["id_persona"].astype(str) + "_" + y_train["codmes"].astype(str)
y_train["target"] = train.target
y_train = y_train.set_index("prediction_id")
train = train.drop(["codtarget", "margen","target"], axis=1)


# missing features treatment
train1 = train.copy()
test1 = test.copy()

columnas = train1.loc[:,(train1.isnull().sum()!=0)].columns
for i in columnas:
    train1[str(i)+"_Na"]=pd.get_dummies(train1[i].isnull(),prefix=i).iloc[:,0]
    test1[str(i)+"_Na"]=pd.get_dummies(test1[i].isnull(),prefix=i).iloc[:,0]
    
train1 = train1.loc[:,test1.isnull().sum()/len(test1)<0.4] ## descartamos las variables que tienen mas de 40% NaN
test1 = test1.loc[:,test1.isnull().sum()/len(test1)<0.4]
test1 = test1.loc[:,train1.isnull().sum()/len(train1)<0.4] ## descartamos las variables que tienen mas de 40% NaN
train1 = train1.loc[:,train1.isnull().sum()/len(train1)<0.4]

# imputation using mean, median by mounth

columnas = train1.loc[:,(train1.isnull().sum()!=0)].columns
for j in [201901,201902,201903,201904,201905,201906,201907]:
    for i in columnas:
        if train1[i].value_counts().count()<=10:
            if j<201905: 
                train1[i][train1.codmes==j]=train1[i][train1.codmes==j].fillna(train1[i][train1.codmes==j].value_counts().index[0])
            else:
                test1[i][test1.codmes==j]=test1[i][test1.codmes==j].fillna(test1[i][test1.codmes==j].value_counts().index[0])
        elif train1[i].value_counts().count()>10:
            if j<201905:
                train1[i][train1.codmes==j]=train1[i][train1.codmes==j].fillna(train1[i][train1.codmes==j].median())
            else:
                test1[i][test1.codmes==j]=test1[i][test1.codmes==j].fillna(test1[i][test1.codmes==j].median())

correlated_features = set()
correlation_matrix = train1[train1.codmes==201901].corr()
for i in range(len(correlation_matrix.columns)):
     for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) ==  1:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
correlated_features=list(correlated_features)

train1.drop(correlated_features,axis=1,inplace=True)
test1.drop(correlated_features,axis=1,inplace=True)


### Feature selection using feature importance

from lightgbm import LGBMClassifier
drop_cols = ["codmes"]
fi = []
for mes in [201901,201902,201903]:
        print("*"*10, mes, "*"*10)
        Xt = train1[train1.codmes <= mes]
        Xt = Xt[Xt.codmes != 201904]
        yt = y_train.loc[Xt.index, "codtarget"]
        Xt = Xt.drop(drop_cols, axis=1)
          
        Xv = train1[train1.codmes > mes]
        yv = y_train.loc[Xv.index, "codtarget"]
             
                
        learner = LGBMClassifier(n_estimators=10000,learning_rate=0.09,num_iterations=1000,lambda_l2 =5,
                                 lambda_l1 =12,num_leaves =8,max_depth=3,min_data_in_leaf =800)
        learner.fit(Xt, yt  , eval_metric="auc",eval_set= [(Xt, yt),(Xv.drop(drop_cols, axis=1), yv)], verbose=50,early_stopping_rounds=70)
        fi.append(pd.Series(learner.feature_importances_ / learner.feature_importances_.sum(), index=Xt.columns))
    

fi = pd.concat(fi, axis=1).mean(axis=1)

var = fi.sort_values().tail(90).to_frame().index 

train2 = train1[var].join(train1.codmes)
test2 = test1[var].join(test1.codmes)

#######################################################
######## Model using lightGBM framework ###############
#######################################################
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=123)

drop_cols = ["codmes"]
test_probs = []
train_probs = []
for i,(a,b) in enumerate(kf.split(train2,y_train.loc[train.index, "target"])) :
        print("*+*+*+*+*entrenando fold: {} ".format(i+1))
        Xt = train2.iloc[a,:]
        yt = y_train.loc[Xt.index, "target"]
        Xt = Xt.drop(drop_cols, axis=1)
        Xv = train2.iloc[b,:]
        yv = y_train.loc[Xv.index, "target"]
    
        learner = LGBMClassifier(n_estimators=10,learning_rate=0.09,num_iterations=1000,lambda_l2=1 ,lambda_l1 =4
                                 ,num_leaves =7,max_depth=5,min_data_in_leaf =500,early_stopping_rounds=70)
        learner.fit(Xt, yt  , eval_metric="auc",eval_set= [(Xt, yt),(Xv.drop(drop_cols, axis=1), yv)], verbose=50)
        train_probs.append(pd.Series(learner.predict_proba(Xv.drop(drop_cols, axis=1))[:, -1],
                                index=Xv.index, name="probs"+ str(mes)))
        est = pd.Series(learner.predict(Xv.drop(drop_cols, axis=1)))
        est2 = pd.Series(learner.predict(Xt))
        print("F1-score validation:",f1_score(yv, est))
        print("F1-score train:",f1_score(yt, est2))
        test_probs.append(pd.Series(learner.predict_proba(test2.drop(drop_cols, axis=1))[:, -1],
                                index=test2.index, name="fold_" + str(mes)  ))
        
    #y_train = y_train.join(train_probs)
    #optimization = differential_evolution(lambda c: -((y_train.loc[y_train.codmes!=mes,["probs"+ str(mes)]] > c[0]) * y_train.margen[y_train.codmes!=mes] / y_train.margen[y_train.codmes!=mes].sum()).sum(), [(0, 0.1)])
    #(( ((y_train.loc[y_train.codmes!=mes,["probs"+ str(mes)]]> optimization["x"][0]) * y_train.margen[y_train.codmes!=mes] / y_train.margen[y_train.codmes!=mes].sum()).sum()) )

test_probs = pd.concat(test_probs, axis=1).mean(axis=1)
train_probs = pd.concat(train_probs, axis=1).mean(axis=1)
