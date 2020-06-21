import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

link = '/kaggle/input/interbank-internacional-2019/'
campanias = pd.read_csv(link+'ib_base_campanias/ib_base_campanias.csv')

camp_canal = campanias.groupby(["codmes", "id_persona", "canal_asignado"]).size().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")
camp_prod = campanias.groupby(["codmes", "id_persona", "producto"]).size().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")
del campanias

import gc
gc.collect()

meses = {
    201901: slice(201808, 201810),
    201902: slice(201809, 201811),
    201903: slice(201810, 201812),
    201904: slice(201811, 201901),
    201905: slice(201812, 201902),
    201906: slice(201901, 201903),
    201907: slice(201902, 201904)
}
#meses = {
 #   201901: slice(201808, 201812),
  #  201902: slice(201808, 201901),
   # 201903: slice(201901, 201902),
   # 201904: slice(201901, 201903),
   # 201905: slice(201902, 201903),
   # 201906: slice(201811, 201904),
   #201907: slice(201812, 201904)
#}
complementos = []
for mes in meses.keys():
    print("*"*10, mes, "*"*10)
    res = pd.concat([
        camp_canal.loc[meses[mes]].groupby("id_persona").sum(),
        camp_prod.loc[meses[mes]].groupby("id_persona").sum()
        
    ],axis=1)
    res["codmes"] = mes
    res = res.reset_index().set_index(["id_persona", "codmes"]).astype("float32")
    complementos.append(res)

gc.collect()
print("contatenando complementos")
complementos = pd.concat(complementos)
gc.collect()

complementos=complementos.loc[:,complementos.var()!=0]
complementos.shape
