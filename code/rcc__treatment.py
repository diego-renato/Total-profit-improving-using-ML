import pandas as pd
import click
@click.command()
@click.option('--dataset', type=click.Path(exists=True))
def get_rcc(dataset: str):

    rcc = pd.read_csv(dataset)

    rcc.clasif.fillna(-1, inplace=True) ### Rafael Crescenzi et al idea
    rcc.rango_mora.fillna(-1, inplace=True)### Rafael Crescenzi et al idea

    replace = rcc.cod_banco.value_counts().index[(rcc.cod_banco.value_counts()/len(rcc)*100<0.6)]
    rcc.cod_banco.replace(replace, "presencia_cod_banco_<200",inplace=True)

    replace = rcc.producto.value_counts().index[(rcc.producto.value_counts()<7000)]
    rcc.producto.replace(replace, "presencia_producto_<7000",inplace=True)

    replace = rcc.rango_mora.value_counts().index[ rcc.rango_mora.value_counts().index>=4]
    rcc.rango_mora.replace(replace, "rango_mora_>=4",inplace=True)

    rcc["presencia_rcc"] = 1


    rcc_nro_producto=rcc.groupby(["codmes", "id_persona"]).agg({"producto":"count","mto_saldo":"median"}).reset_index().set_index("codmes").sort_index().astype("int32")
    rcc_nro_producto["nro_pro_x_mto_median"]=rcc_nro_producto["producto"]*rcc_nro_producto["mto_saldo"]
    rcc_nro_producto.drop(["producto","mto_saldo"],axis=1,inplace=True)

    rcc_nro_banco=rcc.groupby(["codmes", "id_persona"]).agg({"cod_banco":"count","mto_saldo":"median"}).reset_index().set_index("codmes").sort_index().astype("int32")
    rcc_nro_banco["nro_banco_x_mto_median"]=rcc_nro_banco["cod_banco"]*rcc_nro_banco["mto_saldo"]
    rcc_nro_banco.drop(["cod_banco","mto_saldo"],axis=1,inplace=True)

    rcc_banco_mto_saldo =  rcc.groupby(["codmes", "id_persona", "cod_banco"]).mto_saldo.sum().unstack(level=2, fill_value=0)
    rcc_banco_mto_saldo.columns = [str(c) + "_v1_saldo" for c in rcc_banco_mto_saldo.columns]
    rcc_banco_mto_saldo = rcc_banco_mto_saldo.reset_index().set_index("codmes").sort_index().astype("int32")

    rcc_rango_mto_saldo = rcc.groupby(["codmes", "id_persona", "rango_mora"]).mto_saldo.sum().unstack(level=2, fill_value=0)
    rcc_rango_mto_saldo.columns =["rango_-1_saldo","rango_1_saldo","rango_2_saldo","rango_3_saldo","rango_>=4_saldo"]
    rcc_rango_mto_saldo = rcc_rango_mto_saldo.reset_index().set_index("codmes").sort_index().astype("int32")

    rcc_rango_mto_saldo_var = rcc.groupby(["codmes", "id_persona", "rango_mora"]).mto_saldo.sum().unstack(level=2, fill_value=0)
    rcc_rango_mto_saldo_var.columns =["rango_-1_saldo_var","rango_1_saldo_var","rango_2_saldo_var","rango_3_saldo_var","rango_>=4_saldo_var"]
    rcc_rango_mto_saldo_var = rcc_rango_mto_saldo_var.reset_index().set_index("codmes").sort_index().astype("int32")

    rcc_rango_mora = pd.get_dummies(rcc.rango_mora,prefix_sep="rango")
    rcc_rango_mora.columns = ["rango_-1","rango_1","rango_2","rango_3","rango_>=4"]
    rcc_rango_mora = rcc_rango_mora.join(rcc["id_persona"]).join(rcc["codmes"]).set_index("codmes").sort_index().astype("int32")

    rcc_producto_mto_saldo = rcc.groupby(["codmes", "id_persona", "producto"]).mto_saldo.sum().unstack(level=2, fill_value=0)
    rcc_producto_mto_saldo.columns =[str(c) + "_pro_saldo" for c in rcc_producto_mto_saldo.columns]
    rcc_producto_mto_saldo = rcc_producto_mto_saldo.reset_index().set_index("codmes").sort_index().astype("int32")

    rcc_NaN_clasif_mto_saldo = rcc.groupby(["codmes", "id_persona", "clasif"]).mto_saldo.sum().unstack(level=2, fill_value=0)
    rcc_NaN_clasif_mto_saldo.columns = ["clas_-1_saldo","clas_0_saldo","clas_1_saldo"]
    rcc_NaN_clasif_mto_saldo = rcc_NaN_clasif_mto_saldo.reset_index().set_index("codmes").sort_index().astype("int32")
    rcc_NaN_clasif_mto_saldo.drop("clas_-1_saldo",axis=1,inplace=True)

    rcc_NaN_clasif_mto_saldo_var = rcc.groupby(["codmes", "id_persona", "clasif"]).mto_saldo.sum().unstack(level=2, fill_value=0)
    rcc_NaN_clasif_mto_saldo_var.columns = ["clas_-1_saldo_var","clas_0_saldo_var","clas_1_saldo_var"]
    rcc_NaN_clasif_mto_saldo_var = rcc_NaN_clasif_mto_saldo_var.reset_index().set_index("codmes").sort_index().astype("int32")
    rcc_NaN_clasif_mto_saldo_var.drop("clas_-1_saldo_var",axis=1,inplace=True)

    rcc_nro_clasif = rcc.groupby(["codmes", "id_persona"]).mto_saldo.sum().reset_index().set_index("codmes").sort_index().astype("int32")

    rcc_presencia = rcc.groupby(["codmes", "id_persona"]).presencia_rcc.sum().reset_index().set_index("codmes").sort_index().astype("int32")
    del rcc



    import gc
    gc.collect()

    meses = {
        201901: slice(201807, 201809),
        201902: slice(201808, 201810),
        201903: slice(201809, 201811),
        201904: slice(201810, 201812),
        201905: slice(201811, 201901),
        201906: slice(201812, 201902),
        201907: slice(201901, 201903)
    }
    rcc_tc = []
    for mes in meses.keys():
        print("esperando, trabajando en el mes:", mes, "!"*10)
        res = pd.concat([
            rcc_nro_producto.loc[meses[mes]].groupby("id_persona").sum(),
            rcc_nro_banco.loc[meses[mes]].groupby("id_persona").sum(),
            rcc_banco_mto_saldo.loc[meses[mes]].groupby("id_persona").sum(),
            #rcc_banco_mto_saldo_var.loc[meses[mes]].groupby("id_persona").var(),
            rcc_rango_mto_saldo.loc[meses[mes]].groupby("id_persona").sum(),
            rcc_rango_mto_saldo_var.loc[meses[mes]].groupby("id_persona").var(),
            rcc_rango_mora.loc[meses[mes]].groupby("id_persona").sum(),
            rcc_producto_mto_saldo.loc[meses[mes]].groupby("id_persona").sum(),
            #rcc_nro_clasificadas_mora.loc[meses[mes]].groupby("id_persona").sum(),
            rcc_NaN_clasif_mto_saldo.loc[meses[mes]].groupby("id_persona").sum(),
            rcc_NaN_clasif_mto_saldo_var.loc[meses[mes]].groupby("id_persona").var(),
            rcc_nro_clasif.loc[meses[mes]].groupby("id_persona").sum(),
            rcc_presencia.loc[meses[mes]].groupby("id_persona").sum()
        ],axis=1)
        res["codmes"] = mes
        res = res.reset_index().set_index(["id_persona", "codmes"]).astype("float32")
        rcc_tc.append(res)
    gc.collect()
    print("contatenando los datos")
    rcc_tc = pd.concat(rcc_tc)
    gc.collect()

    rcc_tc = rcc_tc.loc[:,rcc_tc.var()>0.001]

    correlated_features = set()
    correlation_matrix =rcc_tc.corr()

    for i in range(len(correlation_matrix.columns)):
         for j in range(i):
                if abs(correlation_matrix.iloc[i, j])  >0.8:
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)
                    print(correlation_matrix.columns[i],"....",correlation_matrix.columns[j],correlation_matrix.iloc[i,j])

    rcc_tc.drop(list(correlated_features),axis=1,inplace=True)

    rcc_tc.to_csv("rcc_final.csv", header=True)
if __name__ == '__main__':
    get_rcc()


