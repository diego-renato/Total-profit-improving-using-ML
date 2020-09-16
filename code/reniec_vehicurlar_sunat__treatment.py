import pandas as pd
import click
@click.command()
@click.option('--sunat_dataset', type=click.Path(exists=True))
@click.option('--reniec_dataset', type=click.Path(exists=True))
@click.option('--vehicular_dataset', type=click.Path(exists=True))
def get_personal_info(sunat_dataset: str, reniec_dataset: str, vehicular_dataset: str):

    sunat = pd.read_csv(sunat_dataset)
    reniec = pd.read_csv(reniec_dataset)
    vehicular = pd.read_csv(vehicular_dataset)

    vehicular.veh_var1.fillna(vehicular.veh_var1.median(),inplace=True)
    vehicular.veh_var2.fillna(vehicular.veh_var2.median(),inplace=True)
    vehicular.marca.fillna("SIN_MARCA",inplace=True)

    replace = vehicular.marca.value_counts().index[(vehicular.marca.value_counts()/len(vehicular)*100<0.6)]
    vehicular.marca.replace(replace, "presencia_auto_<3",inplace=True)

    replace = sunat.activ_econo.value_counts().index[(sunat.activ_econo.value_counts()/len(sunat.activ_econo)*100<0.3)]
    sunat.activ_econo.replace(replace, "presencia_sunat_<1_por",inplace=True)

    for i in [2,4,5]:
        if i==2:
            replace = reniec["soc_var"+str(i)].value_counts().index[(reniec["soc_var"+str(i)].value_counts()/len(reniec)*100<2.5)]
            reniec["soc_var"+str(i)].replace(replace, 2.5,inplace=True)
        else:
            replace = reniec["soc_var"+str(i)].value_counts().index[(reniec["soc_var"+str(i)].value_counts()/len(reniec)*100<1.5)]
            reniec["soc_var"+str(i)].replace(replace, 1.5,inplace=True)

    sunat_activ_econo_one_hot = pd.get_dummies(sunat.activ_econo)
    sunat_activ_econo_one_hot = pd.concat([sunat_activ_econo_one_hot,sunat.id_persona],axis=1)
    sunat_activ_econo_one_hot =sunat_activ_econo_one_hot.loc[sunat_activ_econo_one_hot.id_persona.duplicated()==False,].set_index("id_persona")

    sunat_meses_alta = sunat.groupby("id_persona").meses_alta.sum()
    sunat_años_alta = pd.DataFrame(sunat_meses_alta)
    sunat_act_años= sunat.groupby(["id_persona", "activ_econo"]).meses_alta.mean().unstack(level=1, fill_value=0).astype("float32")
    sunat_act_años_max= sunat.groupby(["id_persona", "activ_econo"]).meses_alta.max().unstack(level=1, fill_value=0).astype("float32")
    sunat_act_años_min= sunat.groupby(["id_persona", "activ_econo"]).meses_alta.min().unstack(level=1, fill_value=0).astype("float32")
    sunat_act_años_var= sunat.groupby(["id_persona", "activ_econo"]).meses_alta.var().unstack(level=1, fill_value=0).astype("float32")

    #vehicular
    vehicular1 = vehicular.groupby(["id_persona", "marca"]).veh_var1.sum().unstack(level=1, fill_value=0).astype("float32")
    vehicular2 = vehicular.groupby(["id_persona", "marca"]).veh_var2.sum().unstack(level=1, fill_value=0).astype("float32")
    vehicular1_max = vehicular.groupby(["id_persona", "marca"]).veh_var1.max().unstack(level=1, fill_value=0).astype("float32")
    vehicular2_max = vehicular.groupby(["id_persona", "marca"]).veh_var2.max().unstack(level=1, fill_value=0).astype("float32")
    vehicular1_min = vehicular.groupby(["id_persona", "marca"]).veh_var1.min().unstack(level=1, fill_value=0).astype("float32")
    vehicular2_min = vehicular.groupby(["id_persona", "marca"]).veh_var2.min().unstack(level=1, fill_value=0).astype("float32")
    #reniec = reniec.set_index("id_persona").astype("float32")
    reniec_var2 = reniec.groupby(["id_persona", "soc_var2"]).soc_var1.max().unstack(level=1, fill_value=0).astype("float32")
    reniec_var3 = reniec.groupby(["id_persona", "soc_var3"]).soc_var1.max().unstack(level=1, fill_value=0).astype("float32")
    reniec_var4 = reniec.groupby(["id_persona", "soc_var4"]).soc_var1.max().unstack(level=1, fill_value=0).astype("float32")
    reniec_var5 = reniec.groupby(["id_persona", "soc_var5"]).soc_var1.max().unstack(level=1, fill_value=0).astype("float32")
    reniec_var6 = reniec.groupby(["id_persona", "soc_var6"]).soc_var1.max().unstack(level=1, fill_value=0).astype("float32")
    #reniec_var7 = reniec.groupby(["id_persona", "cluster"]).soc_var1.sum().unstack(level=1, fill_value=0).astype("float32")
    del vehicular

    vehicular1.columns = [c + "_v1" for c in vehicular1.columns]
    vehicular2.columns = [c + "_v2" for c in vehicular2.columns]
    vehicular1_max.columns = [c + "_v1_max" for c in vehicular1_max.columns]
    vehicular2_max.columns = [c + "_v2_max" for c in vehicular2_max.columns]
    vehicular1_min.columns = [c + "_v1_min" for c in vehicular1_min.columns]
    vehicular2_min.columns = [c + "_v2_min" for c in vehicular2_min.columns]
    sunat_activ_econo_one_hot.columns = [c + "__sunat" for c in sunat_activ_econo_one_hot.columns]
    sunat_act_años.columns = [c + "_mean_anhos" for c in sunat_act_años.columns]
    sunat_act_años_max.columns = [c + "_mean_anhos_max" for c in sunat_act_años_max.columns]
    sunat_act_años_min.columns = [c + "_mean_anhos_min" for c in sunat_act_años_min.columns]
    sunat_act_años_var.columns = [c + "_mean_anhos_var" for c in sunat_act_años_var.columns]
    reniec_var2.columns =  [str(c) + "_soc_2_max__soc_1" for c in reniec_var2.columns]
    reniec_var3.columns =  [str(c) + "_soc_3_max__soc_1" for c in reniec_var3.columns]
    reniec_var4.columns =  [str(c) + "_soc_4_max__soc_1" for c in reniec_var4.columns]
    reniec_var5.columns =  [str(c) + "_soc_5_max__soc_1" for c in reniec_var5.columns]
    reniec_var6.columns =  [str(c) + "_soc_6_max__soc_1" for c in reniec_var6.columns]


    vehicular1 = vehicular1.loc[:,vehicular1.var()>=0.00002]
    vehicular2 = vehicular2.loc[:,vehicular2.var()>=0.00002]
    vehicular1_max = vehicular1_max.loc[:,vehicular1_max.var()>=0.00002]
    vehicular2_max = vehicular2_max.loc[:,vehicular2_max.var()>=0.00002]
    vehicular1_min = vehicular1_min.loc[:,vehicular1_min.var()>=0.00002]
    vehicular2_min = vehicular2_min.loc[:,vehicular2_min.var()>=0.00002]

    personal_info_dataset = vehicular1.join(vehicular2).join(reniec_var2).join(reniec_var3).join(reniec_var4).join(reniec_var5).join(
        reniec_var6).join(sunat_meses_alta).join(sunat_act_años).join(sunat_activ_econo_one_hot).join(
        vehicular1_max).join(vehicular2_max).join(vehicular1_min).join(vehicular2_min).join(sunat_act_años_max).join(
        sunat_act_años_min).join(sunat_act_años_var)
    personal_info_dataset.to_csv("personal_info_dataset.csv", header=True)

if __name__ == '__main__':
    get_personal_info()

