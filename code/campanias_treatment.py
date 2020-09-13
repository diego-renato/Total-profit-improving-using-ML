import pandas as pd
import numpy as np
import click
import gc

@click.command()
@click.option('--dataset', type=click.Path(exists=True))
def get_dataset(dataset: str):

    campanias = pd.read_csv(dataset)

    camp_canal = campanias.groupby(["codmes", "id_persona", "canal_asignado"]).size().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")
    camp_prod = campanias.groupby(["codmes", "id_persona", "producto"]).size().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")
    del campanias

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
    complementos = pd.concat(complementos)
    gc.collect()

    complementos=complementos.loc[:, complementos.var() != 0]
    complementos.to_csv("final_campanias.csv", header=True)

if __name__ == '__main__':
    get_dataset()