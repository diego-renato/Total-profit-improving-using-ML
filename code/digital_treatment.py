import pandas as pd
import click
import gc
@click.command()
@click.option('--dataset', type=click.Path(exists=True))
def get_digital(dataset: str):

    digital = pd.read_csv(dataset)
    digital["codmes"] = (digital.codday/100).astype(int)

    numero_arribadas = ['benefit', 'email', 'facebook', 'goog', 'youtb', 'compb']
    numero_sesiones = ['movil', 'desktop']
    tiempo_visita = ['time_ctasimple', 'time_mllp', 'time_mllst', 'time_ctasld', 'time_tc']
    numero_buscadores = ['busqtc', 'busqvisa', 'busqamex',
                         'busqmc', 'busqcsimp', 'busqmill', 'busqcsld','busq']

    digital["numero_arribadas"] = digital[numero_arribadas].sum(axis=1, skipna=True)
    digital["numero_sesiones"] = digital[numero_sesiones].sum(axis=1, skipna=True)
    digital["tiempo_visita"] = digital[tiempo_visita].sum(axis=1, skipna=True)
    digital["numero_buscadores"] = digital[numero_buscadores].sum(axis=1, skipna=True)
    digital_simu_prestamo = digital.groupby(['codmes',"id_persona"]).simu_prestamo.mean()
    digital_numero_arribadas_mean = digital.groupby(['codmes',"id_persona"]).numero_arribadas.mean()
    digital_numero_sesiones_mean = digital.groupby(['codmes',"id_persona"]).numero_sesiones.mean()
    digital_tiempo_visita_mean = digital.groupby(['codmes',"id_persona"]).tiempo_visita.mean()
    digital_numero_buscadores_mean = digital.groupby(['codmes',"id_persona"]).numero_buscadores.mean()


    gc.collect()

    meses = {
        201901: slice(201811),
        201902: slice(2018011,201912),
        201903: slice(2018012),
        201904: slice(201901),
        201905: slice(201902),
        201906: slice(201903),
        201907: slice(201904)
    }

    digital_final = []
    for mes in meses.keys():
        print("*"*10, mes, "*"*10)
        res = pd.concat([
            digital_simu_prestamo.loc[meses[mes]].groupby("id_persona").sum(),
            digital_numero_arribadas_mean.loc[meses[mes]].groupby("id_persona").sum(),
            digital_numero_sesiones_mean.loc[meses[mes]].groupby("id_persona").sum(),
            digital_tiempo_visita_mean.loc[meses[mes]].groupby("id_persona").sum(),
            digital_numero_buscadores_mean.loc[meses[mes]].groupby("id_persona").sum()

        ], axis=1)
        res["codmes"] = mes
        res = res.reset_index().set_index(["id_persona", "codmes"]).astype("float32")
        digital_final.append(res)

    gc.collect()
    print("contatenando digitales")
    digital_final = pd.concat(digital_final)
    gc.collect()
    digital_final.to_csv("final_digital.csv", header=True)

if __name__ == '__main__':
    get_digital()