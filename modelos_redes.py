"""Redes neuronales recurrentes (RNN simple y LSTM) para predecir el log-retorno de la siguiente vela."""

import itertools
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import config
import preparacion

TIPOS = {"rnn": "RNN simple", "lstm": "LSTM"}


def _keras():
    from tensorflow import keras
    return keras


def construir_modelo(tipo, ventana, n_caracteristicas, unidades, capas, dropout, tasa=config.TASA_APRENDIZAJE):
    keras = _keras()
    celda = keras.layers.SimpleRNN if tipo == "rnn" else keras.layers.LSTM
    modelo = keras.Sequential(name=tipo)
    modelo.add(keras.Input(shape=(ventana, n_caracteristicas)))
    for capa in range(capas):
        modelo.add(celda(unidades, return_sequences=capa < capas - 1, dropout=dropout))
    modelo.add(keras.layers.Dense(1))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=tasa), loss="mse")
    return modelo


def entrenar_modelo(modelo, x_entrenamiento, y_entrenamiento, x_validacion, y_validacion, verbose=0):
    keras = _keras()
    parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PACIENCIA, restore_best_weights=True)
    historial = modelo.fit(
        x_entrenamiento, y_entrenamiento.reshape(-1, 1),
        validation_data=(x_validacion, y_validacion.reshape(-1, 1)),
        epochs=config.EPOCAS,
        batch_size=config.TAMANO_LOTE,
        callbacks=[parada],
        shuffle=True,
        verbose=verbose,
    )
    return pd.DataFrame(historial.history)


def _evaluar_pliegue(tipo, f, pliegue, ventana, unidades, capas, dropout, tasa):
    """Entrena con el tramo de entrenamiento del pliegue y mide el error en su tramo de validación."""
    escalado = preparacion.ajustar_escalado(f, pliegue)
    conjuntos = preparacion.datos_para_red(f, pliegue, escalado, ventana)
    x_tr, y_tr = conjuntos["entrenamiento"]
    x_va, y_va = conjuntos["validacion"]
    modelo = construir_modelo(tipo, ventana, x_tr.shape[2], unidades, capas, dropout, tasa)
    historial = entrenar_modelo(modelo, x_tr, y_tr, x_va, y_va)
    pred = predecir(modelo, x_va, escalado)
    rmse = float(np.sqrt(np.mean((escalado.invertir_y(y_va) - pred) ** 2)))
    mejor_epoca = int(historial["val_loss"].to_numpy().argmin()) + 1
    return rmse, mejor_epoca, historial, int(modelo.count_params())


def buscar_hiperparametros(tipo, f, cortes, escalado_final, rejilla=config.REJILLA_REDES, k=config.PLIEGUES_CV,
                           al_terminar=None):
    """Búsqueda en rejilla con validación cruzada temporal (TimeSeriesSplit).

    Cada combinación se entrena k veces, una por pliegue, y se califica con el RMSE promedio de validación.
    La mejor se reentrena con todo el tramo de desarrollo durante el número de épocas promedio de sus pliegues.
    """
    keras = _keras()
    pliegues = preparacion.pliegues_temporales(cortes, k)
    tasas = rejilla.get("tasa", [config.TASA_APRENDIZAJE])
    combinaciones = list(itertools.product(rejilla["ventana"], rejilla["unidades"], rejilla["capas"], rejilla["dropout"], tasas))
    filas, mejor_fila, historiales_mejor = [], None, None
    for numero, (ventana, unidades, capas, dropout, tasa) in enumerate(combinaciones, start=1):
        rmse, epocas, historiales = [], [], []
        for numero_pliegue, pliegue in enumerate(pliegues, start=1):
            keras.utils.set_random_seed(config.SEMILLA)
            error, epoca, historial, parametros = _evaluar_pliegue(tipo, f, pliegue, ventana, unidades, capas, dropout, tasa)
            rmse.append(error)
            epocas.append(epoca)
            historiales.append(historial.assign(pliegue=str(numero_pliegue)))
        fila = {
            "ventana": ventana, "unidades": unidades, "capas": capas, "dropout": dropout, "tasa": tasa,
            "rmse_cv_pct": 100 * float(np.mean(rmse)), "desv_pliegues_pct": 100 * float(np.std(rmse)),
            "epocas_promedio": float(np.mean(epocas)), "parametros": parametros,
        }
        filas.append(fila)
        if al_terminar:
            al_terminar(numero, len(combinaciones), fila)
        if mejor_fila is None or fila["rmse_cv_pct"] < mejor_fila["rmse_cv_pct"]:
            mejor_fila, historiales_mejor = fila, historiales

    keras.utils.set_random_seed(config.SEMILLA)
    desarrollo = preparacion.cortes_desarrollo(cortes)
    conjuntos = preparacion.datos_para_red(f, desarrollo, escalado_final, mejor_fila["ventana"])
    x_dev, y_dev = conjuntos["entrenamiento"]
    modelo = construir_modelo(tipo, mejor_fila["ventana"], x_dev.shape[2], mejor_fila["unidades"], mejor_fila["capas"],
                              mejor_fila["dropout"], mejor_fila["tasa"])
    epocas_finales = max(1, int(round(mejor_fila["epocas_promedio"])))
    final = modelo.fit(x_dev, y_dev.reshape(-1, 1), epochs=epocas_finales, batch_size=config.TAMANO_LOTE,
                       shuffle=True, verbose=0)
    historial = pd.concat(historiales_mejor + [pd.DataFrame(final.history).assign(pliegue="final")])
    historial = historial.rename_axis("epoca").reset_index()
    mejor = {"fila": {**mejor_fila, "epocas_finales": epocas_finales}, "modelo": modelo,
             "historial": historial, "conjuntos": conjuntos}
    return mejor, pd.DataFrame(filas).sort_values("rmse_cv_pct").reset_index(drop=True)


def predecir(modelo, x, escalado):
    """Devuelve log-retornos en la escala original."""
    return escalado.invertir_y(modelo.predict(x, verbose=0).ravel())


def guardar_modelo(modelo, ruta):
    modelo.save(str(ruta))


def cargar_modelo(ruta):
    return _keras().models.load_model(str(ruta), compile=False)