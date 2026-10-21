"""Transformación de velas en características, división temporal y ventanas deslizantes."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import config


def construir_caracteristicas(df):
    """Una fila por vela. `objetivo` es el log-retorno de la vela siguiente (lo que se quiere predecir)."""
    f = pd.DataFrame(index=df.index)
    log_cierre = np.log(df["Close"].astype(float))
    f["log_ret"] = log_cierre.diff()
    f["rango"] = np.log(df["High"].astype(float) / df["Low"].astype(float))
    log_volumen = np.log1p(df["Volume"].astype(float))
    media_volumen = log_volumen.rolling(config.VENTANA_VOLUMEN, min_periods=7).mean()
    f["vol_rel"] = log_volumen - media_volumen
    f["cierre"] = df["Close"].astype(float)
    f = f.dropna(subset=["log_ret", "rango", "vol_rel"])
    f["objetivo"] = f["log_ret"].shift(-1)
    return f


@dataclass
class Cortes:
    """Fronteras de la división temporal sobre las filas que tienen objetivo."""
    fin_entrenamiento: int
    fin_validacion: int
    fin_prueba: int

    def tramo(self, nombre):
        return {
            "entrenamiento": (0, self.fin_entrenamiento),
            "validacion": (self.fin_entrenamiento, self.fin_validacion),
            "prueba": (self.fin_validacion, self.fin_prueba),
        }[nombre]


def calcular_cortes(f, proporcion_validacion=config.PROPORCION_VALIDACION, proporcion_prueba=config.PROPORCION_PRUEBA):
    n = int(f["objetivo"].notna().sum())
    n_prueba = int(round(n * proporcion_prueba))
    n_validacion = int(round(n * proporcion_validacion))
    return Cortes(n - n_validacion - n_prueba, n - n_prueba, n)


def cortes_desarrollo(cortes):
    """Desarrollo = entrenamiento + validación; la prueba no cambia."""
    return Cortes(cortes.fin_validacion, cortes.fin_validacion, cortes.fin_prueba)


def pliegues_temporales(cortes, k):
    """Validación cruzada temporal (TimeSeriesSplit) sobre el tramo de desarrollo.

    Cada pliegue entrena con las filas [0, fin_entrenamiento) y valida con [fin_entrenamiento, fin_validacion):
    siempre se valida con datos posteriores a los de entrenamiento.
    """
    filas = np.arange(cortes.fin_validacion)
    return [
        Cortes(int(entrenamiento[-1]) + 1, int(validacion[-1]) + 1, cortes.fin_prueba)
        for entrenamiento, validacion in TimeSeriesSplit(n_splits=k).split(filas)
    ]


def ventanas(matriz, ventana):
    """(n, k) -> (n - ventana + 1, ventana, k). La ventana j termina en la fila j + ventana - 1."""
    vistas = np.lib.stride_tricks.sliding_window_view(matriz, window_shape=ventana, axis=0)
    return np.ascontiguousarray(np.transpose(vistas, (0, 2, 1)))


@dataclass
class Escalado:
    columnas: list
    escalador_x: StandardScaler
    media_y: float
    desv_y: float

    def transformar_x(self, f):
        return self.escalador_x.transform(f[self.columnas].to_numpy())

    def transformar_y(self, y):
        return (y - self.media_y) / self.desv_y

    def invertir_y(self, y_escalado):
        return y_escalado * self.desv_y + self.media_y


def ajustar_escalado(f, cortes, columnas=config.CARACTERISTICAS):
    entrenamiento = f.iloc[: cortes.fin_entrenamiento]
    escalador = StandardScaler().fit(entrenamiento[columnas].to_numpy())
    y = entrenamiento["objetivo"].to_numpy()
    return Escalado(list(columnas), escalador, float(y.mean()), float(y.std()))


def datos_para_red(f, cortes, escalado, ventana):
    """Arma (X, y) de entrenamiento, validación y prueba, más la última ventana para predecir en vivo."""
    x_escalado = escalado.transformar_x(f)
    todas = ventanas(x_escalado, ventana)
    y = escalado.transformar_y(f["objetivo"].to_numpy())

    def tomar(a, b):
        a = max(a, ventana - 1)
        return todas[a - ventana + 1: b - ventana + 1], y[a:b]

    return {
        "entrenamiento": tomar(*cortes.tramo("entrenamiento")),
        "validacion": tomar(*cortes.tramo("validacion")),
        "prueba": tomar(*cortes.tramo("prueba")),
        "ultima": todas[-1:],
    }
