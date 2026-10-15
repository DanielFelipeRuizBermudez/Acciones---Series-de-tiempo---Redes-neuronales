"""Métricas de evaluación sobre el conjunto de prueba."""

import numpy as np
import pandas as pd


def acierto_direccion(r_real, r_pred):
    return float(np.mean((np.asarray(r_pred) > 0) == (np.asarray(r_real) > 0)))


def margen_95(proporcion, n):
    return float(1.96 * np.sqrt(proporcion * (1 - proporcion) / n))


def metricas(nombre, r_real, cierre_actual, r_pred=None, direccion=None, con_direccion=True):
    """r_* son log-retornos de la vela siguiente. `direccion` (booleana) se usa si el modelo solo da sube/baja."""
    r_real = np.asarray(r_real, dtype=float)
    n = len(r_real)
    fila = {"modelo": nombre, "n": n}
    if r_pred is not None:
        r_pred = np.asarray(r_pred, dtype=float)
        precio_real = cierre_actual * np.exp(r_real)
        precio_pred = cierre_actual * np.exp(r_pred)
        fila["mae_precio"] = float(np.mean(np.abs(precio_real - precio_pred)))
        fila["rmse_retorno_pct"] = float(100 * np.sqrt(np.mean((r_real - r_pred) ** 2)))
        if direccion is None and con_direccion:
            direccion = r_pred > 0
    if direccion is not None:
        acierto = float(np.mean(np.asarray(direccion) == (r_real > 0)))
        fila["acierto_direccion"] = acierto
        fila["margen_95"] = margen_95(acierto, n)
    return fila


def lineas_base(r_real, r_actual, cierre_actual):
    """Referencias mínimas que un modelo debería superar."""
    return [
        metricas("Ingenuo (precio sin cambio)", r_real, cierre_actual, r_pred=np.zeros_like(r_real), con_direccion=False),
        metricas("Siempre sube", r_real, cierre_actual, direccion=np.ones(len(r_real), dtype=bool)),
        metricas("Repite la última vela", r_real, cierre_actual, r_pred=r_actual),
    ]


def tabla_metricas(filas):
    columnas = ["modelo", "acierto_direccion", "margen_95", "mae_precio", "rmse_retorno_pct", "n"]
    tabla = pd.DataFrame(filas)
    return tabla[[c for c in columnas if c in tabla]]
