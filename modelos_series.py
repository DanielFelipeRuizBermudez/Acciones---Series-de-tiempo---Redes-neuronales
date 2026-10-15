"""Modelos de series de tiempo: AR(p) sobre log-retornos y Prophet sobre el log del precio."""

import itertools
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import config
import datos
import preparacion

for _nombre in ("cmdstanpy", "prophet"):
    _registro = logging.getLogger(_nombre)
    if not _registro.handlers:
        _manejador = logging.StreamHandler()
        _manejador.setLevel(logging.WARNING)
        _registro.addHandler(_manejador)
    _registro.setLevel(logging.WARNING)
    _registro.propagate = False


# ---------------------------------------------------------------- AR(p)

def matriz_rezagos(retornos, p):
    """Fila j = [r_t, r_(t-1), ..., r_(t-p+1)] con t = j + p - 1."""
    vistas = np.lib.stride_tricks.sliding_window_view(retornos, window_shape=p)
    return np.ascontiguousarray(vistas[:, ::-1])


def _xy_ar(f, p, a, b):
    rezagos = matriz_rezagos(f["log_ret"].to_numpy(), p)
    a = max(a, p - 1)
    return rezagos[a - p + 1: b - p + 1], f["objetivo"].to_numpy()[a:b]


def ajustar_ar(f, p, fin):
    """Ajusta el AR(p) con las filas [0, fin)."""
    x, y = _xy_ar(f, p, 0, fin)
    return LinearRegression().fit(x, y)


def predecir_ar(modelo, f, p, a, b):
    x, _ = _xy_ar(f, p, a, b)
    return modelo.predict(x)


def buscar_ar(f, cortes, p_max=config.AR_MAX_P, k=config.PLIEGUES_CV):
    """Elige p por validación cruzada temporal y reajusta con todo el tramo de desarrollo."""
    pliegues = preparacion.pliegues_temporales(cortes, k)
    objetivo = f["objetivo"].to_numpy()
    filas = []
    for p in range(1, p_max + 1):
        rmse = []
        for pliegue in pliegues:
            a, b = pliegue.tramo("validacion")
            pred = predecir_ar(ajustar_ar(f, p, pliegue.fin_entrenamiento), f, p, a, b)
            rmse.append(np.sqrt(np.mean((objetivo[a:b] - pred) ** 2)))
        filas.append({"p": p, "rmse_cv_pct": 100 * float(np.mean(rmse)), "desv_pliegues_pct": 100 * float(np.std(rmse))})
    tabla = pd.DataFrame(filas)
    mejor_p = int(tabla.loc[tabla["rmse_cv_pct"].idxmin(), "p"])
    return mejor_p, ajustar_ar(f, mejor_p, cortes.fin_validacion), tabla


def ar_a_dict(modelo, p):
    return {"p": p, "intercepto": float(modelo.intercept_), "coeficientes": [float(c) for c in modelo.coef_]}


def predecir_ar_desde_dict(parametros, ultimos_retornos):
    """ultimos_retornos en orden cronológico; se usan los p más recientes."""
    p = parametros["p"]
    rezagos = np.asarray(ultimos_retornos, dtype=float)[-p:][::-1]
    return float(parametros["intercepto"] + np.dot(parametros["coeficientes"], rezagos))


# ---------------------------------------------------------------- Prophet

def a_formato_prophet(serie_cierre):
    return pd.DataFrame({
        "ds": serie_cierre.index.tz_convert(config.ZONA_MERCADO).tz_localize(None),
        "y": np.log(serie_cierre.to_numpy(dtype=float)),
    })


def parametros_prophet(parametros=None):
    completos = dict(config.PROPHET_PARAMETROS_POR_DEFECTO)
    if parametros:
        completos.update({k: v for k, v in parametros.items() if k in completos})
    return completos


def ajustar_prophet(serie_cierre, parametros=None, incertidumbre=True):
    from prophet import Prophet

    modelo = Prophet(
        **parametros_prophet(parametros),
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality="auto",
        uncertainty_samples=1000 if incertidumbre else 0,
    )
    modelo.fit(a_formato_prophet(serie_cierre))
    return modelo


def pronosticar_prophet(modelo, marcas):
    """Pronóstico en las marcas de tiempo dadas. yhat y sus bandas quedan en escala de precio."""
    futuro = pd.DataFrame({"ds": pd.DatetimeIndex(marcas).tz_convert(config.ZONA_MERCADO).tz_localize(None)})
    pronostico = modelo.predict(futuro)
    for columna in ("yhat", "yhat_lower", "yhat_upper", "trend"):
        if columna in pronostico:
            pronostico[f"{columna}_precio"] = np.exp(pronostico[columna])
    return pronostico


def backtest_prophet(f, a, b, parametros, cada):
    """Para cada fila i en [a, b) predice el precio de la vela i+1 usando solo datos hasta i.

    Para no reajustar en cada vela, el modelo se reajusta cada `cada` velas y entre reajustes
    se usa el último modelo ajustado (que nunca ve datos posteriores a la fila donde se ajustó).
    Devuelve el log-retorno implícito log(precio_predicho / cierre_actual).
    """
    cierre = f["cierre"]
    marcas = f.index
    precios = np.full(b - a, np.nan)
    for inicio in range(a, b, cada):
        fin = min(inicio + cada, b)
        modelo = ajustar_prophet(cierre.iloc[: inicio + 1], parametros, incertidumbre=False)
        pronostico = pronosticar_prophet(modelo, marcas[inicio + 1: fin + 1])
        precios[inicio - a: fin - a] = pronostico["yhat_precio"].to_numpy()
    return np.log(precios / cierre.to_numpy()[a:b])


def buscar_prophet(f, cortes, intervalo, rejilla=config.PROPHET_REJILLA, al_terminar=None):
    """Búsqueda en rejilla: elige los hiperparámetros con menor error cuadrático medio del retorno en validación."""
    a, b = cortes.tramo("validacion")
    cada = datos.velas_por_dia(intervalo) * config.PROPHET_DIAS_ENTRE_REAJUSTES_VALIDACION
    real = f["objetivo"].to_numpy()[a:b]
    nombres = list(rejilla)
    combinaciones = list(itertools.product(*(rejilla[n] for n in nombres)))
    filas = []
    for numero, valores in enumerate(combinaciones, start=1):
        parametros = dict(zip(nombres, valores))
        pred = backtest_prophet(f, a, b, parametros, cada)
        fila = {**parametros, "mse_validacion": float(np.mean((real - pred) ** 2))}
        filas.append(fila)
        if al_terminar:
            al_terminar(numero, len(combinaciones), fila)
    tabla = pd.DataFrame(filas).sort_values("mse_validacion").reset_index(drop=True)
    mejor = {n: float(tabla.loc[0, n]) for n in nombres}
    return mejor, tabla


def efectos_estacionales(modelo, intervalo):
    """Efecto (en %) de la hora del día y del día de la semana según Prophet."""
    lunes = pd.Timestamp("2024-01-01", tz=config.ZONA_MERCADO)
    apertura = lunes + pd.Timedelta(hours=config.APERTURA[0], minutes=config.APERTURA[1])
    paso = datos.duracion(intervalo)
    dia = pd.DatetimeIndex([apertura + k * paso for k in range(datos.velas_por_dia(intervalo))])
    semana = pd.DatetimeIndex([lunes + pd.Timedelta(days=d, hours=12) for d in range(5)])
    p_dia = modelo.predict(pd.DataFrame({"ds": dia.tz_localize(None)}))
    p_semana = modelo.predict(pd.DataFrame({"ds": semana.tz_localize(None)}))
    hora = pd.DataFrame({"hora": dia.strftime("%H:%M"), "efecto_pct": 100 * (np.exp(p_dia["daily"]) - 1)})
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    semanal = pd.DataFrame({"dia": dias, "efecto_pct": 100 * (np.exp(p_semana["weekly"]) - 1)})
    return hora, semanal