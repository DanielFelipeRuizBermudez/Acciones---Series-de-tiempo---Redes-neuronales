"""Descarga de precios con yfinance, almacenamiento incremental y utilidades de horario de mercado."""

import os
import time

import numpy as np
import pandas as pd

# yfinance (y con él curl_cffi) se carga antes que TensorFlow. En Linux, si TensorFlow se carga primero,
# curl_cffi termina usando la libcurl de TensorFlow y la descarga falla con
# "Impersonating chromeXXX is not supported".
import yfinance as yf

import config

COLUMNAS = ["Open", "High", "Low", "Close", "Volume"]


def duracion(intervalo):
    return pd.Timedelta(minutes=config.MINUTOS_INTERVALO[intervalo])


def _apertura(dia):
    return dia + pd.Timedelta(hours=config.APERTURA[0], minutes=config.APERTURA[1])


def _cierre(dia):
    return dia + pd.Timedelta(hours=config.CIERRE[0], minutes=config.CIERRE[1])


def ahora_mercado():
    return pd.Timestamp.now(tz=config.ZONA_MERCADO)


def mercado_abierto(ahora=None):
    ahora = (ahora if ahora is not None else ahora_mercado()).tz_convert(config.ZONA_MERCADO)
    if ahora.weekday() >= 5:
        return False
    dia = ahora.normalize()
    return _apertura(dia) <= ahora < _cierre(dia)


def fin_de_vela(inicio, intervalo):
    return min(inicio + duracion(intervalo), _cierre(inicio.normalize()))


def siguiente_vela(inicio, intervalo):
    inicio = inicio.tz_convert(config.ZONA_MERCADO)
    candidata = inicio + duracion(intervalo)
    if candidata < _cierre(inicio.normalize()):
        return candidata
    dia = (inicio.normalize() + pd.offsets.BDay(1)).normalize()
    return _apertura(dia)


def proximas_velas(inicio, intervalo, cantidad):
    marcas = []
    actual = inicio
    for _ in range(cantidad):
        actual = siguiente_vela(actual, intervalo)
        marcas.append(actual)
    return pd.DatetimeIndex(marcas)


def velas_completas(df, intervalo, ahora=None):
    if df.empty:
        return df
    ahora = ahora if ahora is not None else ahora_mercado()
    if fin_de_vela(df.index[-1], intervalo) > ahora:
        return df.iloc[:-1]
    return df


def velas_por_dia(intervalo):
    minutos_sesion = (config.CIERRE[0] * 60 + config.CIERRE[1]) - (config.APERTURA[0] * 60 + config.APERTURA[1])
    return int(np.ceil(minutos_sesion / config.MINUTOS_INTERVALO[intervalo]))


def remuestrear(df, minutos):
    """Agrupa velas pequeñas (p. ej. 2m) en velas de `minutos`, alineadas con la apertura de las 9:30."""
    paso = pd.Timedelta(minutes=minutos)
    desfase = pd.Timedelta(hours=config.APERTURA[0], minutes=config.APERTURA[1]) % paso
    agregacion = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    salida = df.resample(paso, offset=desfase, label="left", closed="left").agg(agregacion)
    return salida.dropna(subset=["Close"])


def ultima_sesion(df):
    if df.empty:
        return df
    dia = df.index[-1].normalize()
    return df[df.index >= dia]


def ruta_csv(ticker, intervalo):
    return config.DIR_DATOS / f"{ticker.upper()}_{intervalo}.csv"


def _vacio():
    return pd.DataFrame(columns=COLUMNAS, index=pd.DatetimeIndex([], tz=config.ZONA_MERCADO), dtype=float)


def leer_local(ticker, intervalo):
    ruta = ruta_csv(ticker, intervalo)
    if not ruta.exists():
        return _vacio()
    df = pd.read_csv(ruta, index_col=0)
    if df.empty:
        return _vacio()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(config.ZONA_MERCADO)
    return df[COLUMNAS].sort_index()


def guardar_local(df, ticker, intervalo):
    config.DIR_DATOS.mkdir(parents=True, exist_ok=True)
    ruta = ruta_csv(ticker, intervalo)
    salida = df[COLUMNAS].copy()
    salida.index = salida.index.tz_convert("UTC")
    salida.index.name = "fecha_utc"
    temporal = ruta.with_name(f"{ruta.stem}.{os.getpid()}.tmp")
    salida.to_csv(temporal)
    for intento in range(5):
        try:
            os.replace(temporal, ruta)
            return
        except PermissionError:
            time.sleep(0.5 * (intento + 1))
    temporal.unlink(missing_ok=True)


def descargar(ticker, intervalo, inicio, fin=None):
    crudo = yf.Ticker(ticker).history(
        interval=intervalo,
        start=inicio,
        end=fin,
        auto_adjust=True,
        prepost=False,
        actions=False,
    )
    if crudo is None or crudo.empty:
        return _vacio()
    df = crudo[COLUMNAS].dropna(subset=["Close"]).astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(config.ZONA_MERCADO)
    df.index.name = None
    return df[~df.index.duplicated(keep="last")].sort_index()


def actualizar(ticker, intervalo):
    """Descarga lo nuevo desde Yahoo, lo une con el CSV local y devuelve (datos, mensaje_de_error)."""
    ticker = ticker.upper()
    local = leer_local(ticker, intervalo)
    ahora = pd.Timestamp.now(tz="UTC")
    limite = ahora - pd.Timedelta(days=config.HISTORIA_MAXIMA_DIAS[intervalo])
    if local.empty:
        inicio = limite
    else:
        inicio = max(local.index[-1].tz_convert("UTC") - pd.Timedelta(days=config.SOLAPAMIENTO_DIAS), limite)

    try:
        nuevo = descargar(ticker, intervalo, inicio.to_pydatetime())
    except Exception as error:
        return local, f"No se pudo descargar {ticker} ({intervalo}): {error}"

    if nuevo.empty:
        if local.empty:
            return local, f"Yahoo no devolvió datos para {ticker} ({intervalo}). Es posible que el símbolo no exista."
        return local, None

    if local.empty:
        combinado = nuevo
    else:
        combinado = pd.concat([local, nuevo])
        combinado = combinado[~combinado.index.duplicated(keep="last")].sort_index()
    guardar_local(combinado, ticker, intervalo)
    return combinado, None