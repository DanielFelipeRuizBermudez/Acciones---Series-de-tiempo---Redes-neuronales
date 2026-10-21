"""Estrategias de compra y venta y su backtest.

Cada estrategia decide al cierre de una vela si queda comprada (1) o fuera del mercado (0) durante la vela
siguiente, usando solo información disponible hasta ese cierre. No se usan ventas en corto.
"""

import numpy as np
import pandas as pd

import config

NOMBRE_BH = "Comprar y mantener"
NOMBRE_CRUCE = f"Cruce de medias {config.SMA_RAPIDA}/{config.SMA_LENTA}"
NOMBRE_RSI = f"RSI({config.RSI_PERIODO}) {config.RSI_COMPRA}/{config.RSI_VENTA}"
NOMBRE_CONSENSO = "Consenso de modelos"


def media_movil(cierre, n):
    return cierre.rolling(n, min_periods=n).mean()


def rsi(cierre, n=config.RSI_PERIODO):
    """Índice de fuerza relativa con el suavizado de Wilder."""
    cambio = cierre.diff()
    subida = cambio.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    bajada = (-cambio.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return 100 - 100 / (1 + subida / bajada)


def posicion_cruce(cierre, rapida=config.SMA_RAPIDA, lenta=config.SMA_LENTA):
    return (media_movil(cierre, rapida) > media_movil(cierre, lenta)).astype(float)


def posicion_rsi(cierre, n=config.RSI_PERIODO, compra=config.RSI_COMPRA, venta=config.RSI_VENTA):
    """Compra cuando el RSI baja de `compra` y vende cuando supera `venta`; entre medio mantiene el estado."""
    valores = rsi(cierre, n).to_numpy()
    posicion = np.zeros(len(valores))
    dentro = 0.0
    for i, valor in enumerate(valores):
        if not np.isnan(valor):
            if dentro == 0 and valor < compra:
                dentro = 1.0
            elif dentro == 1 and valor > venta:
                dentro = 0.0
        posicion[i] = dentro
    return pd.Series(posicion, index=cierre.index)


def posicion_modelo(retorno_predicho, umbral):
    return (np.asarray(retorno_predicho, dtype=float) > umbral).astype(float)


def posicion_consenso(predicciones, umbral):
    """Comprado cuando más de la mitad de los modelos predice un retorno mayor que el umbral."""
    votos = (predicciones.to_numpy(dtype=float) > umbral).sum(axis=1)
    return (votos > predicciones.shape[1] / 2).astype(float)


def simular(posicion, retorno_siguiente, costo):
    """Retorno neto por vela y capital acumulado (base 1). El costo se cobra en cada compra y cada venta."""
    posicion = np.asarray(posicion, dtype=float)
    cambios = np.abs(np.diff(np.r_[0.0, posicion]))
    neto = posicion * np.asarray(retorno_siguiente, dtype=float) - costo * cambios
    return neto, np.cumprod(1 + neto)


def _operaciones(posicion, neto):
    """Resultado de cada operación completa: desde la compra hasta la venta, con sus costos."""
    resultados, inicio = [], None
    for i, valor in enumerate(np.r_[posicion, 0.0]):
        if valor == 1 and inicio is None:
            inicio = i
        elif valor == 0 and inicio is not None:
            fin = min(i + 1, len(neto))
            resultados.append(float(np.prod(1 + neto[inicio:fin]) - 1))
            inicio = None
    return resultados


def metricas(nombre, posicion, neto, capital):
    posicion = np.asarray(posicion, dtype=float)
    pico = np.maximum.accumulate(capital)
    operaciones = _operaciones(posicion, neto)
    desviacion = float(np.std(neto))
    return {
        "estrategia": nombre,
        "retorno_total_pct": 100 * float(capital[-1] - 1),
        "maxima_caida_pct": 100 * float(np.min(capital / pico - 1)),
        "operaciones": len(operaciones),
        "operaciones_ganadoras_pct": 100 * float(np.mean(np.array(operaciones) > 0)) if operaciones else np.nan,
        "tiempo_invertido_pct": 100 * float(posicion.mean()),
        "sharpe_anual": float(np.mean(neto) / desviacion * np.sqrt(config.VELAS_POR_ANO)) if desviacion > 0 else np.nan,
    }


def backtest(cierre, predicciones=None, costo_pct=config.COSTO_OPERACION_PCT, umbral_pct=0.0, velas=512):
    """Simula todas las estrategias en el mismo periodo.

    `cierre`: serie de cierres por hora con toda la historia disponible.
    `predicciones`: el archivo predicciones_prueba.csv del entrenamiento (fuera de muestra). Si existe, el
    periodo es el de prueba; si no, las últimas `velas` velas y solo estrategias técnicas.
    Devuelve (capital por estrategia, tabla de métricas).
    """
    costo, umbral = costo_pct / 100, np.log1p(umbral_pct / 100)
    cierre_utc = cierre.copy()
    cierre_utc.index = cierre_utc.index.tz_convert("UTC")
    if predicciones is not None:
        referencias = pd.DatetimeIndex(pd.to_datetime(predicciones["marca_referencia"], utc=True))
        objetivos = pd.DatetimeIndex(pd.to_datetime(predicciones["marca_objetivo"], utc=True))
        retorno = np.expm1(predicciones["retorno_real"].to_numpy(dtype=float))
    else:
        referencias = cierre_utc.index[-velas - 1:-1]
        objetivos = cierre_utc.index[-velas:]
        retorno = (cierre_utc.shift(-1) / cierre_utc - 1).loc[referencias].to_numpy()

    posiciones = {
        NOMBRE_BH: np.ones(len(referencias)),
        NOMBRE_CRUCE: posicion_cruce(cierre_utc).reindex(referencias).fillna(0).to_numpy(),
        NOMBRE_RSI: posicion_rsi(cierre_utc).reindex(referencias).fillna(0).to_numpy(),
    }
    if predicciones is not None:
        columnas = [c for c in predicciones.columns if c.startswith("AR(") or c in ("Prophet", "RNN simple", "LSTM")]
        for columna in columnas:
            posiciones[f"Señal de {columna}"] = posicion_modelo(predicciones[columna], umbral)
        if len(columnas) >= 2:
            posiciones[NOMBRE_CONSENSO] = posicion_consenso(predicciones[columnas], umbral)

    capitales, filas = {}, []
    for nombre, posicion in posiciones.items():
        neto, capital = simular(posicion, retorno, costo)
        capitales[nombre] = capital
        filas.append(metricas(nombre, posicion, neto, capital))
    curvas = pd.DataFrame(capitales, index=objetivos.tz_convert(config.ZONA_MERCADO))
    tabla = pd.DataFrame(filas)
    tabla["vs_comprar_y_mantener_pct"] = tabla["retorno_total_pct"] - tabla.loc[0, "retorno_total_pct"]
    return curvas, tabla


def senales_actuales(cierre, tabla_siguiente=None, umbral_pct=0.0):
    """Posición que tomaría cada estrategia para la próxima vela con los datos más recientes."""
    umbral = np.log1p(umbral_pct / 100)
    rapida = media_movil(cierre, config.SMA_RAPIDA).iloc[-1]
    lenta = media_movil(cierre, config.SMA_LENTA).iloc[-1]
    valor_rsi = rsi(cierre).iloc[-1]
    filas = [
        {"estrategia": NOMBRE_BH, "posicion": 1, "motivo": "Siempre comprado"},
        {"estrategia": NOMBRE_CRUCE, "posicion": int(rapida > lenta),
         "motivo": f"Media de {config.SMA_RAPIDA} h = {rapida:.2f} {'>' if rapida > lenta else '≤'} "
                   f"media de {config.SMA_LENTA} h = {lenta:.2f}"},
        {"estrategia": NOMBRE_RSI, "posicion": int(posicion_rsi(cierre).iloc[-1]),
         "motivo": f"RSI = {valor_rsi:.0f} (compra bajo {config.RSI_COMPRA}, vende sobre {config.RSI_VENTA}; "
                   "entre medio mantiene lo que venía haciendo)"},
    ]
    if tabla_siguiente is not None and not tabla_siguiente.empty:
        retornos = np.log1p(tabla_siguiente["retorno_pct"].to_numpy(dtype=float) / 100)
        for (_, fila), retorno in zip(tabla_siguiente.iterrows(), retornos):
            filas.append({"estrategia": f"Señal de {fila['modelo']}", "posicion": int(retorno > umbral),
                          "motivo": f"Predice {fila['retorno_pct']:+.3f}% para la próxima hora"})
        if len(retornos) >= 2:
            votos = int((retornos > umbral).sum())
            filas.append({"estrategia": NOMBRE_CONSENSO, "posicion": int(votos > len(retornos) / 2),
                          "motivo": f"{votos} de {len(retornos)} modelos "
                                    f"{'predice' if votos == 1 else 'predicen'} una subida mayor que el umbral"})
    return pd.DataFrame(filas)
