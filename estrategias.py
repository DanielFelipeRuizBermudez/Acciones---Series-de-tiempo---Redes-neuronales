"""Estrategias de compra y venta y su backtest.

Cada estrategia decide al cierre de una vela si queda comprada (1) o fuera del mercado (0) durante la vela siguiente,
usando solo información disponible hasta ese cierre. No se usan ventas en corto.

- Comprar y mantener: compra al inicio y no vende. Es la referencia.
- Tres velas: después de VELAS_SEGUIDAS velas rojas seguidas compra (el precio viene bajando y se espera un rebote);
  después de VELAS_SEGUIDAS verdes seguidas vende; en cualquier otro caso mantiene lo que venía haciendo.
- CNN de velas: comprada cuando la red convolucional, mirando la imagen de las últimas velas, da una probabilidad de
  subida mayor que CNN_UMBRAL.
"""

import numpy as np
import pandas as pd

import config

NOMBRE_BH = "Comprar y mantener"
NOMBRE_VELAS = f"{config.VELAS_SEGUIDAS} velas seguidas"
NOMBRE_CNN = "CNN de velas"
COLUMNA_CNN = "CNN de velas"


def colores(velas):
    """1 si la vela es verde (cerró por encima de su apertura), -1 si es roja, 0 si cerró igual."""
    return np.sign(velas["Close"].to_numpy(dtype=float) - velas["Open"].to_numpy(dtype=float))


def posicion_tres_velas(velas, seguidas=config.VELAS_SEGUIDAS):
    """Compra tras `seguidas` velas rojas seguidas, vende tras `seguidas` verdes y si no mantiene el estado."""
    color = colores(velas)
    posicion = np.zeros(len(color))
    dentro = 0.0
    for i in range(len(color)):
        if i >= seguidas - 1:
            ultimas = color[i - seguidas + 1: i + 1]
            if (ultimas < 0).all():
                dentro = 1.0
            elif (ultimas > 0).all():
                dentro = 0.0
        posicion[i] = dentro
    return pd.Series(posicion, index=velas.index)


def posicion_cnn(probabilidades, umbral=config.CNN_UMBRAL):
    return (np.asarray(probabilidades, dtype=float) > umbral).astype(float)


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


def backtest(velas, predicciones=None, costo_pct=config.COSTO_OPERACION_PCT, ultimas=512):
    """Simula las estrategias en el mismo periodo.

    `velas`: velas de 1 hora completas (Open, High, Low, Close) con toda la historia disponible.
    `predicciones`: el archivo predicciones_prueba.csv del entrenamiento (fuera de muestra). Si existe, el periodo es
    el de prueba y se incluye la CNN; si no, se usan las últimas `ultimas` velas y solo las estrategias sin modelos.
    Devuelve (capital por estrategia, tabla de métricas).
    """
    costo = costo_pct / 100
    velas_utc = velas.copy()
    velas_utc.index = velas_utc.index.tz_convert("UTC")
    cierre = velas_utc["Close"]
    if predicciones is not None:
        referencias = pd.DatetimeIndex(pd.to_datetime(predicciones["marca_referencia"], utc=True))
        objetivos = pd.DatetimeIndex(pd.to_datetime(predicciones["marca_objetivo"], utc=True))
        retorno = np.expm1(predicciones["retorno_real"].to_numpy(dtype=float))
    else:
        referencias = cierre.index[-ultimas - 1:-1]
        objetivos = cierre.index[-ultimas:]
        retorno = (cierre.shift(-1) / cierre - 1).loc[referencias].to_numpy()

    posiciones = {
        NOMBRE_BH: np.ones(len(referencias)),
        NOMBRE_VELAS: posicion_tres_velas(velas_utc).reindex(referencias).fillna(0).to_numpy(),
    }
    if predicciones is not None and COLUMNA_CNN in predicciones:
        posiciones[NOMBRE_CNN] = posicion_cnn(predicciones[COLUMNA_CNN].fillna(0))

    capitales, filas = {}, []
    for nombre, posicion in posiciones.items():
        neto, capital = simular(posicion, retorno, costo)
        capitales[nombre] = capital
        filas.append(metricas(nombre, posicion, neto, capital))
    curvas = pd.DataFrame(capitales, index=objetivos.tz_convert(config.ZONA_MERCADO))
    tabla = pd.DataFrame(filas)
    tabla["vs_comprar_y_mantener_pct"] = tabla["retorno_total_pct"] - tabla.loc[0, "retorno_total_pct"]
    return curvas, tabla


def senales_actuales(velas, probabilidad_cnn=None, seguidas=config.VELAS_SEGUIDAS):
    """Posición que tomaría cada estrategia para la próxima vela con los datos más recientes."""
    nombres = {1: "verde", -1: "roja", 0: "sin cambio"}
    color = colores(velas.iloc[-seguidas:])
    posicion = int(posicion_tres_velas(velas, seguidas).iloc[-1])
    if (color < 0).all():
        motivo = f"Las últimas {seguidas} velas son rojas: el precio viene bajando y compra esperando un rebote"
    elif (color > 0).all():
        motivo = f"Las últimas {seguidas} velas son verdes: el precio viene subiendo y vende"
    else:
        estado = "comprado" if posicion == 1 else "fuera del mercado"
        motivo = (f"Últimas {seguidas} velas: {', '.join(nombres[int(c)] for c in color)}. No son {seguidas} del mismo "
                  f"color, así que mantiene lo que venía haciendo ({estado})")
    filas = [
        {"estrategia": NOMBRE_BH, "posicion": 1, "motivo": "Siempre comprado"},
        {"estrategia": NOMBRE_VELAS, "posicion": posicion, "motivo": motivo},
    ]
    if probabilidad_cnn is not None:
        filas.append({"estrategia": NOMBRE_CNN, "posicion": int(probabilidad_cnn > config.CNN_UMBRAL),
                      "motivo": f"La CNN da {100 * probabilidad_cnn:.0f}% de probabilidad de que la próxima hora suba "
                                f"(compra si supera {100 * config.CNN_UMBRAL:.0f}%)"})
    return pd.DataFrame(filas)