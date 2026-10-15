"""Pronóstico a varias velas (horas) con los modelos ganadores.

AR, RNN y LSTM se usan de forma recursiva: cada predicción se agrega a la historia y sirve de entrada
para la siguiente. Prophet evalúa directamente su función del tiempo en las horas futuras.
"""

import numpy as np
import pandas as pd

import config
import datos
import modelos_redes
import modelos_series
import preparacion

Z_80 = 1.2816


def _recorridos_ar(parametros, retornos, posiciones, horizonte):
    p = parametros["p"]
    coeficientes = np.asarray(parametros["coeficientes"], dtype=float)
    salida = np.zeros((len(posiciones), horizonte))
    for fila, j in enumerate(posiciones):
        historia = list(retornos[j - p + 1: j + 1])
        for h in range(horizonte):
            r = parametros["intercepto"] + float(coeficientes @ np.asarray(historia[-p:])[::-1])
            salida[fila, h] = r
            historia.append(r)
    return salida


def _recorridos_red(modelo, escalado, ventana, f, posiciones, horizonte):
    """Retornos predichos (orígenes × horizonte).

    En cada paso el log-retorno nuevo es el predicho; las demás características se fijan en su
    promedio de las últimas velas, porque la red no las predice.
    """
    columnas = escalado.columnas
    matriz = f[columnas].to_numpy(dtype=float)
    largo = max(ventana, config.VENTANA_VOLUMEN)
    bloques = np.stack([matriz[j - largo + 1: j + 1] for j in posiciones])
    i_ret = columnas.index("log_ret")
    promedios = bloques[:, -config.VENTANA_VOLUMEN:, :].mean(axis=1)
    historia = bloques[:, -ventana:, :].copy()
    n, k = len(posiciones), len(columnas)
    salida = np.zeros((n, horizonte))
    for h in range(horizonte):
        x = escalado.escalador_x.transform(historia.reshape(-1, k)).reshape(n, ventana, k)
        r = modelos_redes.predecir(modelo, x, escalado)
        salida[:, h] = r
        nueva = promedios.copy()
        nueva[:, i_ret] = r
        historia = np.concatenate([historia[:, 1:, :], nueva[:, None, :]], axis=1)
    return salida


def precios_futuros(f, posiciones, marcas, horizonte, artefactos=None, modelos_prophet=None):
    """Precio predicho por cada modelo en las `horizonte` velas que siguen a cada origen.

    `marcas[i]` son las horas futuras del origen `posiciones[i]`; `modelos_prophet[i]` es un Prophet
    ajustado solo con datos hasta ese origen (o None para omitir Prophet).
    Devuelve {nombre_modelo: matriz orígenes × horizonte}.
    """
    cierre = f["cierre"].to_numpy(dtype=float)[posiciones][:, None]
    resultado = {}
    if artefactos is not None:
        parametros = artefactos["resumen"]["ar"]
        r = _recorridos_ar(parametros, f["log_ret"].to_numpy(dtype=float), posiciones, horizonte)
        resultado[f"AR({parametros['p']})"] = cierre * np.exp(np.cumsum(r, axis=1))
    if modelos_prophet is not None and all(m is not None for m in modelos_prophet):
        resultado["Prophet"] = np.stack([
            modelos_series.pronosticar_prophet(modelo, marca)["yhat_precio"].to_numpy()
            for modelo, marca in zip(modelos_prophet, marcas)
        ])
    if artefactos is not None:
        for tipo, modelo in artefactos["redes"].items():
            ventana = artefactos["resumen"]["redes"][tipo]["ventana"]
            r = _recorridos_red(modelo, artefactos["escalado"], ventana, f, posiciones, horizonte)
            resultado[modelos_redes.TIPOS[tipo]] = cierre * np.exp(np.cumsum(r, axis=1))
    return resultado


def banda_caminata_aleatoria(f, posicion, horizonte, velas_volatilidad=500):
    """Intervalo del 80% para el precio si se comportara como una caminata aleatoria: P·exp(±z·σ·√h)."""
    retornos = f["log_ret"].iloc[max(0, posicion - velas_volatilidad + 1): posicion + 1]
    sigma = float(retornos.std())
    h = np.arange(1, horizonte + 1)
    cierre = float(f["cierre"].iloc[posicion])
    return cierre * np.exp(-Z_80 * sigma * np.sqrt(h)), cierre * np.exp(Z_80 * sigma * np.sqrt(h)), sigma


def pronosticar(df_completas, intervalo, horizonte, artefactos=None, modelo_prophet=None):
    """Pronóstico desde la última vela de `df_completas`. Devuelve (tabla por hora futura, info del origen)."""
    f = preparacion.construir_caracteristicas(df_completas)
    posicion = len(f) - 1
    marcas = datos.proximas_velas(f.index[-1], intervalo, horizonte)
    precios = precios_futuros(f, [posicion], [marcas], horizonte, artefactos,
                              [modelo_prophet] if modelo_prophet is not None else None)
    tabla = pd.DataFrame({nombre: valores[0] for nombre, valores in precios.items()}, index=marcas)
    baja, alta, sigma = banda_caminata_aleatoria(f, posicion, horizonte)
    tabla["banda_baja"], tabla["banda_alta"] = baja, alta
    tabla.index.name = "vela"
    info = {"origen": f.index[-1], "cierre_origen": float(f["cierre"].iloc[-1]), "sigma": sigma}
    return tabla, info


def errores_por_horizonte(df_completas, horizonte, posiciones, artefactos=None, prophet_para=None):
    """Error y acierto de dirección promedio a 1, 2, ..., `horizonte` velas, sobre varios orígenes.

    Solo se usan orígenes con `horizonte` velas reales después. `prophet_para(j)` debe devolver un
    Prophet ajustado únicamente con datos anteriores al origen j.
    """
    f = preparacion.construir_caracteristicas(df_completas)
    posiciones = [j for j in posiciones if j + horizonte < len(f)]
    marcas = [f.index[j + 1: j + 1 + horizonte] for j in posiciones]
    modelos = [prophet_para(j) for j in posiciones] if prophet_para else None
    precios = precios_futuros(f, posiciones, marcas, horizonte, artefactos, modelos)
    cierre = f["cierre"].to_numpy(dtype=float)
    origen = cierre[posiciones][:, None]
    real = np.stack([cierre[j + 1: j + 1 + horizonte] for j in posiciones])
    precios["Ingenuo (sin cambio)"] = np.repeat(origen, horizonte, axis=1)
    h = pd.Index(np.arange(1, horizonte + 1), name="horas_adelante")
    error = pd.DataFrame({n: 100 * np.mean(np.abs(p - real) / real, axis=0) for n, p in precios.items()}, index=h)
    acierto = pd.DataFrame({n: 100 * np.mean((p > origen) == (real > origen), axis=0)
                            for n, p in precios.items() if not n.startswith("Ingenuo")}, index=h)
    acierto["Siempre sube"] = 100 * np.mean(real > origen, axis=0)
    return error, acierto, len(posiciones)
