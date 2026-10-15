"""Predicción de la siguiente vela con los modelos guardados (lo usa el dashboard y, más adelante, el agente)."""

import json

import joblib
import numpy as np
import pandas as pd

import config
import datos
import modelos_redes
import modelos_series
import preparacion


def cargar_artefactos(ticker, intervalo=config.INTERVALO_MODELO):
    """Devuelve None si la acción todavía no se ha entrenado."""
    carpeta = config.directorio_modelos(ticker, intervalo)
    if not (carpeta / "resumen.json").exists():
        return None
    with open(carpeta / "resumen.json", encoding="utf-8") as archivo:
        resumen = json.load(archivo)
    artefactos = {"carpeta": carpeta, "resumen": resumen, "escalado": joblib.load(carpeta / "escalado.pkl"), "redes": {}}
    for tipo in modelos_redes.TIPOS:
        ruta = carpeta / f"{tipo}.keras"
        if ruta.exists():
            artefactos["redes"][tipo] = modelos_redes.cargar_modelo(ruta)
    return artefactos


def _fila(modelo, cierre, retorno):
    return {
        "modelo": modelo,
        "retorno_pct": 100 * (np.exp(retorno) - 1),
        "precio_predicho": cierre * np.exp(retorno),
        "direccion": "Sube" if retorno > 0 else "Baja",
    }


def predecir_siguiente(df_completas, intervalo, artefactos=None, modelo_prophet=None):
    """Predice la vela que sigue a la última vela completa.

    Devuelve (tabla_de_predicciones, info) donde info trae la vela de referencia y la vela objetivo.
    """
    f = preparacion.construir_caracteristicas(df_completas)
    referencia = f.index[-1]
    cierre = float(f["cierre"].iloc[-1])
    objetivo = datos.siguiente_vela(referencia, intervalo)
    filas = []

    if artefactos is not None:
        resumen = artefactos["resumen"]
        retorno_ar = modelos_series.predecir_ar_desde_dict(resumen["ar"], f["log_ret"].to_numpy())
        filas.append(_fila(f"AR({resumen['ar']['p']})", cierre, retorno_ar))

    if modelo_prophet is not None:
        pronostico = modelos_series.pronosticar_prophet(modelo_prophet, [objetivo])
        retorno_prophet = float(np.log(pronostico["yhat_precio"].iloc[0] / cierre))
        filas.append(_fila("Prophet", cierre, retorno_prophet))

    if artefactos is not None:
        escalado = artefactos["escalado"]
        x_escalado = escalado.transformar_x(f)
        for tipo, modelo in artefactos["redes"].items():
            ventana = artefactos["resumen"]["redes"][tipo]["ventana"]
            x = preparacion.ventanas(x_escalado[-ventana:], ventana)
            retorno = float(modelos_redes.predecir(modelo, x, escalado)[0])
            filas.append(_fila(modelos_redes.TIPOS[tipo], cierre, retorno))

    tabla = pd.DataFrame(filas)
    info = {"referencia": referencia, "cierre_referencia": cierre, "objetivo": objetivo}
    return tabla, info


def evaluar_ultima_sesion(completas, intervalo, artefactos=None, modelo_prophet_previo=None):
    """Lo que cada modelo habría predicho para cada vela de la última sesión, usando solo datos anteriores a ella.

    `modelo_prophet_previo` debe estar ajustado con datos hasta antes de la sesión.
    """
    dia = completas.index[-1].normalize()
    filas = []
    for j in np.flatnonzero(completas.index >= dia):
        tabla, info = predecir_siguiente(completas.iloc[:j], intervalo, artefactos, modelo_prophet_previo)
        cierre_real = float(completas["Close"].iloc[j])
        fila = {
            "vela": completas.index[j],
            "cierre_anterior": info["cierre_referencia"],
            "cierre_real": cierre_real,
            "cambio_real_pct": 100 * (cierre_real / info["cierre_referencia"] - 1),
        }
        fila["real"] = "Sube" if cierre_real > info["cierre_referencia"] else "Baja"
        for _, prediccion_modelo in tabla.iterrows():
            fila[prediccion_modelo["modelo"]] = prediccion_modelo["direccion"]
        filas.append(fila)
    return pd.DataFrame(filas)


def consenso(tabla):
    if tabla.empty:
        return None
    suben = int((tabla["direccion"] == "Sube").sum())
    bajan = len(tabla) - suben
    if suben == bajan:
        return {"direccion": "Empate", "suben": suben, "bajan": bajan}
    return {"direccion": "Sube" if suben > bajan else "Baja", "suben": suben, "bajan": bajan}


def resumen_para_agente(ticker, df_completas, tabla, info, metricas=None):
    """Resumen en texto plano del estado de la acción, pensado para pasárselo después a un agente (Gemini)."""
    ultimo = df_completas.iloc[-1]
    lineas = [
        f"Acción: {ticker}",
        f"Última vela completa: {info['referencia'].strftime('%Y-%m-%d %H:%M')} (hora NY), cierre {ultimo['Close']:.2f}",
        f"Vela a predecir: {info['objetivo'].strftime('%Y-%m-%d %H:%M')} (hora NY)",
    ]
    for _, fila in tabla.iterrows():
        lineas.append(f"- {fila['modelo']}: {fila['direccion']} ({fila['retorno_pct']:+.3f}%, precio {fila['precio_predicho']:.2f})")
    if metricas is not None:
        for _, fila in metricas.dropna(subset=["acierto_direccion"]).iterrows():
            lineas.append(f"Acierto histórico de {fila['modelo']}: {100 * fila['acierto_direccion']:.1f}%")
    return "\n".join(lineas)