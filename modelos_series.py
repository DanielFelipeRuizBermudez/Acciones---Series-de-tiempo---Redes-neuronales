"""Modelos de series de tiempo: AR(p) y ARIMA(p, d, q) sobre el log del precio, y Prophet.

El log-retorno es la primera diferencia del log del precio: r_t = ln P_t - ln P_(t-1). Por eso un ARMA(p, q)
ajustado sobre el log-retorno es un ARIMA(p, 1, q) sobre ln(precio), y el AR(p) es un ARIMA(p, 1, 0).
"""

import itertools
import logging
import warnings

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


# ---------------------------------------------------------------- Box-Jenkins: integración y ARIMA

ESCALA_ARIMA = 100.0
NOMBRES_DIFERENCIA = {0: "ln(precio)", 1: "Δ ln(precio) = log-retorno", 2: "Δ² ln(precio)"}


def prueba_adf(serie):
    """Dickey-Fuller aumentada con constante; H0: la serie tiene raíz unitaria (no es estacionaria)."""
    from statsmodels.tsa.stattools import adfuller

    estadistico, p_valor, rezagos, n, criticos, _ = adfuller(np.asarray(serie, dtype=float), autolag="AIC")
    return {"estadistico": float(estadistico), "p_valor": float(p_valor), "rezagos": int(rezagos), "n": int(n),
            "critico_5": float(criticos["5%"])}


def orden_integracion(log_precio, alfa=config.ALFA_ADF, d_max=2):
    """Diferencia ln(precio) hasta que la prueba ADF rechace la raíz unitaria.

    Devuelve (d, pasos): d es el número de diferencias necesarias y cada paso trae la prueba de una serie.
    """
    serie = np.asarray(log_precio, dtype=float)
    pasos = []
    for d in range(d_max + 1):
        resultado = prueba_adf(serie)
        estacionaria = resultado["p_valor"] < alfa
        pasos.append({"d": d, "serie": NOMBRES_DIFERENCIA[d], **resultado, "estacionaria": bool(estacionaria)})
        if estacionaria:
            return d, pasos
        serie = np.diff(serie)
    return d_max, pasos


REQUISITOS_ARIMA = {
    "estacionario": "Estacionario: raíces del AR fuera del círculo unitario",
    "invertible": "Invertible: raíces del MA fuera del círculo unitario",
    "coef_significativos": "Coeficientes AR y MA significativos",
    "sin_autocorrelacion": "Residuos sin autocorrelación (Ljung-Box)",
    "residuos_normales": "Residuos normales (Jarque-Bera)",
}


def ajustar_arma(w, p, q):
    """ARMA(p, q) con media, por máxima verosimilitud, con errores estándar robustos (sándwich)."""
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(np.asarray(w, dtype=float), order=(p, 0, q), trend="c").fit(cov_type="robust")


def diagnostico_residuos(residuos, p, q, rezagos=config.LJUNG_BOX_REZAGOS):
    """Ljung-Box (H0: residuos sin autocorrelación) y Jarque-Bera (H0: residuos normales)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera

    e = np.asarray(residuos, dtype=float)
    ljung_box = acorr_ljungbox(e, lags=list(rezagos), model_df=p + q)
    jb, jb_p, asimetria, curtosis = jarque_bera(e)
    return {
        "ljung_box": [{"rezagos": int(k), "estadistico": float(fila["lb_stat"]), "p_valor": float(fila["lb_pvalue"])}
                      for k, fila in ljung_box.iterrows()],
        "jarque_bera": {"estadistico": float(jb), "p_valor": float(jb_p), "asimetria": float(asimetria),
                        "curtosis": float(curtosis)},
    }


def coeficientes_arima(ajuste, alfa=config.ALFA_PRUEBAS):
    """Prueba z de cada coeficiente (H0: vale 0) con errores estándar robustos e intervalo del 95%."""
    from scipy import stats

    filas = []
    for nombre, estimacion, error in zip(ajuste.model.param_names, np.asarray(ajuste.params, dtype=float),
                                         np.asarray(ajuste.bse, dtype=float)):
        if nombre == "sigma2":
            continue
        z = estimacion / error if np.isfinite(error) and error > 0 else np.nan
        p_valor = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        filas.append({"parametro": nombre, "estimacion": float(estimacion), "error_estandar": float(error),
                      "z": float(z), "p_valor": p_valor, "ic_bajo": float(estimacion - 1.96 * error),
                      "ic_alto": float(estimacion + 1.96 * error),
                      "significativo": bool(np.isfinite(p_valor) and p_valor < alfa)})
    return filas


def correlacion_estimaciones(ajuste):
    """Correlación entre las estimaciones de los coeficientes AR y MA."""
    nombres = list(ajuste.model.param_names)
    indices = [i for i, n in enumerate(nombres) if n.startswith(("ar.", "ma."))]
    if len(indices) < 2:
        return {"nombres": [nombres[i] for i in indices], "matriz": []}
    covarianza = np.asarray(ajuste.cov_params(), dtype=float)[np.ix_(indices, indices)]
    desvio = np.sqrt(np.diag(covarianza))
    with np.errstate(divide="ignore", invalid="ignore"):
        matriz = covarianza / np.outer(desvio, desvio)
    return {"nombres": [nombres[i] for i in indices],
            "matriz": [[float(v) if np.isfinite(v) else None for v in fila] for fila in matriz]}


def raices_casi_comunes(raices_ar, raices_ma, distancia=config.DISTANCIA_RAICES_COMUNES):
    """True si alguna raíz del AR casi coincide con una del MA: se cancelan y el modelo sobra parámetros."""
    ar = np.atleast_1d(raices_ar)
    ma = np.atleast_1d(raices_ma)
    if ar.size == 0 or ma.size == 0:
        return False
    return bool(np.min(np.abs(ar[:, None] - ma[None, :])) < distancia)


def requisitos_arima(ajuste, p, q, alfa=config.ALFA_PRUEBAS):
    """Evalúa los requisitos de Box-Jenkins sobre un ajuste. Devuelve (cumple, coeficientes, diagnóstico)."""
    coeficientes = coeficientes_arima(ajuste, alfa)
    diagnostico = diagnostico_residuos(ajuste.resid, p, q)
    p_ljung = min(prueba["p_valor"] for prueba in diagnostico["ljung_box"])
    cumple = {
        "estacionario": bool(np.all(np.abs(np.atleast_1d(ajuste.arroots)) > 1)),
        "invertible": bool(np.all(np.abs(np.atleast_1d(ajuste.maroots)) > 1)),
        "coef_significativos": all(c["significativo"] for c in coeficientes if c["parametro"].startswith(("ar.", "ma."))),
        "sin_autocorrelacion": bool(np.isfinite(p_ljung) and p_ljung >= alfa),
        "residuos_normales": bool(diagnostico["jarque_bera"]["p_valor"] >= alfa),
    }
    return cumple, coeficientes, diagnostico


def _raices(valores):
    return [[float(np.real(z)), float(np.imag(z))] for z in np.atleast_1d(valores)]


def arima_a_dict(ajuste, p, q, d=1):
    valores = dict(zip(ajuste.model.param_names, np.asarray(ajuste.params, dtype=float)))
    cumple, coeficientes, diagnostico = requisitos_arima(ajuste, p, q)
    return {
        "p": p, "d": d, "q": q, "nombre": f"ARIMA({p},{d},{q})", "escala": ESCALA_ARIMA,
        "constante": float(valores.get("const", 0.0)),
        "ar": [float(valores[f"ar.L{i}"]) for i in range(1, p + 1)],
        "ma": [float(valores[f"ma.L{j}"]) for j in range(1, q + 1)],
        "sigma2": float(valores["sigma2"]),
        "aic": float(ajuste.aic), "bic": float(ajuste.bic),
        "raices_ar": _raices(ajuste.arroots),
        "raices_ma": _raices(ajuste.maroots),
        "raices_casi_comunes": raices_casi_comunes(ajuste.arroots, ajuste.maroots),
        "coeficientes": coeficientes,
        "correlacion": correlacion_estimaciones(ajuste),
        "requisitos": cumple,
        "requisitos_cumplidos": int(sum(cumple.values())),
        "diagnostico": diagnostico,
    }


def buscar_arima(f, cortes, p_max=config.ARIMA_MAX_P, q_max=config.ARIMA_MAX_Q, al_terminar=None):
    """Ajusta ARMA(p, q) sobre el log-retorno (en %) del tramo de desarrollo y elige con los requisitos de Box-Jenkins.

    Gana el que cumple más requisitos (estacionario, invertible, coeficientes significativos, residuos sin
    autocorrelación y residuos normales); entre los que cumplen los mismos, el de menor AIC. Solo compiten los
    ajustes que convergieron, si hay alguno.
    Devuelve (parámetros del elegido, tabla de la búsqueda, residuos del elegido).
    """
    w = ESCALA_ARIMA * f["log_ret"].to_numpy(dtype=float)[: cortes.fin_validacion + 1]
    combinaciones = [(p, q) for p in range(p_max + 1) for q in range(q_max + 1)]
    filas, ajustes = [], {}
    for numero, (p, q) in enumerate(combinaciones, start=1):
        ajuste = ajustar_arma(w, p, q)
        retornos_mle = ajuste.mle_retvals if isinstance(ajuste.mle_retvals, dict) else {}
        cumple, coeficientes, diagnostico = requisitos_arima(ajuste, p, q)
        fila = {"p": p, "q": q, "aic": float(ajuste.aic), "bic": float(ajuste.bic),
                "log_verosimilitud": float(ajuste.llf), "convergio": bool(retornos_mle.get("converged", True)),
                **cumple, "requisitos": int(sum(cumple.values())),
                "ljung_box_p": min(prueba["p_valor"] for prueba in diagnostico["ljung_box"]),
                "jarque_bera_p": diagnostico["jarque_bera"]["p_valor"],
                "raices_casi_comunes": raices_casi_comunes(ajuste.arroots, ajuste.maroots)}
        filas.append(fila)
        ajustes[(p, q)] = ajuste
        if al_terminar:
            al_terminar(numero, len(combinaciones), fila)
    tabla = pd.DataFrame(filas)
    candidatas = tabla[tabla["convergio"]] if tabla["convergio"].any() else tabla
    elegida = candidatas.sort_values(["requisitos", "aic"], ascending=[False, True]).iloc[0]
    p, q = int(elegida["p"]), int(elegida["q"])
    tabla["elegido"] = (tabla["p"] == p) & (tabla["q"] == q)
    tabla = tabla.sort_values(["requisitos", "aic"], ascending=[False, True]).reset_index(drop=True)
    ajuste = ajustes[(p, q)]
    return arima_a_dict(ajuste, p, q), tabla, np.asarray(ajuste.resid, dtype=float)


def predecir_arima(parametros, retornos, posiciones, horizonte=1):
    """Log-retornos pronosticados por el ARIMA, con sus parámetros fijos (orígenes × horizonte).

    Cada origen j usa solo los retornos hasta j: los residuos pasados salen del filtro del ARMA y los futuros
    valen 0. Así el pronóstico a varias horas tiende a la media del log-retorno.
    """
    from scipy.signal import lfilter

    escala = parametros.get("escala", ESCALA_ARIMA)
    media = parametros["constante"]
    phi = np.asarray(parametros["ar"], dtype=float)
    theta = np.asarray(parametros["ma"], dtype=float)
    u = escala * np.asarray(retornos, dtype=float) - media
    e = lfilter(np.r_[1.0, -phi], np.r_[1.0, theta], u)
    p, q = len(phi), len(theta)
    salida = np.zeros((len(posiciones), horizonte))
    for fila, j in enumerate(posiciones):
        u_previos = list(u[max(0, j - p + 1): j + 1]) if p else []
        e_previos = list(e[max(0, j - q + 1): j + 1]) if q else []
        for h in range(horizonte):
            siguiente = sum(phi[i] * u_previos[-1 - i] for i in range(min(p, len(u_previos))))
            siguiente += sum(theta[k] * e_previos[-1 - k] for k in range(min(q, len(e_previos))))
            salida[fila, h] = siguiente
            u_previos.append(siguiente)
            e_previos.append(0.0)
    return (salida + media) / escala


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