"""Dashboard de predicción de acciones.  Ejecutar con:  python -m streamlit run app.py"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import agente
import config
import datos
import estrategias
import interfaz_agente
import interfaz_cuenta
import interfaz_etica
import modelos_redes
import modelos_series
import portafolio as portafolio_simulado
import prediccion
import preparacion
import varias_horas
import velas_cnn

st.set_page_config(page_title=config.NOMBRE_APP, page_icon=str(config.ICONO) if config.ICONO.exists() else None,
                   layout="wide")
if config.LOGO.exists():
    st.logo(str(config.LOGO), icon_image=str(config.ICONO) if config.ICONO.exists() else None)

INTERVALO = config.INTERVALO_MODELO
VIVO = config.INTERVALO_VIVO
COLOR_REAL = "#2a78d6"
COLOR_REFERENCIA = "#898781"
COLOR_SUBE = "#0ca30c"
COLOR_BAJA = "#d03b3b"
COLOR_ACENTO = "#eb6834"
COLORES_MODELO = {"AR": "#1baf7a", "ARIMA": "#8c564b", "RNN simple": "#e87ba4",
                  "LSTM": "#9085e9", "CNN de velas": "#eda100"}
COLORES_PLIEGUE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SALTOS_HORARIO = [dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")]


def familia_modelo(nombre):
    if nombre.startswith("ARIMA("):
        return "ARIMA"
    if nombre.startswith("AR("):
        return "AR"
    return nombre


def es_modelo(nombre):
    return familia_modelo(nombre) in COLORES_MODELO


def color_modelo(nombre):
    return COLORES_MODELO.get(familia_modelo(nombre), COLOR_REFERENCIA)


def a_hora_ny(marcas):
    return pd.DatetimeIndex(marcas).tz_convert(config.ZONA_MERCADO).tz_localize(None)


DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def hora_doble(marca, con_fecha=True):
    ny = marca.tz_convert(config.ZONA_MERCADO)
    bogota = marca.tz_convert(config.ZONA_USUARIO)
    fecha = f"{DIAS[ny.weekday()]} {ny:%d/%m} " if con_fecha else ""
    return f"{fecha}{ny:%H:%M} NY ({bogota:%H:%M} Bogotá)"


def estilo(fig, alto=380, titulo=None, saltos=True, hover="x unified"):
    fig.update_layout(
        height=alto,
        title=titulo,
        margin=dict(l=10, r=10, t=50 if titulo else 20, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        hovermode=hover,
    )
    if saltos:
        fig.update_xaxes(rangebreaks=SALTOS_HORARIO)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)")
    fig.update_xaxes(showgrid=False)
    return fig


# ------------------------------------------------------------------ datos y modelos en caché

@st.cache_data(ttl=config.SEGUNDOS_ACTUALIZACION - 10, show_spinner="Consultando Yahoo Finance…")
def obtener(ticker, intervalo):
    return datos.actualizar(ticker, intervalo)


@st.cache_resource(show_spinner="Cargando modelos entrenados…", max_entries=6)
def cargar_artefactos(ticker, intervalo, version):
    return prediccion.cargar_artefactos(ticker, intervalo)


def artefactos_de(ticker):
    ruta = config.directorio_modelos(ticker, INTERVALO) / "resumen.json"
    version = ruta.stat().st_mtime if ruta.exists() else None
    return cargar_artefactos(ticker, INTERVALO, version) if version else None


def completas_hasta(ticker, intervalo, ultima_iso):
    df = datos.velas_completas(datos.leer_local(ticker, intervalo), intervalo)
    return df[df.index <= pd.Timestamp(ultima_iso)]


@st.cache_data(show_spinner=False, max_entries=12)
def prediccion_siguiente(ticker, intervalo, ultima_iso, version, _arts):
    completas = completas_hasta(ticker, intervalo, ultima_iso)
    return prediccion.predecir_siguiente(completas, intervalo, _arts)


@st.cache_data(show_spinner="Simulando las estrategias…", max_entries=24)
def backtest_estrategias(ticker, intervalo, ultima_iso, costo_pct, version, _arts):
    completas = completas_hasta(ticker, intervalo, ultima_iso)
    predicciones = pd.read_csv(_arts["carpeta"] / "predicciones_prueba.csv") if _arts else None
    return estrategias.backtest(completas, predicciones, costo_pct)


@st.cache_data(show_spinner="La CNN está mirando las últimas velas…", max_entries=12)
def cnn_actual(ticker, intervalo, ultima_iso, version, _arts):
    """(probabilidad de que suba la próxima vela, imagen que mira la CNN) o (None, None)."""
    return prediccion.cnn_siguiente(completas_hasta(ticker, intervalo, ultima_iso), _arts)


@st.cache_data(show_spinner="Pronosticando varias horas…", max_entries=24)
def pronostico_varias_horas(ticker, intervalo, origen_iso, horas, version, _arts):
    completas = completas_hasta(ticker, intervalo, origen_iso)
    return varias_horas.pronosticar(completas, intervalo, horas, _arts)


@st.cache_data(show_spinner="Evaluando cada horizonte en las últimas sesiones (la primera vez tarda unos segundos)…",
               max_entries=12)
def evaluacion_varias_horas(ticker, intervalo, ultima_iso, horas, sesiones, version, _arts):
    completas = completas_hasta(ticker, intervalo, ultima_iso)
    f = preparacion.construir_caracteristicas(completas)
    ultimo_origen = len(f) - 1 - horas
    posiciones = list(range(max(ultimo_origen - sesiones * datos.velas_por_dia(intervalo) + 1, 60), ultimo_origen + 1))
    return varias_horas.errores_por_horizonte(completas, horas, posiciones, _arts)


@st.cache_data(show_spinner="Calculando pruebas de estacionariedad…", max_entries=6)
def analisis_serie(ticker, intervalo, ultima_iso, rezagos=config.REZAGOS_ACF):
    from statsmodels.tsa.stattools import acf, pacf

    log_precio = np.log(completas_hasta(ticker, intervalo, ultima_iso)["Close"].astype(float))
    retorno = log_precio.diff().dropna()
    d, pasos = modelos_series.orden_integracion(log_precio.to_numpy())
    return {
        "d": d,
        "pasos": pasos,
        "acf_log": acf(log_precio, nlags=rezagos)[1:],
        "pacf_log": pacf(log_precio, nlags=rezagos)[1:],
        "acf_ret": acf(retorno, nlags=rezagos)[1:],
        "pacf_ret": pacf(retorno, nlags=rezagos)[1:],
        "n": len(retorno),
    }


@st.cache_data(show_spinner=False, max_entries=6)
def residuos_arima(ruta, version, rezagos=config.REZAGOS_ACF):
    """Cuantiles teóricos y observados de los residuos estandarizados, y su ACF."""
    from scipy import stats
    from statsmodels.tsa.stattools import acf

    residuos = pd.read_csv(ruta)["residuo_pct"].to_numpy(dtype=float)
    estandarizados = np.sort((residuos - residuos.mean()) / residuos.std())
    teoricos = stats.norm.ppf((np.arange(1, len(estandarizados) + 1) - 0.5) / len(estandarizados))
    return teoricos, estandarizados, acf(residuos, nlags=rezagos)[1:], len(residuos)


@st.cache_data(show_spinner="Revisando cómo les fue a los modelos en la última sesión…", max_entries=6)
def evaluacion_ultima_sesion(ticker, intervalo, ultima_iso, version, _arts):
    completas = completas_hasta(ticker, intervalo, ultima_iso)
    previas = completas[completas.index < completas.index[-1].normalize()]
    if previas.empty:
        return pd.DataFrame()
    return prediccion.evaluar_ultima_sesion(completas, intervalo, _arts)


def tickers_con_modelos():
    return sorted(carpeta.name.rsplit("_", 1)[0] for carpeta in config.DIR_MODELOS.glob(f"*_{INTERVALO}")
                  if (carpeta / "resumen.json").exists())


def cotizacion_para(simbolo):
    df_2m, _ = obtener(simbolo, VIVO)
    return portafolio_simulado.cotizacion(df_2m)


def prediccion_para(simbolo):
    """(tabla de predicciones, info, consenso) de la próxima vela de 1 hora con los modelos de `simbolo`."""
    arts_simbolo = artefactos_de(simbolo)
    df_simbolo, _ = obtener(simbolo, INTERVALO)
    if not arts_simbolo or df_simbolo.empty:
        return pd.DataFrame(), None, None
    ultima = datos.velas_completas(df_simbolo, INTERVALO).index[-1].isoformat()
    tabla, info = prediccion_siguiente(simbolo, INTERVALO, ultima, arts_simbolo["resumen"]["entrenado_en"],
                                       arts_simbolo)
    return tabla, info, prediccion.consenso(tabla)


def series_para(simbolo):
    return [obtener(simbolo, VIVO)[0], obtener(simbolo, INTERVALO)[0]]


# ------------------------------------------------------------------ barra lateral

st.sidebar.header("Acción")
opciones = tickers_con_modelos() or [config.TICKER_POR_DEFECTO]
ticker = st.sidebar.selectbox(
    "Símbolo", opciones,
    index=opciones.index(config.TICKER_POR_DEFECTO) if config.TICKER_POR_DEFECTO in opciones else 0,
    help="Solo aparecen las acciones con modelos entrenados. Una acción nueva aparece cuando se entrena con "
         "entrenar.py --ticker SÍMBOLO.",
)

en_vivo = st.sidebar.toggle("Actualizar en vivo", value=True,
                            help=f"Consulta Yahoo Finance cada {config.SEGUNDOS_ACTUALIZACION // 60} minutos.")
if st.sidebar.button("Actualizar ahora"):
    obtener.clear()
    st.rerun()

arts = artefactos_de(ticker)
if arts:
    st.sidebar.success(f"Modelos entrenados el {arts['resumen']['entrenado_en']}")
else:
    st.sidebar.warning("Esta acción no tiene modelos entrenados. Por ahora solo se muestran los precios.")
    st.sidebar.code(f"python entrenar.py --ticker {ticker}", language="bash")

usuario = interfaz_cuenta.sesion()
interfaz_cuenta.barra_lateral(usuario)

# ------------------------------------------------------------------ encabezado

ahora = datos.ahora_mercado()
estado = "Mercado abierto" if datos.mercado_abierto(ahora) else "Mercado cerrado"
st.title(ticker)
st.caption(f"{estado} · {ahora:%H:%M} en Nueva York · {ahora.tz_convert(config.ZONA_USUARIO):%H:%M} en Bogotá · "
           "la bolsa de EE. UU. opera de 9:30 a 16:00, hora de Nueva York")

df_hora, error_hora = obtener(ticker, INTERVALO)
if df_hora.empty:
    st.error(error_hora or f"No hay datos de {ticker}.")
    st.stop()
if error_hora:
    st.warning(error_hora)
ultima_iso = datos.velas_completas(df_hora, INTERVALO).index[-1].isoformat()

(tab_vivo, tab_horas, tab_estrategias, tab_portafolio, tab_agente, tab_serie, tab_eval, tab_datos,
 tab_mis_datos, tab_etica) = st.tabs(
    ["En vivo y próxima hora", "Varias horas", "Estrategias", "Portafolio simulado", "Agente",
     "Análisis de la serie", "Evaluación de modelos", "Datos recolectados", "Mis datos", "Ética"]
)


# ------------------------------------------------------------------ pestaña 1: en vivo

@st.fragment(run_every=config.SEGUNDOS_ACTUALIZACION if en_vivo else None)
def panel_vivo(ticker, arts):
    df_2m, error_2m = obtener(ticker, VIVO)
    df_1h, error_1h = obtener(ticker, INTERVALO)
    for error in (error_2m, error_1h):
        if error:
            st.warning(error)
    completas = datos.velas_completas(df_1h, INTERVALO)
    sesion = datos.ultima_sesion(df_2m)
    momento = datos.ahora_mercado()

    tabla, info = prediccion.predecir_siguiente(completas, INTERVALO, arts)
    voto = prediccion.consenso(tabla)

    precio_actual = float(sesion["Close"].iloc[-1]) if not sesion.empty else float(df_1h["Close"].iloc[-1])
    previas = df_1h[df_1h.index < (sesion.index[0].normalize() if not sesion.empty else df_1h.index[-1].normalize())]
    col1, col2, col3, col4 = st.columns(4)
    if previas.empty:
        col1.metric("Último precio", f"{precio_actual:,.2f}")
    else:
        cierre_previo = float(previas["Close"].iloc[-1])
        col1.metric("Último precio", f"{precio_actual:,.2f}",
                    f"{100 * (precio_actual / cierre_previo - 1):+.2f}% vs cierre anterior")
    if not sesion.empty:
        ultima_vela = sesion.index[-1]
        col2.metric("Último dato de 2 min", f"{ultima_vela.tz_convert(config.ZONA_MERCADO):%H:%M} NY")
        col2.caption(f"{ultima_vela.tz_convert(config.ZONA_USUARIO):%H:%M} en Bogotá · "
                     f"{ultima_vela.tz_convert(config.ZONA_MERCADO):%d/%m/%Y}")
    col3.metric("Velas de 2 min recolectadas", f"{len(df_2m):,}")
    col3.caption(f"{len(df_1h):,} velas de 1 hora")
    if voto:
        flecha = {"Sube": "▲", "Baja": "▼", "Empate": "■"}[voto["direccion"]]
        col4.metric("Consenso de los modelos", f"{flecha} {voto['direccion']}")
        col4.caption(f"{voto['suben']} {'sube' if voto['suben'] == 1 else 'suben'} · "
                     f"{voto['bajan']} {'baja' if voto['bajan'] == 1 else 'bajan'}")

    st.subheader("Predicción de la próxima hora")
    fin_objetivo = datos.fin_de_vela(info["objetivo"], INTERVALO)
    st.markdown(
        f"Cada modelo predice si la vela de **{hora_doble(info['objetivo'])}** cerrará por encima o por debajo "
        f"del cierre de la última vela completa (**{hora_doble(info['referencia'])}**, "
        f"precio **{info['cierre_referencia']:,.2f}**)."
    )
    columnas = st.columns(max(len(tabla), 1))
    for columna, (_, fila) in zip(columnas, tabla.iterrows()):
        flecha = "▲" if fila["direccion"] == "Sube" else "▼"
        columna.metric(fila["modelo"], f"{fila['precio_predicho']:,.2f}", f"{fila['retorno_pct']:+.3f}%")
        columna.caption(f"{flecha} {fila['direccion']}")
    if info["objetivo"] <= momento < fin_objetivo and not sesion.empty:
        cambio = 100 * (precio_actual / info["cierre_referencia"] - 1)
        st.info(f"La vela predicha está en curso y cierra a las {hora_doble(fin_objetivo, con_fecha=False)}. "
                f"Por ahora el precio va **{cambio:+.2f}%** respecto al cierre de referencia.")

    if not sesion.empty:
        x = a_hora_ny(sesion.index)
        fig = go.Figure(go.Candlestick(
            x=x, open=sesion["Open"], high=sesion["High"], low=sesion["Low"], close=sesion["Close"],
            name="Velas de 2 min", increasing_line_color=COLOR_SUBE, decreasing_line_color=COLOR_BAJA,
        ))
        prediccion_en_sesion = info["objetivo"].normalize() == sesion.index[-1].normalize()
        if prediccion_en_sesion:
            tramo = list(a_hora_ny([info["objetivo"], fin_objetivo]))
            fig.add_vrect(x0=tramo[0], x1=tramo[1], fillcolor="rgba(128,128,128,0.12)", line_width=0,
                          annotation_text="Hora predicha", annotation_position="top left")
            fig.add_trace(go.Scatter(x=tramo, y=[info["cierre_referencia"]] * 2, mode="lines", name="Cierre de referencia",
                                     line=dict(color=COLOR_REFERENCIA, width=2, dash="dash")))
            for _, fila in tabla.iterrows():
                fig.add_trace(go.Scatter(x=tramo, y=[fila["precio_predicho"]] * 2, mode="lines", name=fila["modelo"],
                                         line=dict(color=color_modelo(fila["modelo"]), width=2, dash="dot")))
        fecha_sesion = sesion.index[0].tz_convert(config.ZONA_MERCADO).strftime("%d/%m/%Y")
        estilo(fig, alto=430, titulo=f"Sesión del {fecha_sesion} (velas de 2 min, hora de Nueva York)", saltos=False)
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, key="grafico_vivo")
        if not prediccion_en_sesion:
            st.caption(f"La próxima vela a predecir es la de {hora_doble(info['objetivo'])}, en la siguiente sesión: "
                       "sus predicciones aparecen en el gráfico por hora de abajo. Cuando abra el mercado, esta vista "
                       "marcará la hora predicha y el precio que espera cada modelo.")

    if arts:
        version = arts["resumen"]["entrenado_en"]
        revision = evaluacion_ultima_sesion(ticker, INTERVALO, completas.index[-1].isoformat(), version, arts)
        if not revision.empty:
            columnas_modelos = [c for c in revision.columns if es_modelo(c)]
            dia_revision = revision["vela"].iloc[0].tz_convert(config.ZONA_MERCADO)
            st.markdown(f"**¿Cómo les fue a los modelos en la sesión del {DIAS[dia_revision.weekday()]} "
                        f"{dia_revision:%d/%m}?**")
            vista = pd.DataFrame({
                "vela (NY)": [f"{v.tz_convert(config.ZONA_MERCADO):%H:%M}" for v in revision["vela"]],
                "Bogotá": [f"{v.tz_convert(config.ZONA_USUARIO):%H:%M}" for v in revision["vela"]],
                "cierre anterior": revision["cierre_anterior"].round(2),
                "cierre real": revision["cierre_real"].round(2),
                "real": [f"{'▲' if r == 'Sube' else '▼'} {c:+.2f}%" for r, c in zip(revision["real"], revision["cambio_real_pct"])],
            })
            aciertos = []
            for modelo in columnas_modelos:
                correcto = revision[modelo] == revision["real"]
                vista[modelo] = [f"{'▲' if p == 'Sube' else '▼'} {'acierta' if ok else 'falla'}"
                                 for p, ok in zip(revision[modelo], correcto)]
                aciertos.append(f"{modelo} {int(correcto.sum())}/{len(correcto)}")
            st.dataframe(vista, hide_index=True)
            st.caption("Cada fila es lo que el modelo habría predicho para esa hora usando solo los datos hasta la hora "
                       f"anterior. Aciertos: {' · '.join(aciertos)}. Con tan pocas velas el resultado de un solo día "
                       "depende mucho del azar; la medida confiable es la pestaña Evaluación de modelos.")

    ultimas = completas.iloc[-5 * datos.velas_por_dia(INTERVALO):]
    fig_h = go.Figure(go.Scatter(x=a_hora_ny(ultimas.index), y=ultimas["Close"], mode="lines+markers",
                                 name="Cierre por hora", line=dict(color=COLOR_REAL, width=2), marker=dict(size=6)))
    x_ref, x_obj = a_hora_ny([info["referencia"]])[0], a_hora_ny([info["objetivo"]])[0]
    for _, fila in tabla.iterrows():
        fig_h.add_trace(go.Scatter(
            x=[x_ref, x_obj], y=[info["cierre_referencia"], fila["precio_predicho"]], mode="lines+markers",
            name=fila["modelo"], line=dict(color=color_modelo(fila["modelo"]), width=2, dash="dot"),
            marker=dict(size=[0, 10], symbol="diamond"),
        ))
    st.plotly_chart(estilo(fig_h, alto=380, titulo="Últimas 5 sesiones por hora y predicción de la siguiente vela"),
                    key="grafico_horas")

    with st.expander("Resumen de texto (servirá de entrada para el agente de Gemini)"):
        metricas = pd.read_csv(arts["carpeta"] / "metricas.csv") if arts else None
        st.code(prediccion.resumen_para_agente(ticker, completas, tabla, info, metricas), language="text")

    estado_vivo = f"se actualiza cada {config.SEGUNDOS_ACTUALIZACION // 60} min" if en_vivo else "actualización en vivo apagada"
    st.caption(f"Última consulta a Yahoo: {momento.tz_convert(config.ZONA_USUARIO):%H:%M:%S} (Bogotá) · {estado_vivo}. "
               "Yahoo entrega velas ya formadas; la vela más reciente puede estar incompleta y se excluye de las predicciones.")


with tab_vivo:
    panel_vivo(ticker, arts)


# ------------------------------------------------------------------ pestaña: varias horas

def hora_corta(marca):
    ny = marca.tz_convert(config.ZONA_MERCADO)
    return f"{DIAS[ny.weekday()]} {ny:%d/%m %H:%M}"


def rezagos_modelo(nombre, arts):
    """Cuántos log-retornos pasados usa cada modelo recursivo para predecir la hora siguiente."""
    resumen = arts["resumen"]
    if nombre.startswith("ARIMA("):
        return resumen["arima"]["p"]
    if nombre.startswith("AR("):
        return resumen["ar"]["p"]
    for tipo, nombre_tipo in modelos_redes.TIPOS.items():
        if nombre == nombre_tipo:
            return resumen["redes"][tipo]["ventana"]
    return 0


def tabla_paso_a_paso(nombre, futuro, cierre_origen, historia_r, arts):
    """Cada hora futura con la ventana que entró al modelo, el log-retorno predicho y el precio resultante."""
    k = rezagos_modelo(nombre, arts)
    precios = futuro[nombre].to_numpy(dtype=float)
    predichos = np.diff(np.log(np.r_[cierre_origen, precios]))
    secuencia = list(historia_r) + list(predichos)
    n_real = len(historia_r)
    filas = []
    for h in range(len(predichos)):
        fin = n_real + h
        indices = list(range(fin - 1, max(fin - k, 0) - 1, -1)) if k else []
        entrada = ", ".join(f"[{100 * secuencia[i]:.3f}]" if i >= n_real else f"{100 * secuencia[i]:.3f}"
                            for i in indices)
        filas.append({
            "paso": h + 1,
            "hora que predice": hora_corta(futuro.index[h]),
            (f"entrada: últimos {k} log-retornos (%)" if k else "entrada"): entrada or "ninguno: solo la media μ",
            "log-retorno predicho (%)": round(100 * float(predichos[h]), 4),
            "precio predicho": round(float(precios[h]), 2),
        })
    return pd.DataFrame(filas)


with tab_horas:
    st.markdown(
        "Pronóstico **hora por hora** con el modelo ganador de cada tipo. AR, ARIMA, RNN y LSTM predicen la hora "
        "siguiente con su ventana de log-retornos; después esa predicción entra a la ventana como si fuera un dato "
        "real y se predice la hora que sigue, y así sucesivamente (**pronóstico recursivo**). El precio de cada hora "
        "es el de la hora anterior multiplicado por e elevado al log-retorno predicho."
    )
    st.latex(r"\hat r_{t+1} = f(r_t, r_{t-1}, \dots) \qquad \hat r_{t+2} = f(\hat r_{t+1}, r_t, \dots) \qquad "
             r"\hat r_{t+3} = f(\hat r_{t+2}, \hat r_{t+1}, r_t, \dots) \qquad \hat P_{t+h} = \hat P_{t+h-1}\, e^{\hat r_{t+h}}")
    control_1, control_2 = st.columns([1, 2])
    horas = control_1.slider("Horas de mercado hacia adelante", min_value=1, max_value=35, value=7,
                             help="Cada paso es una vela de 1 hora. Una sesión tiene 7: 14 son 2 sesiones y 35, una "
                                  "semana de mercado.")
    modo = control_2.radio("Pronosticar desde", ["La última vela (hacia el futuro)",
                                                 "Un momento pasado (para comparar con lo que pasó)"])
    completas_tab = completas_hasta(ticker, INTERVALO, ultima_iso)
    if modo.startswith("Un momento"):
        candidatas = completas_tab.index[-10 * datos.velas_por_dia(INTERVALO):-1]
        opciones_origen = {hora_corta(m): m for m in candidatas}
        defecto = candidatas[max(0, len(candidatas) - horas)]
        elegido = st.select_slider("Origen del pronóstico (hora de Nueva York)", options=list(opciones_origen),
                                   value=hora_corta(defecto))
        origen = opciones_origen[elegido]
    else:
        origen = completas_tab.index[-1]

    version_modelos = arts["resumen"]["entrenado_en"] if arts else None
    futuro, info_h = pronostico_varias_horas(ticker, INTERVALO, origen.isoformat(), horas, version_modelos, arts)
    modelos_h = [c for c in futuro.columns if not c.startswith("banda")]
    previas_h = completas_tab[completas_tab.index <= origen].iloc[-3 * datos.velas_por_dia(INTERVALO):]
    reales_h = completas_tab[completas_tab.index > origen].iloc[:horas]
    x_futuro = a_hora_ny(futuro.index)
    x_origen = a_hora_ny([origen])[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_futuro, y=futuro["banda_alta"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_futuro, y=futuro["banda_baja"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(137,135,129,0.18)", name="80% de una caminata aleatoria"))
    reales_todos = pd.concat([previas_h, reales_h])
    fig.add_trace(go.Scatter(x=a_hora_ny(reales_todos.index), y=reales_todos["Close"], mode="lines+markers",
                             name="Precio real", line=dict(color=COLOR_REAL, width=2), marker=dict(size=5)))
    for nombre in modelos_h:
        fig.add_trace(go.Scatter(x=[x_origen, *x_futuro], y=[info_h["cierre_origen"], *futuro[nombre]],
                                 mode="lines+markers", name=nombre, marker=dict(size=5),
                                 line=dict(color=color_modelo(nombre), width=2, dash="dot")))
    fig.add_vline(x=x_origen, line_dash="dash", line_color=COLOR_REFERENCIA)
    titulo_h = (f"Pronóstico hora por hora: {horas} {'hora' if horas == 1 else 'horas'} desde {hora_doble(origen)}"
                + (" y lo que pasó en realidad" if not reales_h.empty else ""))
    fig.update_xaxes(tickformat="%d/%m %H:%M")
    st.plotly_chart(estilo(fig, alto=460, titulo=titulo_h), key="grafico_varias_horas")
    st.caption(f"La franja gris es el rango donde caería el precio el 80% de las veces si se moviera al azar, con la "
               f"volatilidad de las últimas 500 horas ({100 * info_h['sigma']:.2f}% por hora): crece como √horas. "
               "Un pronóstico que no se sale de esa franja no dice mucho más que el azar.")

    pasos = sorted({1, min(7, horas), horas})
    filas_h = []
    for nombre in modelos_h:
        fila = {"modelo": nombre}
        for paso in pasos:
            fila[f"a {paso} h"] = round(float(futuro[nombre].iloc[paso - 1]), 2)
        cambio = 100 * (futuro[nombre].iloc[-1] / info_h["cierre_origen"] - 1)
        fila["cambio total"] = f"{'▲' if cambio > 0 else '▼'} {cambio:+.2f}%"
        if not reales_h.empty:
            n_real = len(reales_h)
            predicho = futuro[nombre].iloc[:n_real].to_numpy()
            real = reales_h["Close"].to_numpy()
            fila[f"real a {n_real} h"] = round(float(real[-1]), 2)
            fila["error medio"] = f"{100 * np.mean(np.abs(predicho - real) / real):.2f}%"
            acierta = (predicho[-1] > info_h["cierre_origen"]) == (real[-1] > info_h["cierre_origen"])
            fila["dirección"] = "acierta" if acierta else "falla"
        filas_h.append(fila)
    st.dataframe(pd.DataFrame(filas_h), hide_index=True)
    if not reales_h.empty and len(reales_h) < horas:
        st.caption(f"Solo hay {len(reales_h)} horas reales después del origen; la comparación usa esas.")

    recursivos = modelos_h
    if recursivos and arts:
        st.markdown("**Paso a paso del pronóstico recursivo**")
        modelo_paso = st.selectbox("Modelo", recursivos, key="modelo_paso_a_paso")
        historia_r = np.log(completas_tab.loc[completas_tab.index <= origen, "Close"].astype(float)).diff().dropna()
        st.dataframe(tabla_paso_a_paso(modelo_paso, futuro, info_h["cierre_origen"], historia_r.to_numpy(), arts),
                     hide_index=True)
        nota_arima = (" El ARIMA usa además sus últimos residuos en la parte MA; los residuos de las horas futuras "
                      "valen 0, porque todavía no se conocen." if modelo_paso.startswith("ARIMA(") else "")
        st.caption("La entrada va de la hora más reciente a la más antigua. Los valores entre corchetes son "
                   "predicciones del propio modelo que ya entraron a la ventana; los demás son log-retornos reales "
                   f"hasta el origen.{nota_arima}")

    st.markdown("**¿Qué tan rápido pierde precisión cada modelo al alejarse?**")
    if st.checkbox("Calcular el error promedio por horizonte en las últimas 10 sesiones"):
        error_h, acierto_h, n_origenes = evaluacion_varias_horas(ticker, INTERVALO, ultima_iso, horas, 10,
                                                                 version_modelos, arts)
        medida = st.radio("Medida", ["Error medio del precio (%)", "Acierto de dirección (%)"], horizontal=True)
        tabla_medida = error_h if medida.startswith("Error") else acierto_h
        fig_e = go.Figure()
        for nombre in tabla_medida.columns:
            base = nombre.startswith("Ingenuo") or nombre == "Siempre sube"
            fig_e.add_trace(go.Scatter(x=tabla_medida.index, y=tabla_medida[nombre], mode="lines+markers",
                                       name=nombre, marker=dict(size=5),
                                       line=dict(color=COLOR_REFERENCIA if base else color_modelo(nombre), width=2,
                                                 dash="dash" if base else "solid")))
        if not medida.startswith("Error"):
            fig_e.add_hline(y=50, line_dash="dot", line_color=COLOR_REFERENCIA)
        fig_e.update_xaxes(title="Horas de mercado hacia adelante")
        fig_e.update_yaxes(ticksuffix="%")
        st.plotly_chart(estilo(fig_e, alto=380, titulo=f"{medida} según el horizonte · {n_origenes} orígenes",
                               saltos=False), key="grafico_error_horizonte")
        st.caption("Cada origen es una hora de las últimas 10 sesiones; todos los modelos predicen desde los mismos "
                   "orígenes y solo con datos anteriores a cada uno. En gris, las referencias: el precio sin cambio y "
                   "'siempre sube'. Un modelo aporta si su línea queda por debajo del ingenuo (error) o por encima de "
                   "'siempre sube' (acierto).")


# ------------------------------------------------------------------ pestaña: estrategias

COLORES_ESTRATEGIA = {
    estrategias.NOMBRE_BH: COLOR_REFERENCIA,
    estrategias.NOMBRE_VELAS: "#e34948",
    estrategias.NOMBRE_CNN: "#eda100",
}


def color_estrategia(nombre):
    return COLORES_ESTRATEGIA.get(nombre, COLOR_REFERENCIA)


def posicion_texto(posicion):
    return "▲ Comprado" if posicion == 1 else "▼ Fuera del mercado"


with tab_estrategias:
    st.markdown(
        "Cada estrategia decide al cierre de cada hora si queda **comprada** o **fuera del mercado** durante la hora "
        "siguiente, usando solo la información disponible hasta ese momento. No se usan ventas en corto."
    )
    with st.expander("Cómo decide cada estrategia"):
        st.markdown(
            f"- **{estrategias.NOMBRE_BH}:** compra al inicio y no vende. Es la referencia que las demás deberían "
            "superar.\n"
            f"- **{estrategias.NOMBRE_VELAS}:** mira el color de las últimas {config.VELAS_SEGUIDAS} velas de 1 hora. "
            f"Si las {config.VELAS_SEGUIDAS} son rojas (cada una cerró por debajo de su apertura), el precio viene "
            f"bajando y compra esperando un rebote. Si las {config.VELAS_SEGUIDAS} son verdes, vende. En cualquier "
            "otro caso mantiene lo que venía haciendo.\n"
            f"- **{estrategias.NOMBRE_CNN}:** una red convolucional mira una imagen de las últimas velas, como la de "
            "abajo, y da la probabilidad de que la hora siguiente suba. Queda comprada cuando esa probabilidad supera "
            f"el {100 * config.CNN_UMBRAL:.0f}%."
        )
    costo_pct = st.slider("Costo por cada compra o venta (%)", min_value=0.0, max_value=0.3,
                          value=config.COSTO_OPERACION_PCT, step=0.01,
                          help="Comisión más la diferencia entre precio de compra y de venta (spread).")

    version_est = arts["resumen"]["entrenado_en"] if arts else None
    completas_est = completas_hasta(ticker, INTERVALO, ultima_iso)
    probabilidad_cnn, foto_cnn = cnn_actual(ticker, INTERVALO, ultima_iso, version_est, arts)
    senales = estrategias.senales_actuales(completas_est, probabilidad_cnn)
    siguiente = datos.siguiente_vela(completas_est.index[-1], INTERVALO)
    st.markdown(f"**Posición de cada estrategia para la vela de {hora_doble(siguiente)}**")
    st.dataframe(pd.DataFrame({
        "estrategia": senales["estrategia"],
        "posición": [posicion_texto(p) for p in senales["posicion"]],
        "por qué": senales["motivo"],
    }), hide_index=True)
    st.caption("▲ Comprado = comprar o mantener la compra. ▼ Fuera del mercado = vender o seguir sin comprar.")

    if foto_cnn is not None:
        velas_foto = arts["resumen"]["cnn"]["velas"]
        st.markdown(f"**Lo que mira la CNN ahora:** las últimas {velas_foto} velas de 1 hora, hasta la de "
                    f"{hora_doble(completas_est.index[-1])}")
        st.image(velas_cnn.ampliar(foto_cnn, factor=6),
                 caption=f"Probabilidad de que la próxima hora suba: {100 * probabilidad_cnn:.0f}%")
        st.caption("Cada vela ocupa unos pocos píxeles: verde si cerró por encima de su apertura y roja si cerró por "
                   "debajo; el cuerpo va de la apertura al cierre y la línea delgada del mínimo al máximo. La altura de "
                   "la imagen va del mínimo al máximo de esas velas, así que la red ve la forma del movimiento y no el "
                   "nivel del precio.")
    elif arts:
        st.info("La CNN de velas todavía no está entrenada. Se entrena con `python entrenar.py`.")

    curvas, tabla_est = backtest_estrategias(ticker, INTERVALO, ultima_iso, costo_pct, version_est, arts)
    fig = go.Figure()
    for nombre in curvas.columns:
        es_referencia = nombre == estrategias.NOMBRE_BH
        fig.add_trace(go.Scatter(x=a_hora_ny(curvas.index), y=100 * curvas[nombre], mode="lines", name=nombre,
                                 line=dict(color=color_estrategia(nombre), width=2,
                                           dash="dash" if es_referencia else "solid")))
    fig.add_hline(y=100, line_dash="dot", line_color=COLOR_REFERENCIA)
    fig.update_yaxes(tickprefix="$")
    periodo = f"{curvas.index[0]:%d/%m/%Y} a {curvas.index[-1]:%d/%m/%Y}"
    origen_periodo = "periodo de prueba del entrenamiento" if arts else f"últimas {len(curvas)} horas"
    st.plotly_chart(estilo(fig, alto=430, titulo=f"Valor de 100 dólares invertidos · {origen_periodo}, {periodo}"),
                    key="grafico_estrategias")

    vista_est = pd.DataFrame({
        "estrategia": tabla_est["estrategia"],
        "retorno total": tabla_est["retorno_total_pct"].map(lambda v: f"{v:+.2f}%"),
        "frente a comprar y mantener": tabla_est["vs_comprar_y_mantener_pct"].map(lambda v: f"{v:+.2f} pts"),
        "máxima caída": tabla_est["maxima_caida_pct"].map(lambda v: f"{v:.2f}%"),
        "operaciones": tabla_est["operaciones"],
        "operaciones ganadoras": tabla_est["operaciones_ganadoras_pct"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}%"),
        "tiempo invertido": tabla_est["tiempo_invertido_pct"].map(lambda v: f"{v:.0f}%"),
        "Sharpe anual": tabla_est["sharpe_anual"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
    })
    st.dataframe(vista_est, hide_index=True)
    st.caption(
        "Máxima caída: la peor pérdida desde un máximo previo del capital. Sharpe: retorno promedio por unidad de "
        "riesgo, llevado a un año; por encima de 1 se considera bueno. La CNN usa sus predicciones del periodo de "
        "prueba, hechas con imágenes que no vio al entrenar. Que una estrategia haya funcionado en este periodo no "
        "garantiza que funcione después."
    )


# ------------------------------------------------------------------ pestaña: agente

with tab_portafolio:
    interfaz_cuenta.pestana_portafolio(usuario, tickers_con_modelos(), cotizacion_para, prediccion_para, series_para)


def clave_gemini():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def contexto_agente():
    version_ag = arts["resumen"]["entrenado_en"] if arts else None
    tabla_ag, info_ag = prediccion_siguiente(ticker, INTERVALO, ultima_iso, version_ag, arts)
    futuro_ag, info_futuro_ag = pronostico_varias_horas(ticker, INTERVALO, ultima_iso, datos.velas_por_dia(INTERVALO),
                                                        version_ag, arts)
    completas_ag = completas_hasta(ticker, INTERVALO, ultima_iso)
    _, tabla_est_ag = backtest_estrategias(ticker, INTERVALO, ultima_iso, config.COSTO_OPERACION_PCT, version_ag, arts)
    probabilidad_ag, _ = cnn_actual(ticker, INTERVALO, ultima_iso, version_ag, arts)
    return agente.construir_contexto(
        ticker, completas_ag, tabla_ag, info_ag,
        metricas_modelos=pd.read_csv(arts["carpeta"] / "metricas.csv") if arts else None,
        futuro=futuro_ag, info_futuro=info_futuro_ag,
        senales=estrategias.senales_actuales(completas_ag, probabilidad_ag),
        resultados_estrategias=tabla_est_ag, costo_pct=config.COSTO_OPERACION_PCT,
    )


with tab_agente:
    interfaz_agente.pestana_agente(usuario, ticker, clave_gemini(), contexto_agente)


# ------------------------------------------------------------------ pestaña 2: análisis de la serie

VELAS_SESION = datos.velas_por_dia(INTERVALO)


def grafico_correlacion(valores, n, titulo, clave, columna):
    """Barras de una ACF o PACF con la franja del 95%; los rezagos múltiplos de una sesión van en otro color."""
    rezagos = np.arange(1, len(valores) + 1)
    colores = [COLOR_ACENTO if k % VELAS_SESION == 0 else COLOR_REAL for k in rezagos]
    fig_c = go.Figure(go.Bar(x=rezagos, y=valores, marker_color=colores))
    banda = 1.96 / np.sqrt(n)
    for signo in (1, -1):
        fig_c.add_hline(y=signo * banda, line_dash="dash", line_color=COLOR_REFERENCIA)
    fig_c.update_xaxes(title="Rezago (horas)")
    fig_c.update_layout(showlegend=False)
    columna.plotly_chart(estilo(fig_c, alto=300, titulo=titulo, saltos=False, hover="closest"), key=clave)


def texto_p_valor(p_valor):
    return "< 0.0001" if p_valor < 1e-4 else f"{p_valor:.4f}"


def _con_signo(valor):
    return f"- {abs(valor):.4f}" if valor < 0 else f"+ {valor:.4f}"


REQUISITOS_ARIMA_TEXTO = [
    ("Estacionario", "Raíces del polinomio AR fuera del círculo unitario",
     "El ARMA describe una serie estacionaria: con una raíz dentro o sobre el círculo, el efecto de un choque no se "
     "desvanece y la varianza del pronóstico crece sin límite; además indicaría que faltaba diferenciar."),
    ("Invertible", "Raíces del polinomio MA fuera del círculo unitario",
     "El modelo se puede escribir como un AR infinito sobre el pasado observado: los residuos se recuperan de los datos "
     "y la representación es única (cada MA no invertible tiene uno invertible con la misma ACF)."),
    ("Coeficientes significativos", "Prueba z de cada φ y θ al 5% (H₀: vale 0), con errores estándar robustos",
     "Parsimonia: un coeficiente que no se distingue de 0 no aporta, solo agrega varianza al pronóstico y es señal de "
     "sobreajuste; conviene bajar el orden."),
    ("Residuos sin autocorrelación", "Ljung-Box a " + " y ".join(str(k) for k in config.LJUNG_BOX_REZAGOS)
     + " rezagos (H₀: no hay autocorrelación)",
     "Si queda autocorrelación, el pasado todavía tiene información que el modelo no usó: está mal especificado. Los "
     "residuos deben ser ruido blanco."),
    ("Residuos normales", "Jarque-Bera (H₀: asimetría 0 y curtosis 3)",
     "La verosimilitud, las pruebas de los coeficientes y los intervalos de pronóstico suponen errores normales; si "
     "falla, el pronóstico puntual sigue siendo válido, pero los intervalos y los p-valores dejan de ser exactos."),
]


def nombre_parametro(nombre):
    if nombre == "const":
        return "μ (media)"
    if nombre.startswith("ar.L"):
        return f"φ{nombre[4:]}"
    if nombre.startswith("ma.L"):
        return f"θ{nombre[4:]}"
    return nombre


def ecuacion_arima(arima):
    """El ARMA elegido, escrito sobre el log-retorno en %."""
    media = arima["constante"]
    texto = rf"r_t = {media:.4f}"
    for i, phi in enumerate(arima["ar"], start=1):
        texto += rf" {_con_signo(phi)}\,(r_{{t-{i}}} {_con_signo(-media)})"
    texto += r" + \varepsilon_t"
    for j, theta in enumerate(arima["ma"], start=1):
        texto += rf" {_con_signo(theta)}\,\varepsilon_{{t-{j}}}"
    return texto


with tab_serie:
    st.markdown("Metodología de **Box-Jenkins**: (1) mirar la serie original, (2) diferenciarla hasta que sea "
                "estacionaria según la prueba de Dickey-Fuller aumentada, (3) ajustar ARMA(p, q) a la serie "
                "estacionaria y elegir p y q por AIC, y (4) revisar que los residuos se comporten como ruido blanco.")
    completas = completas_hasta(ticker, INTERVALO, ultima_iso)
    x_serie = a_hora_ny(completas.index)
    analisis = analisis_serie(ticker, INTERVALO, ultima_iso)

    st.subheader("1. Serie original")
    st.caption(f"Toda la historia guardada: {len(completas):,} velas de 1 hora, desde {hora_doble(completas.index[0])} "
               f"hasta {hora_doble(completas.index[-1])}.")
    c1, c2 = st.columns(2)
    fig_p = go.Figure(go.Scatter(x=x_serie, y=completas["Close"], mode="lines", name="Cierre",
                                 line=dict(color=COLOR_REAL, width=1.5)))
    c1.plotly_chart(estilo(fig_p, alto=320, titulo="Precio de cierre por hora"), key="serie_precio")
    fig_l = go.Figure(go.Scatter(x=x_serie, y=np.log(completas["Close"].astype(float)), mode="lines", name="ln(precio)",
                                 line=dict(color=COLOR_REAL, width=1.5)))
    c2.plotly_chart(estilo(fig_l, alto=320, titulo="ln(precio): la serie original del ARIMA"), key="serie_log")
    c3, c4 = st.columns(2)
    grafico_correlacion(analisis["acf_log"], analisis["n"], "ACF de ln(precio)", "serie_acf_log", c3)
    grafico_correlacion(analisis["pacf_log"], analisis["n"], "PACF de ln(precio)", "serie_pacf_log", c4)
    st.caption("La ACF de ln(precio) empieza cerca de 1 y baja muy despacio, y la PACF solo tiene el primer rezago: es "
               "la huella de una raíz unitaria, una serie que no vuelve a una media fija. El logaritmo no diferencia la "
               "serie: mide los cambios en proporción al precio y estabiliza la varianza.")

    st.subheader("2. Diferenciación hasta que la serie sea estacionaria")
    st.latex(r"r_t = \nabla \ln P_t = \ln P_t - \ln P_{t-1} \qquad \Longleftrightarrow \qquad \ln P_t = \ln P_{t-1} + r_t")
    st.markdown("El log-retorno **es** la primera diferencia de ln(precio). Volver al precio es *integrar*: "
                r"$\hat P_{t+1} = P_t \, e^{\hat r_{t+1}}$. Por eso un ARMA(p, q) sobre el log-retorno es un "
                "**ARIMA(p, 1, q)** sobre ln(precio), y el AR(p) de la app es un ARIMA(p, 1, 0).")
    st.markdown("**Prueba de Dickey-Fuller aumentada (ADF).** H₀: la serie tiene raíz unitaria (no es estacionaria). "
                f"Si el p-valor es menor que {config.ALFA_ADF}, se rechaza H₀ y la serie se considera estacionaria; si "
                "no, se diferencia una vez más y se repite la prueba.")
    st.dataframe(pd.DataFrame({
        "diferencias (d)": [paso["d"] for paso in analisis["pasos"]],
        "serie": [paso["serie"] for paso in analisis["pasos"]],
        "estadístico ADF": [round(paso["estadistico"], 3) for paso in analisis["pasos"]],
        "valor crítico 5%": [round(paso["critico_5"], 3) for paso in analisis["pasos"]],
        "p-valor": [texto_p_valor(paso["p_valor"]) for paso in analisis["pasos"]],
        "rezagos usados": [paso["rezagos"] for paso in analisis["pasos"]],
        "conclusión": ["Estacionaria" if paso["estacionaria"] else "No estacionaria" for paso in analisis["pasos"]],
    }), hide_index=True)
    ultimo_paso = analisis["pasos"][-1]
    if ultimo_paso["estacionaria"]:
        st.success(f"Con **d = {analisis['d']}** la serie ya es estacionaria: {ultimo_paso['serie']}. "
                   f"El orden de integración del ARIMA es d = {analisis['d']}.")
    else:
        st.warning("Ninguna de las diferencias probadas pasó la prueba ADF.")

    retorno_pct = 100 * np.log(completas["Close"].astype(float)).diff()
    fig_r = go.Figure(go.Scatter(x=x_serie, y=retorno_pct, mode="lines", name="Log-retorno",
                                 line=dict(color=COLOR_REAL, width=1)))
    fig_r.update_yaxes(ticksuffix="%")
    st.plotly_chart(estilo(fig_r, alto=320, titulo="Log-retorno por hora: la serie diferenciada oscila alrededor de 0"),
                    key="serie_retorno")
    c5, c6 = st.columns(2)
    grafico_correlacion(analisis["acf_ret"], analisis["n"], "ACF del log-retorno", "serie_acf", c5)
    grafico_correlacion(analisis["pacf_ret"], analisis["n"], "PACF del log-retorno", "serie_pacf", c6)
    texto_p = f" El AR(p) eligió **p = {arts['resumen']['ar']['p']}** por validación cruzada." if arts else ""
    st.caption("Las barras que salen de la franja punteada son autocorrelaciones significativas al 5%. La ACF sugiere el "
               "orden q de la parte MA y la PACF el orden p de la parte AR. En naranja, los rezagos de "
               f"{VELAS_SESION}, {2 * VELAS_SESION}, {3 * VELAS_SESION}… horas (una sesión de {VELAS_SESION} velas, "
               f"dos sesiones…): si sobresalen, hay estacionalidad diaria.{texto_p}")

    if not arts or "arima" not in arts["resumen"]:
        st.info("Los pasos 3 y 4 aparecen cuando se entrena el ARIMA con:")
        st.code(f"python entrenar.py --ticker {ticker}", language="bash")
    else:
        arima = arts["resumen"]["arima"]
        carpeta_arima = arts["carpeta"]

        st.subheader("3. Estimación y selección del modelo")
        busqueda = pd.read_csv(carpeta_arima / "busqueda_arima.csv")
        st.markdown(f"Se ajustaron **{len(busqueda)} modelos ARMA(p, q)**, con p de 0 a {int(busqueda['p'].max())} y q "
                    f"de 0 a {int(busqueda['q'].max())}, sobre el log-retorno (en %) del tramo de desarrollo, por máxima "
                    "verosimilitud y con errores estándar robustos. Cada uno se revisa con cinco requisitos:")
        st.dataframe(pd.DataFrame(REQUISITOS_ARIMA_TEXTO, columns=["requisito", "cómo se verifica", "por qué se exige"]),
                     hide_index=True)
        if "requisitos" not in arima:
            st.info("Este modelo se entrenó antes de la selección por requisitos; se actualiza al volver a entrenar.")
        else:
            fallas = [modelos_series.REQUISITOS_ARIMA[clave] for clave, cumple in arima["requisitos"].items()
                      if not cumple]
            st.markdown("**Regla de selección:** gana el modelo que cumple los cinco requisitos; si ninguno los cumple "
                        "todos, el que cumple más, y entre los que cumplen los mismos, el de menor AIC (el criterio de "
                        f"información de Akaike, que premia el ajuste y castiga cada parámetro). Elegido: "
                        f"**{arima['nombre']}**, con **{arima['requisitos_cumplidos']} de 5** requisitos "
                        f"(AIC {arima['aic']:.1f}, BIC {arima['bic']:.1f}).")
            if fallas:
                st.info("Requisitos que no cumple: " + "; ".join(fallas) + ". Ningún modelo de la rejilla cumplió "
                        "más requisitos que este.")
            tabla_req = busqueda.pivot(index="p", columns="q", values="requisitos").sort_index()
            tabla_aic = busqueda.pivot(index="p", columns="q", values="aic").reindex_like(tabla_req)
            diferencia = tabla_aic.to_numpy() - arima["aic"]
            textos = [[f"{int(n)}/5 · {d:+.1f}" for n, d in zip(fila_n, fila_d)]
                      for fila_n, fila_d in zip(tabla_req.to_numpy(), diferencia)]
            fig_req = go.Figure(go.Heatmap(z=tabla_req.to_numpy(), x=[f"q = {q}" for q in tabla_req.columns],
                                           y=[f"p = {p}" for p in tabla_req.index], text=textos, texttemplate="%{text}",
                                           colorscale="Blues", zmin=0, zmax=5, showscale=False,
                                           hovertemplate="%{y}, %{x}<br>%{text}<extra></extra>"))
            fig_req.update_yaxes(autorange="reversed")
            st.plotly_chart(estilo(fig_req, alto=380, titulo="Requisitos cumplidos por cada ARMA(p, q) · AIC menos el "
                                   "AIC del elegido", saltos=False, hover="closest"), key="arima_requisitos")
            st.caption("Cada celda muestra cuántos de los 5 requisitos cumple el modelo y su AIC menos el AIC del elegido "
                       "(negativo = AIC menor que el del elegido).")
            ranking = busqueda.head(10)
            st.dataframe(pd.DataFrame({
                "modelo": [f"ARIMA({int(p)},1,{int(q)})" for p, q in zip(ranking["p"], ranking["q"])],
                "requisitos": [f"{int(n)}/5" for n in ranking["requisitos"]],
                "estacionario": ["sí" if v else "no" for v in ranking["estacionario"]],
                "invertible": ["sí" if v else "no" for v in ranking["invertible"]],
                "coef. significativos": ["sí" if v else "no" for v in ranking["coef_significativos"]],
                "Ljung-Box (p mínimo)": [texto_p_valor(v) for v in ranking["ljung_box_p"]],
                "Jarque-Bera (p)": [texto_p_valor(v) for v in ranking["jarque_bera_p"]],
                "AIC": ranking["aic"].round(1),
                "raíces casi comunes": ["sí" if v else "no" for v in ranking["raices_casi_comunes"]],
            }), hide_index=True)
            with st.expander("Tabla completa de la búsqueda"):
                st.dataframe(busqueda, hide_index=True)

            st.markdown("**Coeficientes del modelo elegido.** Prueba z de cada coeficiente, H₀: el coeficiente vale 0. "
                        "Con errores estándar robustos (sándwich), que siguen siendo válidos aunque los residuos no sean "
                        "normales.")
            st.dataframe(pd.DataFrame({
                "parámetro": [nombre_parametro(c["parametro"]) for c in arima["coeficientes"]],
                "estimación": [round(c["estimacion"], 4) for c in arima["coeficientes"]],
                "error estándar": [round(c["error_estandar"], 4) for c in arima["coeficientes"]],
                "z": [round(c["z"], 2) for c in arima["coeficientes"]],
                "p-valor": [texto_p_valor(c["p_valor"]) for c in arima["coeficientes"]],
                "IC 95%": [f"[{c['ic_bajo']:.4f}, {c['ic_alto']:.4f}]" for c in arima["coeficientes"]],
                "significativo al 5%": ["sí" if c["significativo"] else "no" for c in arima["coeficientes"]],
            }), hide_index=True)
            st.caption("La media μ no cuenta para el requisito: se prueban solo los coeficientes φ (parte AR) y θ (parte "
                       "MA), que son los que definen el orden del modelo. Un coeficiente es significativo si su "
                       "intervalo del 95% no incluye el 0. Un ARIMA(0,1,0) cumple el requisito porque no tiene φ ni θ "
                       "que probar: es la caminata aleatoria con deriva.")
            correlacion = arima.get("correlacion", {})
            if correlacion.get("matriz"):
                etiquetas = [nombre_parametro(n) for n in correlacion["nombres"]]
                matriz = np.array([[np.nan if v is None else v for v in fila] for fila in correlacion["matriz"]])
                fig_cor = go.Figure(go.Heatmap(z=matriz, x=etiquetas, y=etiquetas, zmin=-1, zmax=1,
                                               colorscale="RdBu", text=np.round(matriz, 2), texttemplate="%{text}",
                                               showscale=False))
                fig_cor.update_yaxes(autorange="reversed")
                st.plotly_chart(estilo(fig_cor, alto=320, titulo="Correlación entre las estimaciones", saltos=False,
                                       hover="closest"), key="arima_correlacion")
                st.caption("Correlaciones cercanas a ±1 entre un φ y un θ indican que los dos parámetros explican lo "
                           "mismo: el modelo tiene parámetros de sobra.")
            if arima.get("raices_casi_comunes"):
                st.warning("Una raíz del AR casi coincide con una del MA: se cancelan y el modelo es, en la práctica, "
                           "uno de menor orden.")

        c7, c8 = st.columns([3, 2])
        c7.markdown("**Modelo elegido**, sobre el log-retorno en %:")
        c7.latex(ecuacion_arima(arima))
        c8.markdown(f"Media μ = {arima['constante']:.4f}% por hora y varianza del ruido σ² = {arima['sigma2']:.4f}. "
                    "Como el modelo es estacionario, su pronóstico a varias horas tiende a μ: después de pocas horas "
                    "el ARIMA, igual que el AR, pronostica que el precio crece a ese ritmo constante.")

        filas_raices = []
        for polinomio, raices in (("AR", arima["raices_ar"]), ("MA", arima["raices_ma"])):
            for real, imaginaria in raices:
                modulo = float(np.hypot(real, imaginaria))
                filas_raices.append({"polinomio": polinomio,
                                     "raíz": f"{real:.3f} {'+' if imaginaria >= 0 else '-'} {abs(imaginaria):.3f}i",
                                     "módulo": round(modulo, 3),
                                     "fuera del círculo unitario": "sí" if modulo > 1 else "no"})
        st.markdown("**Raíces de los polinomios.** El modelo es **estacionario** si todas las raíces del polinomio AR "
                    "quedan fuera del círculo unitario, e **invertible** si también quedan fuera las del polinomio MA.")
        if filas_raices:
            c9, c10 = st.columns([2, 3])
            c9.dataframe(pd.DataFrame(filas_raices), hide_index=True)
            angulo = np.linspace(0, 2 * np.pi, 200)
            fig_raiz = go.Figure(go.Scatter(x=np.cos(angulo), y=np.sin(angulo), mode="lines", name="Círculo unitario",
                                            line=dict(color=COLOR_REFERENCIA, dash="dash"), hoverinfo="skip"))
            for polinomio, raices, color in (("AR", arima["raices_ar"], COLORES_MODELO["ARIMA"]),
                                             ("MA", arima["raices_ma"], COLOR_ACENTO)):
                if raices:
                    inversas = 1 / np.array([complex(real, imaginaria) for real, imaginaria in raices])
                    fig_raiz.add_trace(go.Scatter(x=inversas.real, y=inversas.imag, mode="markers",
                                                  name=f"Raíces inversas {polinomio}", marker=dict(size=11, color=color)))
            fig_raiz.update_xaxes(range=[-1.2, 1.2], zeroline=True)
            fig_raiz.update_yaxes(range=[-1.2, 1.2], zeroline=True, scaleanchor="x")
            c10.plotly_chart(estilo(fig_raiz, alto=340, titulo="Raíces inversas: deben quedar dentro del círculo",
                                    saltos=False, hover="closest"), key="arima_raices")
        else:
            st.caption("El modelo elegido no tiene parte AR ni MA: el log-retorno es su media más ruido blanco, es decir, "
                       "el precio es una caminata aleatoria con deriva.")

        st.subheader("4. Diagnóstico de los residuos")
        diagnostico = arima["diagnostico"]
        filas_diag = [{"prueba": f"Ljung-Box, {prueba['rezagos']} rezagos", "H₀": "los residuos no están correlacionados",
                       "estadístico": round(prueba["estadistico"], 2), "p-valor": texto_p_valor(prueba["p_valor"]),
                       "conclusión": ("No se rechaza: se comportan como ruido blanco"
                                      if prueba["p_valor"] >= config.ALFA_PRUEBAS else "Se rechaza: queda autocorrelación")}
                      for prueba in diagnostico["ljung_box"]]
        jb = diagnostico["jarque_bera"]
        filas_diag.append({"prueba": "Jarque-Bera", "H₀": "los residuos son normales",
                           "estadístico": round(jb["estadistico"], 1), "p-valor": texto_p_valor(jb["p_valor"]),
                           "conclusión": ("No se rechaza: son normales" if jb["p_valor"] >= config.ALFA_PRUEBAS
                                          else "Se rechaza: no son normales")})
        st.dataframe(pd.DataFrame(filas_diag), hide_index=True)
        st.caption(f"Asimetría de los residuos: {jb['asimetria']:.2f}; curtosis: {jb['curtosis']:.1f} (una normal tiene "
                   "0 y 3). Una curtosis alta indica colas pesadas: hay más horas con movimientos grandes de lo que "
                   "predice una normal, algo típico de los retornos financieros. No sesga el pronóstico puntual, pero "
                   "hace que los intervalos basados en la normal queden cortos.")
        ruta_residuos = carpeta_arima / "residuos_arima.csv"
        if ruta_residuos.exists():
            teoricos, observados, acf_residuos, n_residuos = residuos_arima(str(ruta_residuos),
                                                                            arts["resumen"]["entrenado_en"])
            c11, c12 = st.columns(2)
            fig_qq = go.Figure(go.Scatter(x=teoricos, y=observados, mode="markers", name="Residuos",
                                          marker=dict(color=COLORES_MODELO["ARIMA"], size=4)))
            limite = float(max(abs(teoricos).max(), 1))
            fig_qq.add_trace(go.Scatter(x=[-limite, limite], y=[-limite, limite], mode="lines", name="Normal",
                                        line=dict(color=COLOR_REFERENCIA, dash="dash")))
            fig_qq.update_xaxes(title="Cuantil de una normal estándar")
            fig_qq.update_yaxes(title="Residuo estandarizado")
            c11.plotly_chart(estilo(fig_qq, alto=340, titulo="Gráfico QQ de los residuos", saltos=False,
                                    hover="closest"), key="arima_qq")
            grafico_correlacion(acf_residuos, n_residuos, "ACF de los residuos", "arima_acf_residuos", c12)
            st.caption("Si los residuos fueran normales, los puntos del gráfico QQ seguirían la línea punteada; los "
                       "extremos que se alejan son las colas pesadas. En la ACF de los residuos, las barras dentro de la "
                       "franja indican que el modelo ya recogió la autocorrelación del log-retorno.")


# ------------------------------------------------------------------ pestaña 3: evaluación

with tab_eval:
    if not arts:
        st.info("Todavía no hay modelos entrenados para esta acción. Se entrenan desde la terminal con:")
        st.code(f"python entrenar.py --ticker {ticker}", language="bash")
    else:
        carpeta = arts["carpeta"]
        resumen = arts["resumen"]
        st.markdown(f"Entrenado el **{resumen['entrenado_en']}** con **{resumen['velas']:,}** velas de {INTERVALO}. "
                    "División temporal (sin mezclar el orden):")
        st.dataframe(pd.DataFrame(resumen["tramos"]).T.rename_axis("tramo").reset_index(), hide_index=True)
        if "validacion_cruzada" in resumen:
            cv = resumen["validacion_cruzada"]
            st.markdown(f"AR y redes eligen sus hiperparámetros con **validación cruzada temporal** "
                        f"(`TimeSeriesSplit`, {cv['pliegues']} pliegues) sobre entrenamiento + validación. "
                        "ARIMA elige p y q sobre ese mismo tramo con los **requisitos de Box-Jenkins** (raíces, coeficientes "
                        "significativos y residuos), con el AIC para desempatar.")
            st.dataframe(pd.DataFrame(cv["tramos"]).rename_axis("pliegue").reset_index()
                         .assign(pliegue=lambda d: d["pliegue"] + 1), hide_index=True)

        metricas = pd.read_csv(carpeta / "metricas.csv")
        vista = pd.DataFrame({
            "modelo": metricas["modelo"],
            "acierto sube/baja": metricas["acierto_direccion"].map(lambda v: "" if pd.isna(v) else f"{100 * v:.1f}%"),
            "margen 95%": metricas["margen_95"].map(lambda v: "" if pd.isna(v) else f"±{100 * v:.1f} pts"),
            "error medio del precio": metricas["mae_precio"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}"),
            "RMSE del retorno": metricas["rmse_retorno_pct"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}%"),
        })
        st.dataframe(vista, hide_index=True)

        con_acierto = metricas.dropna(subset=["acierto_direccion"])
        fig = go.Figure(go.Bar(
            x=100 * con_acierto["acierto_direccion"], y=con_acierto["modelo"], orientation="h",
            error_x=dict(type="data", array=100 * con_acierto["margen_95"], color=COLOR_REFERENCIA),
            marker_color=[color_modelo(m) for m in con_acierto["modelo"]],
            text=[f"{100 * v:.1f}%" for v in con_acierto["acierto_direccion"]], textposition="inside",
        ))
        fig.add_vline(x=50, line_dash="dash", line_color=COLOR_REFERENCIA)
        fig.update_xaxes(ticksuffix="%", range=[0, 100])
        fig.update_layout(showlegend=False)
        st.plotly_chart(estilo(fig, alto=340, titulo="Acierto de dirección en el conjunto de prueba", saltos=False,
                               hover="closest"),
                        key="eval_acierto")
        st.caption("Un modelo solo aporta si su acierto supera a las líneas base (gris) por más que el margen de error. "
                   "Si las barras de error se cruzan con el 50% o con 'Siempre sube', no hay evidencia de que prediga "
                   "mejor que el azar. AR, ARIMA, RNN y LSTM predicen el log-retorno (regresión) y aciertan cuando "
                   "su signo coincide con el del retorno real; la CNN de velas clasifica directamente (acierta cuando "
                   "da más de 50% y la hora sube, o menos de 50% y baja), por eso no tiene error del precio.")

        predicciones = pd.read_csv(carpeta / "predicciones_prueba.csv")
        columnas_modelo = [c for c in predicciones.columns if es_modelo(c) and c != velas_cnn.NOMBRE]
        elegido = st.selectbox("Comparar precio real y predicho", columnas_modelo)
        tramo = predicciones.tail(150)
        x = a_hora_ny(pd.to_datetime(tramo["marca_objetivo"], utc=True))
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=x, y=tramo["precio_real"], mode="lines", name="Precio real",
                                   line=dict(color=COLOR_REAL, width=2)))
        fig_p.add_trace(go.Scatter(x=x, y=tramo["cierre_referencia"] * np.exp(tramo[elegido]), mode="lines",
                                   name=f"Predicción {elegido}", line=dict(color=color_modelo(elegido), width=2)))
        st.plotly_chart(estilo(fig_p, alto=380, titulo="Últimas 150 velas del conjunto de prueba"), key="eval_real_pred")
        st.caption("Si la predicción parece una copia del precio real desplazada una vela, el modelo está repitiendo el "
                   "último precio: por eso se evalúa la dirección y no solo qué tan parecidas se ven las curvas.")

        st.markdown("**Curvas de aprendizaje de las redes**")

        def curva_aprendizaje(historial, titulo, eje_y, clave, columna):
            fig_h = go.Figure()
            if "pliegue" in historial:
                historial["pliegue"] = historial["pliegue"].astype(str)
                pliegues_hist = [p for p in historial["pliegue"].unique() if p != "final"]
                for i, pliegue in enumerate(pliegues_hist):
                    tramo_h = historial[historial["pliegue"] == pliegue]
                    fig_h.add_trace(go.Scatter(x=tramo_h["epoca"] + 1, y=tramo_h["val_loss"],
                                               name=f"Validación, pliegue {pliegue}",
                                               line=dict(color=COLORES_PLIEGUE[i % len(COLORES_PLIEGUE)], width=2)))
                final = historial[historial["pliegue"] == "final"]
                fig_h.add_trace(go.Scatter(x=final["epoca"] + 1, y=final["loss"], name="Entrenamiento del modelo final",
                                           line=dict(color=COLORES_MODELO["LSTM"], width=2, dash="dash")))
            else:
                fig_h.add_trace(go.Scatter(x=historial["epoca"] + 1, y=historial["loss"], name="Entrenamiento",
                                           line=dict(color=COLOR_REAL, width=2)))
                fig_h.add_trace(go.Scatter(x=historial["epoca"] + 1, y=historial["val_loss"], name="Validación",
                                           line=dict(color=COLOR_ACENTO, width=2)))
            fig_h.update_xaxes(title="Época")
            fig_h.update_yaxes(title=eje_y)
            columna.plotly_chart(estilo(fig_h, alto=340, titulo=titulo, saltos=False), key=clave)

        c1, c2 = st.columns(2)
        for columna, tipo in ((c1, "rnn"), (c2, "lstm")):
            ruta = carpeta / f"historial_{tipo}.csv"
            if ruta.exists():
                mejor = resumen["redes"][tipo]
                titulo = (f"{'RNN simple' if tipo == 'rnn' else 'LSTM'}: ventana {mejor['ventana']}, "
                          f"{mejor['unidades']} unidades, {mejor['capas']} capa(s), "
                          f"{mejor.get('optimizador', 'Adam')} con tasa {mejor['tasa']}")
                curva_aprendizaje(pd.read_csv(ruta), titulo, "MSE (retorno estandarizado)", f"eval_curva_{tipo}",
                                  columna)
        if (carpeta / "historial_cnn.csv").exists() and "cnn" in resumen:
            mejor = resumen["cnn"]
            titulo = (f"{velas_cnn.NOMBRE}: {mejor['velas']} velas, {mejor['filtros']} filtros, "
                      f"{mejor.get('optimizador', 'SGD')} con tasa {mejor['tasa']}")
            curva_aprendizaje(pd.read_csv(carpeta / "historial_cnn.csv"), titulo, "Entropía cruzada binaria",
                              "eval_curva_cnn", st)
        if "validacion_cruzada" in resumen:
            st.caption("Cada línea continua es el error de validación de la mejor combinación en un pliegue: el "
                       "entrenamiento se detiene cuando deja de bajar. La línea punteada es el modelo final, "
                       "reentrenado con todo el tramo de desarrollo durante el promedio de épocas de los pliegues. "
                       "La RNN y la LSTM hacen regresión del log-retorno (error cuadrático medio); la CNN clasifica "
                       "si la hora siguiente sube o baja (entropía cruzada).")

        for archivo, titulo in (("busqueda_rnn.csv", "Búsqueda de hiperparámetros: RNN simple"),
                                ("busqueda_lstm.csv", "Búsqueda de hiperparámetros: LSTM"),
                                ("busqueda_cnn.csv", f"Búsqueda de hiperparámetros: {velas_cnn.NOMBRE}"),
                                ("busqueda_ar.csv", "Selección de p para AR(p)"),
                                ("busqueda_arima.csv", "Selección de p y q para ARIMA (requisitos y AIC)")):
            if (carpeta / archivo).exists():
                with st.expander(titulo):
                    st.dataframe(pd.read_csv(carpeta / archivo), hide_index=True)


# ------------------------------------------------------------------ pestaña 4: datos

with tab_datos:
    st.markdown(
        f"Yahoo solo guarda unos **60 días** de velas de 2 minutos y unos **2 años** de velas de 1 hora. "
        f"Cada consulta se une con lo que ya está guardado en `datos/`, así que mientras el dashboard o "
        f"`recolector.py` estén corriendo, la historia de 2 minutos se sigue acumulando."
    )
    for intervalo, nombre in ((VIVO, "Velas de 2 minutos"), (INTERVALO, "Velas de 1 hora")):
        guardado = datos.leer_local(ticker, intervalo)
        st.markdown(f"**{nombre}**")
        if guardado.empty:
            st.caption("Sin datos guardados todavía.")
            continue
        st.caption(f"{len(guardado):,} velas · desde {hora_doble(guardado.index[0])} hasta {hora_doble(guardado.index[-1])}")
        vista = guardado.tail(10).copy()
        vista.index = a_hora_ny(vista.index)
        st.dataframe(vista.rename_axis("hora NY").reset_index(), hide_index=True)
        st.download_button(f"Descargar CSV ({nombre.lower()})", datos.ruta_csv(ticker, intervalo).read_bytes(),
                           file_name=datos.ruta_csv(ticker, intervalo).name, mime="text/csv", key=f"descarga_{intervalo}")


# ------------------------------------------------------------------ mis datos y pie de página

with tab_mis_datos:
    interfaz_cuenta.pestana_mis_datos(usuario)

with tab_etica:
    interfaz_etica.pestana()

st.divider()
st.caption(f"{config.NOMBRE_APP} es un proyecto académico de la Escuela Colombiana de Ingeniería Julio Garavito. Nada "
           "de lo que se muestra es una recomendación de inversión, y el portafolio simulado usa solo dinero ficticio.")
interfaz_cuenta.mostrar_politica()