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
import modelos_series
import portafolio as portafolio_simulado
import prediccion
import preparacion
import varias_horas
import velas_cnn

st.set_page_config(page_title="Predicción de acciones", layout="wide")

INTERVALO = config.INTERVALO_MODELO
VIVO = config.INTERVALO_VIVO
COLOR_REAL = "#2a78d6"
COLOR_REFERENCIA = "#898781"
COLOR_SUBE = "#0ca30c"
COLOR_BAJA = "#d03b3b"
COLORES_MODELO = {"Prophet": "#eb6834", "AR": "#1baf7a", "RNN simple": "#e87ba4", "LSTM": "#9085e9",
                  "CNN de velas": "#eda100"}
COLORES_PLIEGUE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SALTOS_HORARIO = [dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")]


def color_modelo(nombre):
    return COLORES_MODELO.get("AR" if nombre.startswith("AR(") else nombre, COLOR_REFERENCIA)


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


@st.cache_resource(show_spinner="Ajustando Prophet…", max_entries=48)
def prophet_ajustado(ticker, intervalo, ultima_iso, ajuste, incertidumbre=True):
    serie = completas_hasta(ticker, intervalo, ultima_iso)["Close"]
    return modelos_series.ajustar_prophet(serie, dict(ajuste), incertidumbre=incertidumbre)


@st.cache_data(show_spinner=False, max_entries=12)
def prediccion_siguiente(ticker, intervalo, ultima_iso, ajuste, version, _arts):
    completas = completas_hasta(ticker, intervalo, ultima_iso)
    modelo = prophet_ajustado(ticker, intervalo, ultima_iso, ajuste)
    return prediccion.predecir_siguiente(completas, intervalo, _arts, modelo)


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
def pronostico_varias_horas(ticker, intervalo, origen_iso, horas, ajuste, version, _arts):
    completas = completas_hasta(ticker, intervalo, origen_iso)
    modelo = prophet_ajustado(ticker, intervalo, origen_iso, ajuste)
    return varias_horas.pronosticar(completas, intervalo, horas, _arts, modelo)


@st.cache_data(show_spinner="Evaluando cada horizonte en las últimas sesiones (la primera vez tarda unos segundos)…",
               max_entries=12)
def evaluacion_varias_horas(ticker, intervalo, ultima_iso, horas, sesiones, ajuste, version, _arts):
    completas = completas_hasta(ticker, intervalo, ultima_iso)
    f = preparacion.construir_caracteristicas(completas)
    ultimo_origen = len(f) - 1 - horas
    posiciones = list(range(max(ultimo_origen - sesiones * datos.velas_por_dia(intervalo) + 1, 60), ultimo_origen + 1))
    dias = f.index.normalize()

    def prophet_para(j):
        anteriores = np.flatnonzero(dias < dias[j])
        return prophet_ajustado(ticker, intervalo, f.index[anteriores[-1]].isoformat(), ajuste, incertidumbre=False)

    return varias_horas.errores_por_horizonte(completas, horas, posiciones, _arts, prophet_para)


@st.cache_data(show_spinner="Calculando pronóstico de Prophet…", max_entries=6)
def pronostico_prophet(ticker, intervalo, ultima_iso, ajuste, sesiones=30, velas_futuras=14):
    modelo = prophet_ajustado(ticker, intervalo, ultima_iso, ajuste)
    df = completas_hasta(ticker, intervalo, ultima_iso)
    ultimas = df.iloc[-sesiones * datos.velas_por_dia(intervalo):]
    futuras = datos.proximas_velas(df.index[-1], intervalo, velas_futuras)
    pronostico = modelos_series.pronosticar_prophet(modelo, ultimas.index.append(futuras))
    pronostico["real"] = np.r_[ultimas["Close"].to_numpy(), np.full(len(futuras), np.nan)]
    hora, semana = modelos_series.efectos_estacionales(modelo, intervalo)
    return pronostico, hora, semana, len(futuras)


@st.cache_data(show_spinner="Calculando pruebas de estacionariedad…", max_entries=6)
def analisis_serie(ticker, intervalo, ultima_iso, rezagos=20):
    from statsmodels.tsa.stattools import acf, adfuller, pacf

    precio = completas_hasta(ticker, intervalo, ultima_iso)["Close"]
    retorno = np.log(precio).diff().dropna()
    filas = []
    for nombre, serie in (("Precio de cierre", precio), ("Log-retorno", retorno)):
        estadistico, p_valor, *_ = adfuller(serie.to_numpy(), autolag="AIC")
        filas.append({
            "serie": nombre,
            "estadístico ADF": round(float(estadistico), 3),
            "p-valor": round(float(p_valor), 4),
            "conclusión": "Estacionaria" if p_valor < 0.05 else "No estacionaria",
        })
    return pd.DataFrame(filas), acf(retorno, nlags=rezagos)[1:], pacf(retorno, nlags=rezagos)[1:], len(retorno)


@st.cache_data(show_spinner="Revisando cómo les fue a los modelos en la última sesión…", max_entries=6)
def evaluacion_ultima_sesion(ticker, intervalo, ultima_iso, ajuste, version, _arts):
    completas = completas_hasta(ticker, intervalo, ultima_iso)
    previas = completas[completas.index < completas.index[-1].normalize()]
    if previas.empty:
        return pd.DataFrame()
    modelo_previo = prophet_ajustado(ticker, intervalo, previas.index[-1].isoformat(), ajuste)
    return prediccion.evaluar_ultima_sesion(completas, intervalo, _arts, modelo_previo)


def ajuste_prophet(arts):
    """Hiperparámetros elegidos en el entrenamiento, como tupla para que sirvan de llave de caché."""
    elegidos = arts["resumen"].get("prophet") if arts else None
    return tuple(sorted(modelos_series.parametros_prophet(elegidos).items()))


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
    tabla, info = prediccion_siguiente(simbolo, INTERVALO, ultima, ajuste_prophet(arts_simbolo),
                                       arts_simbolo["resumen"]["entrenado_en"], arts_simbolo)
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
    st.sidebar.warning("Esta acción no tiene modelos entrenados. Por ahora solo se muestra Prophet.")
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

(tab_vivo, tab_horas, tab_estrategias, tab_portafolio, tab_agente, tab_prophet, tab_serie, tab_eval, tab_datos,
 tab_mis_datos) = st.tabs(
    ["En vivo y próxima hora", "Varias horas", "Estrategias", "Portafolio simulado", "Agente", "Prophet",
     "Análisis de la serie", "Evaluación de modelos", "Datos recolectados", "Mis datos"]
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

    modelo_prophet = prophet_ajustado(ticker, INTERVALO, completas.index[-1].isoformat(), ajuste_prophet(arts))
    tabla, info = prediccion.predecir_siguiente(completas, INTERVALO, arts, modelo_prophet)
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

    if arts or modelo_prophet is not None:
        version = arts["resumen"]["entrenado_en"] if arts else None
        revision = evaluacion_ultima_sesion(ticker, INTERVALO, completas.index[-1].isoformat(), ajuste_prophet(arts),
                                            version, arts)
        if not revision.empty:
            columnas_modelos = [c for c in revision.columns if c.startswith("AR(") or c in COLORES_MODELO]
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


with tab_horas:
    st.markdown(
        "Pronóstico de varias horas de mercado hacia adelante con el modelo ganador de cada tipo. "
        "AR, RNN y LSTM predicen una hora, toman esa predicción como si fuera un dato real y siguen con la "
        "siguiente (**pronóstico recursivo**); Prophet evalúa su curva directamente en cada hora futura."
    )
    control_1, control_2 = st.columns([1, 2])
    horas = control_1.slider("Horas de mercado hacia adelante", min_value=1, max_value=35, value=14,
                             help="Una sesión tiene 7 velas de 1 hora: 14 ≈ 2 sesiones, 35 ≈ una semana.")
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
    futuro, info_h = pronostico_varias_horas(ticker, INTERVALO, origen.isoformat(), horas, ajuste_prophet(arts),
                                             version_modelos, arts)
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
    titulo_h = (f"Pronóstico de {horas} {'hora' if horas == 1 else 'horas'} desde {hora_doble(origen)}"
                + (" y lo que pasó en realidad" if not reales_h.empty else ""))
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

    st.markdown("**¿Qué tan rápido pierde precisión cada modelo al alejarse?**")
    if st.checkbox("Calcular el error promedio por horizonte en las últimas 10 sesiones"):
        error_h, acierto_h, n_origenes = evaluacion_varias_horas(ticker, INTERVALO, ultima_iso, horas, 10,
                                                                 ajuste_prophet(arts), version_modelos, arts)
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
    tabla_ag, info_ag = prediccion_siguiente(ticker, INTERVALO, ultima_iso, ajuste_prophet(arts), version_ag, arts)
    futuro_ag, info_futuro_ag = pronostico_varias_horas(ticker, INTERVALO, ultima_iso, datos.velas_por_dia(INTERVALO),
                                                        ajuste_prophet(arts), version_ag, arts)
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


# ------------------------------------------------------------------ pestaña 2: Prophet

with tab_prophet:
    with st.expander("¿Cómo funciona Prophet?", expanded=not arts):
        st.markdown(r"""
Prophet (de Meta; el paquete se llamaba `fbprophet` y hoy se instala como `prophet`) no mira los valores
anteriores como hacen AR o una RNN. Modela la serie como una **suma de funciones del tiempo**:

$$y(t) = g(t) + s(t) + h(t) + \varepsilon_t$$

- **g(t), tendencia:** rectas por tramos. Prophet coloca posibles *puntos de cambio* donde la pendiente puede
  cambiar; `changepoint_prior_scale` controla qué tan flexible es (valores altos = tendencia más nerviosa).
- **s(t), estacionalidad:** sumas de senos y cosenos (series de Fourier). Aquí hay una diaria (efecto de la hora
  del día) y una semanal (efecto del día de la semana).
- **h(t), festivos:** efectos de fechas especiales (no se usan en este proyecto).
- **ε, ruido:** lo que el modelo no explica.

Se ajusta como una regresión sobre el tiempo. En este proyecto se modela **log(precio)**, así los efectos
estacionales quedan en porcentaje. Para predecir, se evalúa la función en la hora futura.

**Limitación:** el precio de una acción se comporta casi como una caminata aleatoria. Prophet capta la tendencia
y los patrones por hora, pero no reacciona a lo que pasó en la última vela; por eso se compara con AR, RNN y LSTM.
""")
    ajuste = ajuste_prophet(arts)
    pronostico, efecto_hora, efecto_semana, n_futuras = pronostico_prophet(ticker, INTERVALO, ultima_iso, ajuste)
    historia, futuro = pronostico.iloc[:-n_futuras], pronostico.iloc[-n_futuras:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pronostico["ds"], y=pronostico["yhat_upper_precio"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pronostico["ds"], y=pronostico["yhat_lower_precio"], mode="lines",
                             line=dict(width=0), fill="tonexty", fillcolor="rgba(235,104,52,0.15)",
                             name="Intervalo del 80%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=historia["ds"], y=historia["real"], mode="markers", name="Precio real",
                             marker=dict(color=COLOR_REAL, size=5)))
    fig.add_trace(go.Scatter(x=historia["ds"], y=historia["yhat_precio"], mode="lines", name="Ajuste de Prophet",
                             line=dict(color=COLORES_MODELO["Prophet"], width=2)))
    fig.add_trace(go.Scatter(x=futuro["ds"], y=futuro["yhat_precio"], mode="lines+markers", name="Pronóstico",
                             line=dict(color=COLORES_MODELO["Prophet"], width=2, dash="dot"), marker=dict(size=8)))
    st.plotly_chart(estilo(fig, alto=450, titulo=f"Prophet: últimas 30 sesiones y próximas {n_futuras} horas "
                                                  f"({', '.join(f'{k} = {v}' for k, v in ajuste)})"),
                    key="prophet_pronostico")

    c1, c2, c3 = st.columns(3)
    fig_t = go.Figure(go.Scatter(x=historia["ds"], y=historia["trend_precio"], mode="lines", name="Tendencia",
                                 line=dict(color=COLORES_MODELO["Prophet"], width=2)))
    c1.plotly_chart(estilo(fig_t, alto=300, titulo="Tendencia g(t)"), key="prophet_tendencia")
    fig_d = go.Figure(go.Bar(x=efecto_hora["hora"], y=efecto_hora["efecto_pct"], marker_color=COLORES_MODELO["Prophet"],
                             name="Efecto"))
    fig_d.update_yaxes(ticksuffix="%")
    c2.plotly_chart(estilo(fig_d, alto=300, titulo="Estacionalidad diaria (hora NY)", saltos=False), key="prophet_diaria")
    fig_s = go.Figure(go.Bar(x=efecto_semana["dia"], y=efecto_semana["efecto_pct"], marker_color=COLORES_MODELO["Prophet"],
                             name="Efecto"))
    fig_s.update_yaxes(ticksuffix="%")
    c3.plotly_chart(estilo(fig_s, alto=300, titulo="Estacionalidad semanal", saltos=False), key="prophet_semanal")
    st.caption("Las barras muestran cuánto sube o baja el precio, en promedio, por la hora del día o el día de la semana "
               "según Prophet. Si los efectos son muy pequeños, la estacionalidad aporta poco a la predicción.")


# ------------------------------------------------------------------ pestaña 3: análisis de la serie

with tab_serie:
    completas = completas_hasta(ticker, INTERVALO, ultima_iso)
    retorno = 100 * np.log(completas["Close"]).diff()
    c1, c2 = st.columns(2)
    fig_p = go.Figure(go.Scatter(x=a_hora_ny(completas.index), y=completas["Close"], mode="lines", name="Cierre",
                                 line=dict(color=COLOR_REAL, width=1.5)))
    c1.plotly_chart(estilo(fig_p, alto=320, titulo="Precio de cierre por hora (no estacionario)"), key="serie_precio")
    fig_r = go.Figure(go.Scatter(x=a_hora_ny(completas.index), y=retorno, mode="lines", name="Log-retorno",
                                 line=dict(color=COLOR_REAL, width=1)))
    fig_r.update_yaxes(ticksuffix="%")
    c2.plotly_chart(estilo(fig_r, alto=320, titulo="Log-retorno por hora (oscila alrededor de 0)"), key="serie_retorno")

    tabla_adf, valores_acf, valores_pacf, n_retornos = analisis_serie(ticker, INTERVALO, ultima_iso)
    st.markdown("**Prueba de Dickey-Fuller aumentada (ADF).** Hipótesis nula: la serie tiene raíz unitaria "
                "(no es estacionaria). Si el p-valor es menor que 0.05, se rechaza y la serie se considera estacionaria.")
    st.dataframe(tabla_adf, hide_index=True)

    banda = 1.96 / np.sqrt(n_retornos)
    c3, c4 = st.columns(2)
    for columna, valores, nombre, clave in ((c3, valores_acf, "ACF", "acf"), (c4, valores_pacf, "PACF", "pacf")):
        rezagos = np.arange(1, len(valores) + 1)
        fig_c = go.Figure(go.Bar(x=rezagos, y=valores, marker_color=COLOR_REAL, name=nombre))
        for signo in (1, -1):
            fig_c.add_hline(y=signo * banda, line_dash="dash", line_color=COLOR_REFERENCIA)
        fig_c.update_xaxes(title="Rezago (horas)")
        columna.plotly_chart(estilo(fig_c, alto=300, titulo=f"{nombre} de los log-retornos", saltos=False), key=f"serie_{clave}")
    texto_p = f" El entrenamiento eligió **p = {arts['resumen']['ar']['p']}** por validación." if arts else ""
    st.caption("Las barras que salen de la franja punteada son autocorrelaciones significativas al 5%. "
               f"La PACF sugiere cuántos rezagos usar en un AR(p).{texto_p}")


# ------------------------------------------------------------------ pestaña 4: evaluación

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
                        "Prophet usa el tramo de validación, porque cada pliegue le exigiría decenas de ajustes más.")
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
            marker_color=[color_modelo(m) if m in COLORES_MODELO or m.startswith("AR(") else COLOR_REFERENCIA
                          for m in con_acierto["modelo"]],
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
                   "mejor que el azar. AR, Prophet, RNN y LSTM predicen el log-retorno (regresión) y aciertan cuando "
                   "su signo coincide con el del retorno real; la CNN de velas clasifica directamente (acierta cuando "
                   "da más de 50% y la hora sube, o menos de 50% y baja), por eso no tiene error del precio.")

        predicciones = pd.read_csv(carpeta / "predicciones_prueba.csv")
        columnas_modelo = [c for c in predicciones.columns
                           if c.startswith("AR(") or (c in COLORES_MODELO and c != velas_cnn.NOMBRE)]
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
                                           line=dict(color=COLORES_MODELO["Prophet"], width=2)))
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
                                ("busqueda_prophet.csv", "Búsqueda de hiperparámetros: Prophet")):
            if (carpeta / archivo).exists():
                with st.expander(titulo):
                    st.dataframe(pd.read_csv(carpeta / archivo), hide_index=True)


# ------------------------------------------------------------------ pestaña 5: datos

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

st.divider()
st.caption(f"{config.NOMBRE_APP} es un proyecto académico de la Escuela Colombiana de Ingeniería Julio Garavito. Nada "
           "de lo que se muestra es una recomendación de inversión, y el portafolio simulado usa solo dinero ficticio.")
interfaz_cuenta.mostrar_politica()