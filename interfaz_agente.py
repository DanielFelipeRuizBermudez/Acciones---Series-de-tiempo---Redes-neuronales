"""Pestaña del agente: chat con Gemini por medio de LiteLLM, con guardrails, cuota diaria e historial opcional."""

import streamlit as st

import agente
import config
import cuentas
import guardrails
import interfaz_cuenta
import registro_agente

PREGUNTAS_RAPIDAS = [
    "¿Cómo está la acción?",
    "¿Qué dicen las estrategias sobre comprar o vender?",
    "¿Qué tan confiables son los modelos?",
]
NOMBRES_DATOS = {"NOMBRE": "nombre", "DOCUMENTO": "documento", "CORREO": "correo", "TELEFONO": "teléfono",
                 "CUENTA": "número de cuenta"}
RESULTADOS = {
    "respondida": "Respondidas",
    "cifras_ocultas": "Respondidas con cifras ocultas",
    "bloqueada_entrada": "Bloqueadas antes de enviarse",
    "bloqueada_salida": "Respuestas bloqueadas",
    "cuota_agotada": "Sin cuota",
    "error": "Con error",
}


@st.cache_resource(show_spinner="Cargando el filtro de datos personales…")
def detector_nombres():
    return guardrails.DetectorNombres()


def _datos_reemplazados(conteo):
    return ", ".join(f"{cantidad} {NOMBRES_DATOS.get(tipo, tipo.lower())}" for tipo, cantidad in conteo.items())


def _responder(bd, token, ticker, modelo, clave_api, contexto, historial, pregunta, detector, guardar):
    entrada = guardrails.revisar_entrada(pregunta, detector)
    with st.chat_message("user"):
        st.markdown(entrada.texto)
        if entrada.datos:
            st.caption(f"Antes de enviar la pregunta se reemplazaron datos personales ({_datos_reemplazados(entrada.datos)}).")
    permitida, _ = registro_agente.consumir_cuota(bd, token)
    with st.chat_message("assistant"):
        if not permitida:
            registro_agente.auditar(bd, modelo, "cuota_agotada")
            st.warning(f"Ya se usaron las {config.AGENTE_CONSULTAS_DIARIAS} preguntas de hoy. La cuota se renueva a "
                       "medianoche, hora de Bogotá.")
            return
        if entrada.bloqueada:
            registro_agente.auditar(bd, modelo, "bloqueada_entrada", entrada.datos, entrada.motivo)
            st.warning(f"La pregunta no se envió a Gemini porque {entrada.motivo}. El agente solo explica los datos de "
                       "la acción que calcula la app.")
            return
        try:
            with st.spinner("Consultando a Gemini…"):
                texto = agente.responder(clave_api, modelo, contexto, historial[-config.AGENTE_MENSAJES_HISTORIAL:],
                                         entrada.texto)
        except Exception as error:
            registro_agente.auditar(bd, modelo, "error", entrada.datos, type(error).__name__)
            st.error(f"No se pudo consultar a Gemini ({type(error).__name__}): {error}")
            return
        salida = guardrails.revisar_salida(texto, contexto, entrada.texto)
        st.markdown(salida.texto)
        if salida.cifras_ocultas:
            st.caption(f"Se ocultaron {salida.cifras_ocultas} cifras que no aparecen en los datos de la app.")
    resultado = ("bloqueada_salida" if salida.bloqueada else "cifras_ocultas" if salida.cifras_ocultas
                 else "respondida")
    registro_agente.auditar(bd, modelo, resultado, entrada.datos, salida.motivo)
    nuevos = [{"rol": "usuario", "texto": entrada.texto}, {"rol": "modelo", "texto": salida.texto}]
    historial += nuevos
    if guardar:
        registro_agente.guardar_mensajes(bd, token, ticker, nuevos)


def _explicacion_guardrails(bd, detector):
    st.markdown(
        f"- **Entrada.** Los datos personales se reemplazan por etiquetas antes de enviar la pregunta. Detector en uso: "
        f"{detector.descripcion}. Las preguntas que intentan cambiar o revelar las instrucciones del agente se bloquean "
        f"sin enviarse, y cada pregunta puede tener hasta {config.AGENTE_MAX_CARACTERES} caracteres.\n"
        "- **Salida.** Se bloquean las respuestas que dan una recomendación personal de inversión, y las cifras que no "
        f"aparecen en los datos de la app se reemplazan por «{guardrails.CIFRA_OCULTA}».\n"
        f"- **Uso.** Cada cuenta tiene {config.AGENTE_CONSULTAS_DIARIAS} preguntas por día. La clave de la API de "
        "Gemini solo está en el servidor; la app la usa por medio de LiteLLM y nunca la muestra.\n"
        "- **Auditoría.** De cada pregunta se registran la fecha, el modelo, el resultado y el tipo de datos personales "
        "que se detectaron, sin el texto y sin el usuario."
    )
    resumen = registro_agente.resumen_auditoria(bd)
    if not resumen.empty:
        resumen["resultado"] = resumen["resultado"].map(RESULTADOS).fillna(resumen["resultado"])
        st.markdown("Preguntas de todos los usuarios en los últimos 7 días:")
        st.dataframe(resumen, hide_index=True)


def pestana_agente(usuario, ticker, clave_api, construir_contexto):
    bd, error = interfaz_cuenta.base()
    if error:
        st.error(error)
        return
    if not usuario:
        st.info(f"Para usar el agente hay que iniciar sesión en la pestaña Portafolio simulado. Cada cuenta tiene "
                f"{config.AGENTE_CONSULTAS_DIARIAS} preguntas por día.")
        return
    if not clave_api:
        st.info("El agente usa la API de Gemini y todavía no encuentra la clave. Se configura en `.streamlit/secrets.toml` "
                "dentro de la carpeta del proyecto (o en los Secrets de la app publicada) con esta línea, y recargando la "
                "página. Ese archivo está en `.gitignore`, así que la clave no se sube a GitHub.")
        st.code('GEMINI_API_KEY = "clave-de-gemini"', language="toml")
        return

    token = usuario["token"]
    detector = detector_nombres()
    guardar = cuentas.historial_activo(bd, token)
    usadas = registro_agente.consultas_hoy(bd, token)
    st.info(
        "Antes de salir de la app, cada pregunta pasa por un filtro que reemplaza nombres, documentos, correos, "
        "teléfonos y números de cuenta por etiquetas como [NOMBRE]. Google Gemini recibe solo esa versión, por medio de "
        "LiteLLM, junto con los datos de la acción que calcula el dashboard; no recibe tu alias, tu correo ni tu "
        f"contraseña. Hoy van {usadas} de {config.AGENTE_CONSULTAS_DIARIAS} preguntas. El historial de preguntas está "
        f"{'activado' if guardar else 'desactivado'} (se cambia en Mis datos)."
    )
    modelo = st.selectbox("Modelo de Gemini", config.GEMINI_MODELOS,
                          help="Si un modelo da error de cuota o no está disponible, se puede probar otro.")
    contexto = construir_contexto()

    clave_chat = f"chat_{ticker}"
    if clave_chat not in st.session_state:
        anteriores = (registro_agente.historial(bd, token, ticker, limite=config.AGENTE_MENSAJES_HISTORIAL)
                      if guardar else [])
        st.session_state[clave_chat] = [{"rol": m["rol"], "texto": m["texto"]} for m in anteriores]
    historial = st.session_state[clave_chat]

    rapida = None
    for columna, texto in zip(st.columns(len(PREGUNTAS_RAPIDAS)), PREGUNTAS_RAPIDAS):
        if columna.button(texto, key=f"rapida_{texto}"):
            rapida = texto
    for mensaje in historial:
        with st.chat_message("user" if mensaje["rol"] == "usuario" else "assistant"):
            st.markdown(mensaje["texto"])
    pregunta = st.chat_input(f"Una pregunta sobre {ticker}…", max_chars=config.AGENTE_MAX_CARACTERES) or rapida
    if pregunta:
        _responder(bd, token, ticker, modelo, clave_api, contexto, historial, pregunta, detector, guardar)
    if historial and st.button("Limpiar la conversación en pantalla"):
        st.session_state[clave_chat] = []
        st.rerun()

    with st.expander("Datos que recibe Gemini en cada pregunta"):
        st.code(f"{agente.INSTRUCCIONES}\nCONTEXTO:\n{contexto}", language="text")
    with st.expander("Cómo funcionan los guardrails"):
        _explicacion_guardrails(bd, detector)
    st.caption("Gemini solo explica los datos que calcula el dashboard. Es un ejercicio académico, no una "
               "recomendación de inversión.")
