"""Pestaña Ética: dónde entra cada principio en la aplicación y qué controles lo cumplen."""

import pandas as pd
import streamlit as st

PRINCIPIOS = {
    "Beneficencia": {
        "color": "#1baf7a",
        "idea": "Hacer un bien concreto: enseñar a leer modelos y mercados sin arriesgar dinero.",
        "controles": [
            ("Explicación de cada modelo y de la serie dentro de la app", "Prophet, Análisis de la serie y Evaluación de modelos"),
            ("Portafolio con 10.000 USD ficticios para aprender a invertir sin riesgo", "Portafolio simulado"),
            ("Comparación del portafolio con comprar y mantener y con el consenso de los modelos", "Portafolio simulado"),
            ("Agente que explica en lenguaje claro los datos que calcula la app", "Agente"),
        ],
    },
    "No maleficencia": {
        "color": "#d03b3b",
        "idea": "No hacer daño: proteger los datos y evitar que la app induzca a decisiones dañinas.",
        "controles": [
            ("Sin dinero real ni conexión con corredores", "Portafolio simulado"),
            ("Bloqueo de recomendaciones personales de inversión en las respuestas del agente", "Agente"),
            ("Reemplazo de cifras que no salen de los datos de la app (contra alucinaciones)", "Agente"),
            ("Bloqueo de inyección de instrucciones y de preguntas de más de 500 caracteres", "Agente"),
            ("Anonimización de nombres, cédulas, correos, teléfonos, cuentas, direcciones y fechas antes de enviar a "
             "Gemini (Presidio y expresiones regulares)", "Agente"),
            ("Instrucciones del agente: usar solo los datos de la app y no cambiar de papel", "Agente"),
            ("LiteLLM configurado para no registrar el contenido de los mensajes", "Agente"),
            ("Contraseñas guardadas solo como hash bcrypt", "Mis datos"),
            ("Datos personales cifrados con la llave pública de cada usuario (RSA-3072 y AES-256-GCM)", "Mis datos"),
            ("Llave privada protegida con la contraseña (PBKDF2 de 600.000 iteraciones y AES-256-GCM)", "Mis datos"),
            ("Índice HMAC del correo para iniciar sesión sin descifrar nada", "Portafolio simulado"),
            ("Bloqueo de 15 minutos tras 5 intentos fallidos y cierre de sesión a los 30 minutos", "Toda la app"),
            ("Los datos personales abiertos en pantalla se ocultan solos a los 5 minutos", "Mis datos"),
            ("Validación en el servidor de cada operación del portafolio", "Portafolio simulado"),
            ("Claves solo en los Secrets del servidor, fuera del repositorio, y conexión HTTPS", "Política, sección 10"),
            ("Confirmación explícita antes de reiniciar el portafolio o eliminar la cuenta", "Mis datos"),
            ("Mayoría de edad comprobada con la fecha de nacimiento", "Crear cuenta"),
            ("Recuperación con un código de 6 dígitos al correo: solo se guarda su hash, vence en 15 minutos, se anula "
             "tras 5 intentos y la respuesta no revela si el correo existe", "Portafolio simulado → Olvidé mi contraseña"),
            ("Llave de emergencia (break-glass) fuera de la app: usarla exige el archivo, su frase y la contraseña del "
             "administrador; no se guarda, y el administrador no ve la llave, la contraseña ni los datos del usuario",
             "Mis datos (administrador)"),
        ],
    },
    "Autonomía": {
        "color": "#2a78d6",
        "idea": "El usuario decide: sobre sus datos, sobre quién los ve y sobre sus inversiones simuladas.",
        "controles": [
            ("Autorización previa, expresa e informada, con prueba (fecha, hora y versión de la política)", "Crear cuenta"),
            ("Datos mínimos, cada uno con una finalidad; no se piden cédula ni datos sensibles", "Crear cuenta y política"),
            ("Historial del agente opcional y apagado por defecto", "Crear cuenta y Mis datos"),
            ("Ver, descargar en JSON y corregir los datos propios", "Mis datos"),
            ("El titular decide cada solicitud de acceso: qué datos comparte y por cuánto tiempo", "Mis datos"),
            ("Revocar un permiso en cualquier momento; la copia compartida se borra", "Mis datos"),
            ("Revocar la autorización eliminando la cuenta, con sus llaves, datos y solicitudes", "Mis datos"),
            ("La decisión de comprar o vender siempre es del usuario", "Portafolio simulado"),
            ("Copia de rescate de la llave opcional: el titular la activa o la quita cuando quiera", "Crear cuenta y Mis datos"),
            ("Aceptación expresa de cada versión nueva de la política, con fecha", "Mis datos"),
        ],
    },
    "Justicia": {
        "color": "#eda100",
        "idea": "Las mismas reglas para todos, sin privilegios ni usos de los datos en contra del usuario.",
        "controles": [
            ("Acceso gratuito con las mismas condiciones: 10.000 USD, comisión de 0,1% y 20 preguntas por día",
             "Portafolio simulado y Agente"),
            ("Los datos no se venden ni se usan para publicidad o para perfilar al usuario", "Política, sección 3"),
            ("Sin ranking público entre usuarios", "Política, sección 3"),
            ("Interfaz en español con las horas de Bogotá y de Nueva York", "Toda la app"),
            ("El administrador no tiene privilegios para leer datos: pide permiso, y para una recuperación solo vuelve "
             "a cerrar la llave del usuario", "Mis datos (administrador)"),
        ],
    },
    "Transparencia": {
        "color": "#9085e9",
        "idea": "Que el usuario sepa qué hace la app, qué tan bien funciona y qué pasa con sus datos.",
        "controles": [
            ("Política de tratamiento al pie de la app y aviso de privacidad en el registro", "Pie de la app y Crear cuenta"),
            ("Acierto de cada modelo con margen de error al 95% y frente a líneas base", "Evaluación de modelos"),
            ("Avisos de que nada de lo mostrado es una recomendación de inversión", "Pie de la app y Agente"),
            ("Vista de los datos exactos que recibe Gemini", "Agente"),
            ("Explicación de los guardrails y aviso cada vez que se reemplazan datos u ocultan cifras", "Agente"),
            ("Auditoría anónima de cada consulta al agente", "Agente"),
            ("Registro de cada vez que el administrador abre datos compartidos, visible para el titular", "Mis datos"),
            ("Motivo obligatorio en cada solicitud de acceso", "Mis datos"),
            ("Huella de la llave pública del usuario visible en Mis datos", "Mis datos"),
            ("Registro de cada recuperación de la cuenta y del uso de la llave de emergencia, visible para el titular",
             "Mis datos"),
            ("Aviso de cambio de política con las novedades, en la barra lateral y en Mis datos", "Barra lateral y Mis datos"),
        ],
    },
}

NORMAS = [
    ("Constitución Política, art. 15 (habeas data)", "Conocer, actualizar y rectificar la información propia", "Mis datos"),
    ("Ley 1581 de 2012, art. 4", "Principios: finalidad, libertad, transparencia, acceso restringido, seguridad",
     "Cifrado con la llave del titular, datos mínimos, política"),
    ("Ley 1581 de 2012, arts. 8 y 9", "Derechos del titular y autorización previa, expresa e informada",
     "Casilla de autorización, Mis datos y solicitudes"),
    ("Ley 1581 de 2012, arts. 14 y 15", "Consultas en 10 días hábiles y reclamos en 15", "Política, sección 9"),
    ("Ley 1581 de 2012, art. 26", "Transferencia internacional con nivel adecuado o consentimiento expreso",
     "Casilla de autorización que menciona servidores fuera de Colombia"),
    ("Decreto 1377 de 2013 (Decreto 1074 de 2015, sección 2.2.2.25)",
     "Contenido de la política y del aviso; datos pertinentes para la finalidad; comunicar los cambios sustanciales",
     "Aviso de privacidad, política y aceptación de cada versión nueva"),
    ("Circular Externa 002 de 2024, SIC", "Datos personales en sistemas de IA: necesidad, seguridad e información",
     "Guardrails, anonimización y auditoría del agente"),
    ("Decreto 661 de 2018", "La asesoría de inversión solo la prestan entidades vigiladas",
     "Bloqueo de recomendaciones personales"),
]


def diagrama():
    """Arquitectura de la app con los principios que entran en cada parte."""
    def etiqueta(titulo, lineas, principios):
        chips = " ".join(f'<FONT COLOR="{PRINCIPIOS[p]["color"]}"><B>■</B></FONT>' for p in principios)
        cuerpo = "".join(f"<BR/>{linea}" for linea in lineas)
        return f'<<B>{titulo}</B>{cuerpo}<BR/>{chips}>'

    nodos = {
        "usuario": ("Usuario", ["navegador"], []),
        "front": ("Front: Streamlit", ["HTTPS, 11 pestañas", "aviso de privacidad y política"],
                  ["Transparencia", "Beneficencia"]),
        "modelos": ("Modelos", ["AR, ARIMA, Prophet", "RNN, LSTM, CNN", "evaluación con margen de error"],
                    ["Beneficencia", "Transparencia"]),
        "cuentas": ("Cuentas y privacidad", ["llaves RSA + AES por usuario", "solicitudes con permiso",
                                             "recuperación y break-glass"],
                    ["No maleficencia", "Autonomía", "Justicia", "Transparencia"]),
        "portafolio": ("Portafolio simulado", ["10.000 USD ficticios", "la decisión es del usuario"],
                       ["Beneficencia", "No maleficencia", "Autonomía", "Justicia"]),
        "agente": ("Agente", ["guardrails de entrada y salida", "LiteLLM"],
                   ["No maleficencia", "Transparencia", "Justicia"]),
        "neon": ("Neon (Postgres)", ["solo datos cifrados", "token aleatorio"], ["No maleficencia"]),
        "yahoo": ("Yahoo Finance", ["precios públicos"], []),
        "gemini": ("Google Gemini", ["solo preguntas anonimizadas"], ["No maleficencia"]),
        "gmail": ("Gmail", ["solo el código de recuperación"], ["No maleficencia"]),
        "emergencia": ("Llave de emergencia", ["fuera de la app", "archivo + frase del admin"],
                       ["No maleficencia", "Autonomía"]),
    }
    lineas = ['digraph G {', 'rankdir=LR; bgcolor="white"; nodesep=0.35; ranksep=0.6;',
              'node [shape=box, style="rounded", fontname="Helvetica", fontsize=11, color="#8a8a8a", margin="0.15,0.08"];',
              'edge [color="#8a8a8a", fontname="Helvetica", fontsize=9];']
    for clave, (titulo, texto, principios) in nodos.items():
        lineas.append(f'{clave} [label={etiqueta(titulo, texto, principios)}];')
    lineas.append('subgraph cluster_backend { label="Backend (Python)"; fontname="Helvetica"; fontsize=11; '
                  'style="rounded,dashed"; color="#b0b0b0"; modelos; cuentas; portafolio; agente; }')
    for origen, destino, texto in [("usuario", "front", ""), ("front", "modelos", ""), ("front", "cuentas", ""),
                                   ("front", "portafolio", ""), ("front", "agente", ""),
                                   ("modelos", "yahoo", "descarga"), ("cuentas", "neon", "TLS"),
                                   ("portafolio", "neon", "TLS"), ("agente", "gemini", "anonimizado"),
                                   ("cuentas", "gmail", "código"), ("emergencia", "cuentas", "solo al romper el vidrio")]:
        lineas.append(f'{origen} -> {destino} [label="{texto}"];')
    lineas.append("}")
    return "\n".join(lineas)


def pestana():
    total = sum(len(p["controles"]) for p in PRINCIPIOS.values())
    st.markdown(f"La aplicación tiene **{total} controles** repartidos en los cinco principios. La **privacidad** no es "
                "un principio aparte: atraviesa tres. **Proteger** los datos es no maleficencia, que el usuario "
                "**decida** sobre ellos es autonomía y que **sepa** qué pasa con ellos es transparencia; además, que el "
                "administrador no tenga privilegios es justicia.")
    columnas = st.columns(len(PRINCIPIOS))
    for columna, (nombre, principio) in zip(columnas, PRINCIPIOS.items()):
        columna.metric(nombre, len(principio["controles"]))

    st.markdown("**Dónde entra cada principio en la arquitectura.** Los cuadros de color indican los principios de "
                "cada parte.")
    st.graphviz_chart(diagrama())
    st.markdown(" · ".join(f'<span style="color:{p["color"]}">■</span> {nombre}' for nombre, p in PRINCIPIOS.items()),
                unsafe_allow_html=True)

    for nombre, principio in PRINCIPIOS.items():
        with st.expander(f"{nombre}: {len(principio['controles'])} controles", expanded=False):
            st.markdown(principio["idea"])
            st.dataframe(pd.DataFrame(principio["controles"], columns=["control", "dónde se ve en la app"]),
                         hide_index=True)

    st.markdown("**Normas que enmarcan la aplicación**")
    st.dataframe(pd.DataFrame(NORMAS, columns=["norma", "qué exige", "dónde se cumple"]), hide_index=True)
    st.caption("Proyecto académico: la aplicación no presta asesoría de inversión y el portafolio usa solo dinero "
               "ficticio.")
