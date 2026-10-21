"""Pestañas de cuenta del dashboard: registro e inicio de sesión, portafolio simulado y "Mis datos"."""

import json
import os

import pandas as pd
import streamlit as st

import base_datos
import config
import cuentas
import datos
import portafolio
import registro_agente

RUTA_POLITICA = config.RAIZ / "politica_datos.md"
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
TIPO_A_TEXTO = {"compra": "Compra", "venta": "Venta"}

AVISO_PRIVACIDAD = (
    "**Aviso de privacidad.** El equipo del proyecto {nombre_app}, estudiantes de la Escuela Colombiana de Ingeniería "
    "Julio Garavito, con domicilio en Bogotá y correo {correo}, es responsable de tus datos personales. Usamos tu "
    "alias, tu correo y tu contraseña (guardada solo como hash) para crear y proteger tu cuenta y llevar tu portafolio "
    "simulado con dinero ficticio, y para responder tus preguntas al agente: antes de enviarse a Google Gemini, "
    "las preguntas se anonimizan. No vendemos tus datos ni los usamos para publicidad. Puedes conocer, actualizar, "
    "rectificar y suprimir tus datos, o revocar esta autorización, desde la pestaña Mis datos o escribiendo a "
    "{correo}. La política completa está en el enlace de abajo."
)


def _rellenar(texto):
    for clave, valor in {
        "{nombre_app}": config.NOMBRE_APP,
        "{correo}": config.CORREO_CONTACTO,
        "{responsable}": config.RESPONSABLE_CONSULTAS,
        "{version}": config.VERSION_POLITICA,
        "{cuota}": str(config.AGENTE_CONSULTAS_DIARIAS),
    }.items():
        texto = texto.replace(clave, valor)
    return texto


def texto_politica():
    return _rellenar(RUTA_POLITICA.read_text(encoding="utf-8"))


def mostrar_politica(titulo="Política de tratamiento de datos personales"):
    with st.expander(titulo):
        st.markdown(texto_politica())


def reglas_del_juego():
    return (
        f"- Cada cuenta empieza con **{config.SALDO_INICIAL:,.0f} USD ficticios**. No hay depósitos, ni dinero real, "
        "ni premios, ni conexión con brókeres.\n"
        "- Solo se opera con acciones que tienen modelos entrenados.\n"
        "- Solo se opera con el mercado abierto (9:30 a 16:00, hora de Nueva York), al último precio de 2 minutos "
        f"disponible, que debe tener menos de {config.PRECIO_MAX_ANTIGUEDAD_MIN} minutos.\n"
        f"- Cada compra y cada venta cobra una comisión simulada del {config.COMISION_SIMULADA_PCT:g} %.\n"
        "- Se compran y venden acciones enteras. No se puede comprar sin saldo ni vender acciones que no se tienen.\n"
        "- Antes de confirmar se muestran la predicción de cada modelo y el consenso. La decisión es tuya: ni los "
        "modelos ni el agente operan por su cuenta.\n"
        "- Puedes reiniciar tu portafolio cuando quieras desde Mis datos.\n"
        "- Todas las cuentas tienen las mismas condiciones y no hay ranking de usuarios."
    )


def _hora(marca):
    marca = pd.Timestamp(marca)
    ny = marca.tz_convert(config.ZONA_MERCADO)
    bogota = marca.tz_convert(config.ZONA_USUARIO)
    return f"{DIAS[ny.weekday()]} {ny:%d/%m %H:%M} NY ({bogota:%H:%M} Bogotá)"


# ------------------------------------------------------------------ base de datos y sesión

def _secreto(nombre):
    try:
        if nombre in st.secrets:
            return str(st.secrets[nombre])
    except Exception:
        pass
    return os.environ.get(nombre)


@st.cache_resource(show_spinner="Conectando con la base de usuarios…")
def _abrir_base(url, clave):
    return base_datos.crear(url, clave)


def base():
    """(BaseDatos, None) o (None, mensaje de error)."""
    try:
        return _abrir_base(_secreto("DATABASE_URL"), _secreto("CLAVE_DATOS")), None
    except Exception as error:
        return None, f"No se pudo abrir la base de usuarios ({type(error).__name__}): {error}"


def _cerrar_sesion(aviso=None):
    """Cierra la sesión y borra de la pantalla todo lo de la cuenta, incluidas las conversaciones con el agente."""
    for clave in [k for k in st.session_state if str(k).startswith("chat_")] + ["sesion", "orden", "resultado_operacion"]:
        st.session_state.pop(clave, None)
    if aviso:
        st.session_state["aviso_sesion"] = aviso


def _abrir_sesion(usuario):
    st.session_state["sesion"] = {
        "token": usuario["token"],
        "alias": usuario["alias"],
        "vence": pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=config.MINUTOS_SESION),
    }


def sesion():
    """Usuario con sesión activa o None. Cada vez que se usa la app, la sesión se extiende MINUTOS_SESION."""
    actual = st.session_state.get("sesion")
    if not actual:
        return None
    ahora = pd.Timestamp.now(tz="UTC")
    if ahora > actual["vence"]:
        _cerrar_sesion(f"La sesión se cerró después de {config.MINUTOS_SESION} minutos sin actividad.")
        return None
    actual["vence"] = ahora + pd.Timedelta(minutes=config.MINUTOS_SESION)
    return actual


def barra_lateral(usuario):
    st.sidebar.divider()
    st.sidebar.header("Cuenta")
    if usuario:
        st.sidebar.caption(f"Sesión iniciada como **{usuario['alias']}**.")
        if st.sidebar.button("Cerrar sesión"):
            _cerrar_sesion()
            st.rerun()
    else:
        st.sidebar.caption("Para jugar con el portafolio simulado, inicia sesión o crea una cuenta en la pestaña "
                           "Portafolio simulado.")


# ------------------------------------------------------------------ registro e inicio de sesión

def _formularios(bd):
    aviso = st.session_state.pop("aviso_sesion", None)
    if aviso:
        st.info(aviso)
    st.markdown(f"Con una cuenta puedes comprar y vender acciones con **{config.SALDO_INICIAL:,.0f} USD ficticios**, "
                "ver cuánto habrías ganado o perdido y compararlo con comprar y mantener. Es un juego académico: "
                "nada de lo que se muestra es una recomendación de inversión.")
    modo = st.radio("Cuenta", ["Iniciar sesión", "Crear cuenta"], horizontal=True, label_visibility="collapsed",
                    key="modo_cuenta")
    if modo == "Iniciar sesión":
        with st.form("form_entrar"):
            correo = st.text_input("Correo")
            contrasena = st.text_input("Contraseña", type="password")
            if st.form_submit_button("Iniciar sesión", type="primary"):
                try:
                    usuario = cuentas.iniciar_sesion(bd, correo, contrasena)
                except cuentas.ErrorCuenta as error:
                    st.error(str(error))
                else:
                    _abrir_sesion(usuario)
                    st.rerun()
        return

    with st.form("form_registro"):
        alias = st.text_input("Alias", max_chars=30, help="Cómo te verás en la aplicación. No tiene que ser tu nombre.")
        correo = st.text_input("Correo", help="Sirve para iniciar sesión. Se guarda cifrado.")
        columna_1, columna_2 = st.columns(2)
        contrasena = columna_1.text_input("Contraseña", type="password",
                                          help=f"Al menos {config.CONTRASENA_MINIMA} caracteres, con letras y números.")
        confirmacion = columna_2.text_input("Repite la contraseña", type="password")
        st.markdown(_rellenar(AVISO_PRIVACIDAD))
        mostrar_politica("Leer la política de tratamiento de datos personales")
        mayor = st.checkbox("Confirmo que soy mayor de 18 años.", value=False)
        autoriza = st.checkbox(
            "Autorizo de forma previa, expresa e informada el tratamiento de mis datos personales para las finalidades "
            "descritas en la Política de tratamiento de datos personales, incluido su almacenamiento en servidores que "
            "pueden estar fuera de Colombia.", value=False)
        historial = st.checkbox("Quiero guardar el historial de mis preguntas al agente (puedo cambiarlo en cualquier "
                                "momento).", value=False)
        if st.form_submit_button("Crear cuenta", type="primary"):
            try:
                usuario = cuentas.registrar(bd, alias, correo, contrasena, confirmacion, mayor, autoriza, historial)
            except cuentas.ErrorCuenta as error:
                st.error(str(error))
            else:
                _abrir_sesion(usuario)
                st.session_state["resultado_operacion"] = (
                    f"Cuenta creada. Tienes {config.SALDO_INICIAL:,.0f} USD ficticios para empezar.")
                st.rerun()
    st.caption("Las dos primeras casillas son obligatorias. Se guarda la fecha y hora en que se marcaron, como prueba "
               "de la autorización.")


# ------------------------------------------------------------------ portafolio simulado

def _tabla_predicciones(tabla):
    return pd.DataFrame({
        "modelo": tabla["modelo"],
        "dirección": tabla["direccion"],
        "retorno predicho": tabla["retorno_pct"].map(lambda v: f"{v:+.3f}%"),
        "precio predicho": tabla["precio_predicho"].map(lambda v: f"{v:,.2f}"),
    })


def _resumen_orden(orden, estado_actual):
    monto = round(orden["cantidad"] * orden["precio"], 2)
    costo = portafolio.comision(monto)
    if orden["tipo"] == "compra":
        efectivo_despues = estado_actual["efectivo"] - monto - costo
        texto = f"Comprar **{orden['cantidad']}** acciones de **{orden['ticker']}**"
    else:
        efectivo_despues = estado_actual["efectivo"] + monto - costo
        texto = f"Vender **{orden['cantidad']}** acciones de **{orden['ticker']}**"
    return (f"{texto} a **{orden['precio']:,.2f} USD** (precio de {_hora(orden['marca'])}).  \n"
            f"Monto: {monto:,.2f} USD · comisión: {costo:,.2f} USD · efectivo después de la operación: "
            f"**{efectivo_despues:,.2f} USD**")


def _seccion_operar(bd, usuario, estado_actual, tickers, cotizacion_de, prediccion_de, acciones):
    st.markdown("#### Comprar o vender")
    abierto = datos.mercado_abierto()
    if not abierto:
        st.info(f"El mercado está cerrado. Se opera de lunes a viernes de 9:30 a 16:00, hora de Nueva York "
                f"({portafolio.horario_bogota()} en Bogotá).")
    columna_1, columna_2, columna_3 = st.columns([1, 1, 1])
    ticker = columna_1.selectbox("Acción", tickers, key="orden_ticker")
    tipo = columna_2.radio("Operación", list(portafolio.TIPOS), format_func=TIPO_A_TEXTO.get, horizontal=True,
                           key="orden_tipo")
    cantidad = int(columna_3.number_input("Acciones", min_value=1, max_value=config.CANTIDAD_MAXIMA_OPERACION,
                                          value=1, step=1, key="orden_cantidad"))
    precio, marca = cotizacion_de(ticker)
    if precio is None:
        st.warning(f"No hay precio de {ticker} en este momento.")
        return
    tienes = acciones.get(ticker, 0)
    st.caption(f"Último precio de {ticker}: **{precio:,.2f} USD** ({_hora(marca)}). Tienes {tienes} acciones y "
               f"alcanza para comprar {portafolio.maximo_comprable(estado_actual['efectivo'], precio)} con la comisión.")

    orden = st.session_state.get("orden")
    if orden and (orden["ticker"], orden["tipo"], orden["cantidad"]) != (ticker, tipo, cantidad):
        st.session_state.pop("orden", None)
        orden = None
    if st.button("Revisar la operación", disabled=not abierto):
        orden = {"ticker": ticker, "tipo": tipo, "cantidad": cantidad, "precio": precio,
                 "marca": pd.Timestamp(marca).isoformat()}
        st.session_state["orden"] = orden
    if not orden:
        return

    with st.container(border=True):
        st.markdown(_resumen_orden(orden, estado_actual))
        tabla, info, consenso = prediccion_de(orden["ticker"])
        if tabla is not None and not tabla.empty:
            st.markdown(f"Lo que predicen los modelos para la vela de {_hora(info['objetivo'])}, frente al cierre de "
                        f"{info['cierre_referencia']:,.2f}:")
            st.dataframe(_tabla_predicciones(tabla), hide_index=True)
            st.markdown(f"**Consenso:** {portafolio.texto_consenso(consenso)}")
        st.caption("Las predicciones no son una recomendación. En la pestaña Evaluación de modelos se puede ver qué tan "
                   "seguido aciertan frente a las líneas base.")
        confirmar, cancelar = st.columns(2)
        if confirmar.button("Confirmar la operación", type="primary"):
            precio_ahora, marca_ahora = cotizacion_de(orden["ticker"])
            if marca_ahora is not None and pd.Timestamp(marca_ahora).isoformat() != orden["marca"]:
                st.session_state["orden"] = {**orden, "precio": precio_ahora,
                                             "marca": pd.Timestamp(marca_ahora).isoformat()}
                st.session_state["resultado_operacion"] = ("El precio se actualizó mientras revisabas. Revisa la "
                                                           "operación con el precio nuevo y confirma otra vez.")
                st.rerun()
            try:
                resultado = portafolio.ejecutar(bd, usuario["token"], orden["tipo"], orden["ticker"], orden["cantidad"],
                                                precio_ahora, marca_ahora, portafolio.texto_consenso(consenso), tickers)
            except portafolio.OperacionRechazada as error:
                st.error(str(error))
            else:
                st.session_state.pop("orden", None)
                st.session_state["resultado_operacion"] = (
                    f"{TIPO_A_TEXTO[resultado['tipo']]} registrada: {resultado['cantidad']} acciones de "
                    f"{resultado['ticker']} a {resultado['precio']:,.2f} USD, con una comisión de "
                    f"{resultado['comision']:,.2f} USD. Efectivo: {resultado['efectivo']:,.2f} USD.")
                st.rerun()
        if cancelar.button("Cancelar"):
            st.session_state.pop("orden", None)
            st.rerun()


def _seccion_historial(operaciones):
    st.markdown("#### Historial de operaciones")
    if operaciones.empty:
        st.caption("Todavía no hay operaciones.")
        return
    vista = pd.DataFrame({
        "fecha": operaciones["fecha"].map(_hora),
        "acción": operaciones["ticker"],
        "operación": operaciones["tipo"].map(TIPO_A_TEXTO),
        "acciones": operaciones["cantidad"],
        "precio": operaciones["precio"],
        "monto": (operaciones["cantidad"] * operaciones["precio"]).round(2),
        "comisión": operaciones["comision"],
        "consenso de los modelos": operaciones["consenso"].fillna("—"),
    }).iloc[::-1]
    st.dataframe(vista, hide_index=True, column_config={
        "precio": st.column_config.NumberColumn(format="%.2f"),
        "monto": st.column_config.NumberColumn(format="%.2f"),
        "comisión": st.column_config.NumberColumn(format="%.2f"),
    })
    cuenta = portafolio.frente_al_consenso(operaciones)
    st.caption(f"De {len(operaciones)} operaciones, {cuenta['a favor']} siguieron al consenso de los modelos (comprar "
               f"cuando decía Sube o vender cuando decía Baja), {cuenta['en contra']} fueron en contra y "
               f"{cuenta['sin consenso']} se hicieron sin consenso.")


def pestana_portafolio(usuario, tickers, cotizacion_de, prediccion_de, series_de):
    bd, error = base()
    if error:
        st.error(error)
        return
    if not tickers:
        st.info("Todavía no hay acciones con modelos entrenados para operar.")
        return
    if not usuario:
        _formularios(bd)
        with st.expander("Reglas del juego"):
            st.markdown(reglas_del_juego())
        return

    estado_actual = portafolio.estado(bd, usuario["token"])
    if estado_actual is None:
        _cerrar_sesion("La cuenta ya no existe.")
        st.rerun()
    mensaje = st.session_state.pop("resultado_operacion", None)
    if mensaje:
        st.success(mensaje)

    operaciones = estado_actual["operaciones"]
    acciones_por_ticker = {}
    for operacion in operaciones.itertuples():
        signo = 1 if operacion.tipo == "compra" else -1
        acciones_por_ticker[operacion.ticker] = acciones_por_ticker.get(operacion.ticker, 0) + signo * operacion.cantidad
    precios = {t: cotizacion_de(t)[0] for t in set(tickers) | set(acciones_por_ticker)}
    tabla = portafolio.posiciones(operaciones, precios)
    valor_acciones = float(tabla["valor"].sum()) if not tabla.empty else 0.0
    total = estado_actual["efectivo"] + valor_acciones
    inicial = estado_actual["saldo_inicial"]

    st.subheader(f"Portafolio de {usuario['alias']}")
    columna_1, columna_2, columna_3, columna_4 = st.columns(4)
    columna_1.metric("Efectivo", f"{estado_actual['efectivo']:,.2f} USD")
    columna_2.metric("Valor de las acciones", f"{valor_acciones:,.2f} USD")
    columna_3.metric("Valor total", f"{total:,.2f} USD", f"{100 * (total / inicial - 1):+.2f}% desde el inicio")
    referencia_ticker = tickers[0]
    referencia = portafolio.comprar_y_mantener(inicial, estado_actual["inicio_en"], series_de(referencia_ticker),
                                               precios[referencia_ticker]) if precios.get(referencia_ticker) else None
    if referencia:
        columna_4.metric(f"Comprar y mantener {referencia_ticker}", f"{referencia['valor']:,.2f} USD",
                         f"{100 * (referencia['valor'] / inicial - 1):+.2f}%")
        diferencia = total - referencia["valor"]
        st.caption(f"Portafolio iniciado el {_hora(estado_actual['inicio_en'])}. Si en ese momento hubieras comprado "
                   f"{referencia['acciones']} acciones de {referencia_ticker} a {referencia['precio_inicial']:,.2f} USD "
                   f"y no hubieras vendido, hoy tendrías {referencia['valor']:,.2f} USD: tu portafolio va "
                   f"{abs(diferencia):,.2f} USD {'por encima' if diferencia >= 0 else 'por debajo'} de esa referencia.")
    else:
        columna_4.metric(f"Comprar y mantener {referencia_ticker}", "—")
        st.caption(f"Portafolio iniciado el {_hora(estado_actual['inicio_en'])}. La comparación con comprar y mantener "
                   "aparece cuando haya precios desde esa fecha.")

    recalculado = portafolio.efectivo_segun_operaciones(inicial, operaciones)
    if abs(recalculado - estado_actual["efectivo"]) < 0.01:
        st.caption("El efectivo coincide con la suma de las operaciones registradas.")
    else:
        st.error(f"El efectivo guardado ({estado_actual['efectivo']:,.2f}) no coincide con el que resulta de las "
                 f"operaciones ({recalculado:,.2f}). Hay que revisar la base de datos.")

    if not tabla.empty:
        st.dataframe(tabla, hide_index=True, column_config={
            columna: st.column_config.NumberColumn(format="%.2f")
            for columna in ("costo promedio", "precio actual", "valor", "ganancia", "ganancia %")
        })

    _seccion_operar(bd, usuario, estado_actual, tickers, cotizacion_de, prediccion_de, acciones_por_ticker)
    _seccion_historial(operaciones)
    with st.expander("Reglas del juego"):
        st.markdown(reglas_del_juego())


# ------------------------------------------------------------------ mis datos

def pestana_mis_datos(usuario):
    bd, error = base()
    if error:
        st.error(error)
        return
    if not usuario:
        st.info("Inicia sesión en la pestaña Portafolio simulado para ver y administrar tus datos. La política de "
                "tratamiento de datos está al final de la página.")
        return
    token = usuario["token"]
    titular = cuentas.datos_del_titular(bd, token)
    if titular is None:
        _cerrar_sesion("La cuenta ya no existe.")
        st.rerun()
    datos_usuario, datos_portafolio = titular["usuario"], titular["portafolio"]

    st.markdown("Estos son **todos** los datos que la aplicación guarda sobre ti. Puedes consultarlos, descargarlos, "
                "corregirlos o eliminarlos (Ley 1581 de 2012, artículo 8).")
    st.dataframe(pd.DataFrame([
        ("Alias", datos_usuario["alias"]),
        ("Correo", datos_usuario["correo"] or "—"),
        ("Contraseña", "Solo se guarda su hash bcrypt; no se puede ver."),
        ("Mayor de 18 años", "Sí" if datos_usuario["mayor_de_edad"] else "No"),
        ("Autorización del tratamiento", f"{_hora(datos_usuario['autorizacion_en'])}, política del "
                                         f"{datos_usuario['version_politica']}"),
        ("Historial de preguntas al agente", "Activado" if datos_usuario["historial_agente"] else "Desactivado"),
        ("Cuenta creada", _hora(datos_usuario["creado_en"])),
        ("Último acceso", _hora(datos_usuario["ultimo_acceso"]) if datos_usuario["ultimo_acceso"] else "—"),
        ("Efectivo ficticio", f"{datos_portafolio['efectivo']:,.2f} USD"),
        ("Inicio del portafolio", _hora(datos_portafolio["inicio_en"])),
        ("Operaciones registradas", str(len(titular["operaciones"]))),
        ("Preguntas al agente hoy", f"{registro_agente.consultas_hoy(bd, token)} de {config.AGENTE_CONSULTAS_DIARIAS}"),
        ("Mensajes guardados del agente", str(len(titular["historial_del_agente"]))),
    ], columns=["dato", "valor"]), hide_index=True)
    st.download_button("Descargar mis datos (JSON)",
                       json.dumps(titular, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                       file_name="mis_datos.json", mime="application/json")

    st.markdown("#### Corregir mis datos")
    columna_1, columna_2 = st.columns(2)
    with columna_1.form("form_alias"):
        nuevo_alias = st.text_input("Alias", value=datos_usuario["alias"], max_chars=30)
        if st.form_submit_button("Guardar alias"):
            try:
                st.session_state["sesion"]["alias"] = cuentas.actualizar_alias(bd, token, nuevo_alias)
            except cuentas.ErrorCuenta as error_alias:
                st.error(str(error_alias))
            else:
                st.rerun()
    with columna_2.form("form_correo"):
        nuevo_correo = st.text_input("Correo", value=datos_usuario["correo"] or "")
        if st.form_submit_button("Guardar correo"):
            try:
                cuentas.actualizar_correo(bd, token, nuevo_correo)
            except cuentas.ErrorCuenta as error_correo:
                st.error(str(error_correo))
            else:
                st.rerun()
    with st.form("form_contrasena", clear_on_submit=True):
        st.markdown("**Cambiar la contraseña**")
        columna_a, columna_b, columna_c = st.columns(3)
        actual = columna_a.text_input("Contraseña actual", type="password")
        nueva = columna_b.text_input("Contraseña nueva", type="password")
        repetida = columna_c.text_input("Repite la nueva", type="password")
        if st.form_submit_button("Cambiar la contraseña"):
            try:
                cuentas.cambiar_contrasena(bd, token, actual, nueva, repetida)
            except cuentas.ErrorCuenta as error_clave:
                st.error(str(error_clave))
            else:
                st.success("Contraseña cambiada.")

    st.markdown("#### Historial del agente")
    activo = st.toggle("Guardar el historial de mis preguntas al agente", value=datos_usuario["historial_agente"],
                       help="Viene desactivado. Mientras esté desactivado, las conversaciones no se guardan. Las "
                            "preguntas se guardan ya anonimizadas.")
    if activo != datos_usuario["historial_agente"]:
        cuentas.cambiar_historial_agente(bd, token, activo)
        st.rerun()
    if titular["historial_del_agente"]:
        if st.button("Borrar mi historial del agente"):
            registro_agente.borrar_historial(bd, token)
            for clave in [k for k in st.session_state if str(k).startswith("chat_")]:
                st.session_state.pop(clave, None)
            st.rerun()

    st.markdown("#### Reiniciar el portafolio")
    reinicio = st.checkbox(f"Entiendo que se borran mis operaciones y vuelvo a {config.SALDO_INICIAL:,.0f} USD "
                           "ficticios.", key="confirmar_reinicio")
    if st.button("Reiniciar el portafolio", disabled=not reinicio):
        portafolio.reiniciar(bd, token)
        st.session_state.pop("confirmar_reinicio", None)
        st.session_state["resultado_operacion"] = "Portafolio reiniciado."
        st.rerun()

    st.markdown("#### Revocar la autorización y eliminar la cuenta")
    st.caption("Se borran de inmediato tu cuenta, tu correo, tu portafolio, tus operaciones y tu historial del agente. "
               "No se puede deshacer.")
    eliminar = st.checkbox("Quiero revocar la autorización y eliminar mi cuenta.", key="confirmar_eliminar")
    if st.button("Eliminar mi cuenta", disabled=not eliminar):
        cuentas.eliminar_cuenta(bd, token)
        st.session_state.pop("confirmar_eliminar", None)
        _cerrar_sesion("La cuenta y todos sus datos se eliminaron.")
        st.rerun()

    st.caption(f"Consultas y reclamos: también puedes escribir a {config.CORREO_CONTACTO}. Las consultas se responden "
               "en máximo 10 días hábiles y los reclamos en 15. Base de datos en uso: " + bd.descripcion + ".")