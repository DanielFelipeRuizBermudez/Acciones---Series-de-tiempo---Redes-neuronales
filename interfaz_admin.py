"""Panel del administrador: pedir acceso a los datos personales de un usuario y verlos solo si él lo autoriza, y romper
el vidrio cuando un usuario recupera su cuenta.

El administrador es la cuenta cuyo correo está en CORREO_ADMIN (Secrets de Streamlit). No tiene ningún permiso para
leer datos: solo puede pedir. Lo que el titular comparte llega cifrado con la llave pública del administrador y se
abre con la contraseña del administrador mientras la autorización esté vigente. Para una recuperación, el
administrador sube la llave de emergencia solo en ese momento; la aplicación vuelve a cerrar la llave del usuario
para su contraseña nueva sin mostrarle nada.
"""

import pandas as pd
import streamlit as st

import cuentas
import interfaz_cuenta as ui
import recuperacion
import solicitudes


def _formulario_solicitud(bd, token):
    usuarios = solicitudes.usuarios_con_datos(bd, token)
    if not usuarios:
        st.caption("Todavía no hay otras cuentas con datos personales.")
        return
    nombres = {u["token"]: f"{u['alias']} · cuenta creada el {ui._hora(u['creado_en'])}" for u in usuarios}
    with st.form("form_pedir_acceso", clear_on_submit=True):
        titular = st.selectbox("Cuenta", list(nombres), format_func=nombres.get)
        campos = st.multiselect("Datos que necesitas", list(cuentas.CAMPOS_PERSONALES),
                                format_func=cuentas.CAMPOS_PERSONALES.get,
                                help="Pide solo los datos necesarios para el motivo.")
        motivo = st.text_area("Motivo", max_chars=solicitudes.MOTIVO_MAXIMO,
                              help="El titular lo lee antes de decidir.")
        minutos = st.selectbox("Durante", list(solicitudes.DURACIONES), index=1,
                               format_func=solicitudes.DURACIONES.get)
        if st.form_submit_button("Enviar la solicitud", type="primary"):
            try:
                solicitudes.crear(bd, token, titular, campos, motivo, minutos)
            except solicitudes.ErrorSolicitud as error:
                st.error(str(error))
            else:
                st.success("Solicitud enviada. El titular la verá al iniciar sesión.")


def _vigentes(bd, token, filas):
    for fila in [f for f in filas if f["estado"] == "aprobada"]:
        with st.container(border=True):
            st.markdown(f"**{ui._sin_formato(fila['alias_titular'])}** autorizó "
                        f"**{solicitudes.nombres_campos(fila['campos_autorizados'])}** hasta "
                        f"{ui._hora(fila['expira_en'])}.")
            with st.form(f"form_ver_{fila['id']}", clear_on_submit=True):
                contrasena = st.text_input("Tu contraseña (abre tu llave privada)", type="password",
                                           key=f"ver_{fila['id']}_contrasena")
                columna_1, columna_2 = st.columns(2)
                ver = columna_1.form_submit_button("Ver los datos", type="primary")
                terminar = columna_2.form_submit_button("Terminar el acceso")
            if ver:
                try:
                    with st.spinner("Abriendo la copia compartida…"):
                        datos, expira = solicitudes.ver(bd, token, fila["id"], contrasena)
                except cuentas.ErrorCuenta as error:
                    ui._bloqueada(error)
                except solicitudes.ErrorSolicitud as error:
                    st.error(str(error))
                else:
                    st.dataframe(pd.DataFrame([(cuentas.CAMPOS_PERSONALES[c], v) for c, v in datos.items()],
                                              columns=["dato", "valor"]), hide_index=True)
                    st.caption(f"Se muestran solo ahora; la consulta quedó registrada y el titular la ve. "
                               f"Disponible hasta {ui._hora(expira)}.")
            if terminar:
                try:
                    solicitudes.cancelar(bd, token, fila["id"])
                except solicitudes.ErrorSolicitud as error:
                    st.error(str(error))
                else:
                    st.rerun()


def _recuperaciones(bd, token):
    st.markdown("#### Administración: recuperaciones con la llave de emergencia")
    filas = recuperacion.pendientes(bd)
    if not filas:
        st.caption("No hay recuperaciones esperando la llave de emergencia.")
        return
    st.caption("Romper el vidrio: el usuario ya demostró con el código de su correo que es el dueño de la cuenta y eligió "
               "una contraseña nueva. Sube el archivo de la llave de emergencia y escribe su frase: la aplicación vuelve "
               "a cerrar la llave privada del usuario para su contraseña nueva. No ves la llave del usuario, ni su "
               "contraseña, ni sus datos, y la llave de emergencia no se guarda. Queda registrado y el usuario lo ve.")
    opciones = {f["id"]: f"{f['alias']} · pedida el {ui._hora(f['creada_en'])} · llave {f['huella_emergencia'][:19]}"
                for f in filas}
    with st.form("form_romper_vidrio", clear_on_submit=True):
        elegida = st.selectbox("Recuperación", list(opciones), format_func=opciones.get)
        archivo = st.file_uploader("Archivo de la llave de emergencia (.pem)", type=["pem"])
        frase = st.text_input("Frase de la llave de emergencia", type="password")
        contrasena = st.text_input("Tu contraseña de administrador", type="password")
        if st.form_submit_button("Romper el vidrio", type="primary"):
            try:
                cuentas.comprobar_contrasena(bd, token, contrasena)
                if archivo is None:
                    raise recuperacion.ErrorRecuperacion("Sube el archivo de la llave de emergencia.")
                with st.spinner("Cerrando la llave del usuario para su contraseña nueva…"):
                    recuperacion.romper_vidrio(bd, elegida, archivo.getvalue(), frase)
            except cuentas.ErrorCuenta as error:
                ui._bloqueada(error)
            except recuperacion.ErrorRecuperacion as error:
                st.error(str(error))
            else:
                st.success("Listo: el usuario recupera sus datos la próxima vez que escriba su contraseña.")


def panel(bd, usuario):
    token = usuario["token"]
    _recuperaciones(bd, token)
    st.markdown("#### Administración: acceso a datos personales")
    st.caption("Como administrador no puedes leer los datos de nadie por tu cuenta. Pides acceso con un motivo; el "
               "titular decide qué datos comparte y por cuánto tiempo; la copia llega cifrada con tu llave pública y "
               "cada consulta queda registrada para el titular.")
    _formulario_solicitud(bd, token)
    filas = solicitudes.del_solicitante(bd, token)
    if not filas:
        return
    _vigentes(bd, token, filas)
    for fila in [f for f in filas if f["estado"] == "pendiente"]:
        if st.button(f"Cancelar la solicitud a {fila['alias_titular']}", key=f"cancelar_{fila['id']}"):
            try:
                solicitudes.cancelar(bd, token, fila["id"])
            except solicitudes.ErrorSolicitud as error:
                st.error(str(error))
            else:
                st.rerun()
    st.dataframe(pd.DataFrame({
        "solicitud": [ui._hora(f["creada_en"]) for f in filas],
        "cuenta": [f["alias_titular"] for f in filas],
        "datos pedidos": [solicitudes.nombres_campos(f["campos"]) for f in filas],
        "datos autorizados": [solicitudes.nombres_campos(f["campos_autorizados"]) or "—" for f in filas],
        "motivo": [f["motivo"] for f in filas],
        "estado": [solicitudes.ESTADOS[f["estado"]] for f in filas],
        "vence": [ui._hora(f["expira_en"]) for f in filas],
        "consultas": [f["consultas"] for f in filas],
    }), hide_index=True)