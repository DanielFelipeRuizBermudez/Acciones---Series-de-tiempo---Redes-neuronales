"""Envío del código de recuperación por correo (SMTP con TLS).

Usa la cuenta de correo del proyecto. Con Gmail hay que activar la verificación en dos pasos y crear una contraseña de
aplicación; el usuario y esa contraseña van en los Secrets de Streamlit (SMTP_USUARIO y SMTP_CLAVE), nunca en el
repositorio. El correo solo lleva el código y su vencimiento: ningún otro dato de la cuenta.
"""

import smtplib
import ssl
from email.message import EmailMessage

import config

SERVIDOR_POR_DEFECTO = "smtp.gmail.com"
PUERTO_POR_DEFECTO = 465


def mensaje_codigo(remitente, destino, codigo, minutos):
    mensaje = EmailMessage()
    mensaje["Subject"] = f"Código para recuperar tu cuenta de {config.NOMBRE_APP}"
    mensaje["From"] = remitente
    mensaje["To"] = destino
    mensaje.set_content(
        f"Tu código para recuperar la cuenta es: {codigo}\n\n"
        f"Vence en {minutos} minutos y sirve una sola vez.\n\n"
        "Si no pediste recuperar tu cuenta, ignora este correo: tu contraseña no ha cambiado.\n\n"
        f"{config.NOMBRE_APP} · proyecto académico"
    )
    return mensaje


def enviar_codigo(destino, codigo, minutos, usuario, clave, servidor=SERVIDOR_POR_DEFECTO,
                  puerto=PUERTO_POR_DEFECTO):
    contexto = ssl.create_default_context()
    with smtplib.SMTP_SSL(servidor, int(puerto), context=contexto, timeout=20) as conexion:
        conexion.login(usuario, clave)
        conexion.send_message(mensaje_codigo(usuario, destino, codigo, minutos))
