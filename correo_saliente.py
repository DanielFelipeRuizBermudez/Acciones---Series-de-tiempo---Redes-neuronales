"""Correos de la aplicación (SMTP con TLS): el código de recuperación y el aviso de inicio de sesión.

Usa la cuenta de correo del proyecto. Con Gmail hay que activar la verificación en dos pasos y crear una contraseña de
aplicación; el usuario y esa contraseña van en los Secrets de Streamlit (SMTP_USUARIO y SMTP_CLAVE), nunca en el
repositorio. Cada correo lleva solo lo necesario: el código y su vencimiento, o la fecha y la hora del inicio de sesión.
"""

import html
import smtplib
import ssl
from email.message import EmailMessage

import config

SERVIDOR_POR_DEFECTO = "smtp.gmail.com"
PUERTO_POR_DEFECTO = 465
ROJO = "#C8102E"
NEGRO = "#111111"


def _html(titulo, parrafos, con_logo):
    """Cuerpo HTML sencillo, en tablas para que se vea bien en los clientes de correo."""
    logo = (f'<img src="cid:logo" alt="{html.escape(config.NOMBRE_APP)}" width="180" '
            'style="display:block;border:0;height:auto;">' if con_logo
            else f'<span style="font-size:28px;font-weight:700;letter-spacing:3px;color:{NEGRO};">'
                 f'{html.escape(config.NOMBRE_APP)}</span>')
    cuerpo = "".join(f'<p style="margin:0 0 14px 0;">{p}</p>' for p in parrafos)
    return f"""<!DOCTYPE html>
<html lang="es"><body style="margin:0;padding:0;background:#f4f4f4;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f4;padding:24px 0;">
<tr><td align="center">
<table role="presentation" width="520" cellpadding="0" cellspacing="0"
       style="background:#ffffff;border-top:4px solid {ROJO};font-family:Arial,Helvetica,sans-serif;color:{NEGRO};">
<tr><td style="padding:24px 28px 8px 28px;">{logo}</td></tr>
<tr><td style="padding:8px 28px 0 28px;font-size:20px;font-weight:700;">{html.escape(titulo)}</td></tr>
<tr><td style="padding:16px 28px 8px 28px;font-size:15px;line-height:1.5;">{cuerpo}</td></tr>
<tr><td style="padding:12px 28px 24px 28px;font-size:12px;color:#666666;border-top:1px solid #eeeeee;">
{html.escape(config.NOMBRE_APP)} · proyecto académico. Este correo se envía automáticamente; no lo respondas.
</td></tr>
</table>
</td></tr>
</table>
</body></html>"""


def _armar(remitente, destino, asunto, titulo, parrafos_texto, parrafos_html=None):
    mensaje = EmailMessage()
    mensaje["Subject"] = asunto
    mensaje["From"] = f"{config.NOMBRE_APP} <{remitente}>"
    mensaje["To"] = destino
    mensaje.set_content("\n\n".join(parrafos_texto + [f"{config.NOMBRE_APP} · proyecto académico"]))
    logo = config.LOGO_CORREO.read_bytes() if config.LOGO_CORREO.exists() else None
    parrafos = parrafos_html or [html.escape(p) for p in parrafos_texto]
    mensaje.add_alternative(_html(titulo, parrafos, logo is not None), subtype="html")
    if logo is not None:
        mensaje.get_payload()[1].add_related(logo, maintype="image", subtype="png", cid="<logo>",
                                             filename="logo.png")
    return mensaje


def mensaje_codigo(remitente, destino, codigo, minutos):
    texto = [
        f"Tu código para recuperar la cuenta es: {codigo}",
        f"Vence en {minutos} minutos y sirve una sola vez.",
        "Si no pediste recuperar tu cuenta, ignora este correo: tu contraseña no ha cambiado.",
    ]
    html_parrafos = [
        "Tu código para recuperar la cuenta es:",
        f'<span style="font-size:32px;font-weight:700;letter-spacing:8px;color:{ROJO};">{html.escape(codigo)}</span>',
        html.escape(texto[1]),
        html.escape(texto[2]),
    ]
    return _armar(remitente, destino, f"Código para recuperar tu cuenta de {config.NOMBRE_APP}",
                  "Recuperar tu cuenta", texto, html_parrafos)


def mensaje_inicio_sesion(remitente, destino, cuando):
    """`cuando`: fecha y hora del inicio de sesión ya escritas para mostrarse."""
    texto = [
        f"Se inició sesión en tu cuenta de {config.NOMBRE_APP} el {cuando}.",
        "Si fuiste tú, no tienes que hacer nada.",
        "Si no fuiste tú, cambia tu contraseña en Mis datos o usa la opción «Olvidé mi contraseña» para elegir una "
        "nueva.",
    ]
    return _armar(remitente, destino, f"Nuevo inicio de sesión en {config.NOMBRE_APP}",
                  "Nuevo inicio de sesión", texto)


def _enviar(mensaje, usuario, clave, servidor, puerto):
    contexto = ssl.create_default_context()
    with smtplib.SMTP_SSL(servidor, int(puerto), context=contexto, timeout=20) as conexion:
        conexion.login(usuario, clave)
        conexion.send_message(mensaje)


def enviar_codigo(destino, codigo, minutos, usuario, clave, servidor=SERVIDOR_POR_DEFECTO,
                  puerto=PUERTO_POR_DEFECTO):
    _enviar(mensaje_codigo(usuario, destino, codigo, minutos), usuario, clave, servidor, puerto)


def enviar_aviso_inicio(destino, cuando, usuario, clave, servidor=SERVIDOR_POR_DEFECTO,
                        puerto=PUERTO_POR_DEFECTO):
    _enviar(mensaje_inicio_sesion(usuario, destino, cuando), usuario, clave, servidor, puerto)