"""Cuentas de usuario: registro con autorización, inicio de sesión seguro y derechos del titular (Ley 1581).

- La contraseña se guarda solo como hash bcrypt.
- Tras INTENTOS_MAXIMOS intentos fallidos la cuenta se bloquea MINUTOS_BLOQUEO minutos.
- Los mensajes de error de inicio de sesión no dicen si el correo existe.
- Se guarda la fecha y hora de la autorización y la versión de la política aceptada, como prueba.
"""

import math
import re
import secrets
from datetime import timedelta

import bcrypt

import config
from base_datos import a_texto, ahora_utc, de_texto, normalizar_correo

PATRON_CORREO = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
MENSAJE_CREDENCIALES = "Correo o contraseña incorrectos."
_HASH_DE_RELLENO = bcrypt.hashpw(b"relleno-para-igualar-tiempos", bcrypt.gensalt()).decode("ascii")


class ErrorCuenta(Exception):
    """Error con un mensaje que se le puede mostrar al usuario."""


def _hash(contrasena):
    return bcrypt.hashpw(contrasena.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def _coincide(contrasena, hash_guardado):
    clave = contrasena.encode("utf-8")
    if len(clave) > 72:
        return False
    return bcrypt.checkpw(clave, hash_guardado.encode("ascii"))


def errores_alias(alias):
    alias = alias.strip()
    if not 3 <= len(alias) <= 30:
        return ["El alias debe tener entre 3 y 30 caracteres."]
    return []


def errores_correo(correo):
    return [] if PATRON_CORREO.match(normalizar_correo(correo)) else ["El correo no es válido."]


def errores_contrasena(contrasena, confirmacion):
    errores = []
    if (len(contrasena) < config.CONTRASENA_MINIMA or not re.search(r"[A-Za-z]", contrasena)
            or not re.search(r"\d", contrasena)):
        errores.append(f"La contraseña debe tener al menos {config.CONTRASENA_MINIMA} caracteres, con letras y números.")
    if len(contrasena.encode("utf-8")) > 72:
        errores.append("La contraseña puede tener como máximo 72 bytes.")
    if contrasena != confirmacion:
        errores.append("Las contraseñas no coinciden.")
    return errores


def registrar(bd, alias, correo, contrasena, confirmacion, mayor_de_edad, autoriza, historial_agente=False):
    """Crea la cuenta y su portafolio con el saldo inicial. Devuelve {'token', 'alias'}."""
    errores = errores_alias(alias) + errores_correo(correo) + errores_contrasena(contrasena, confirmacion)
    if not mayor_de_edad:
        errores.append("La aplicación es solo para mayores de 18 años.")
    if not autoriza:
        errores.append("Para crear la cuenta hay que autorizar el tratamiento de datos personales.")
    if errores:
        raise ErrorCuenta(" ".join(errores))

    token = secrets.token_urlsafe(18)
    ahora = a_texto(ahora_utc())
    indice = bd.indice_correo(correo)
    hash_contrasena = _hash(contrasena)
    with bd.transaccion() as cursor:
        if cursor.ejecutar("SELECT 1 AS existe FROM correos WHERE indice = ?", (indice,)).uno():
            raise ErrorCuenta("Ya existe una cuenta con ese correo.")
        cursor.ejecutar(
            "INSERT INTO usuarios (token, alias, hash_contrasena, mayor_de_edad, autorizacion_en, version_politica, "
            "historial_agente, creado_en, ultimo_acceso, intentos_fallidos) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (token, alias.strip(), hash_contrasena, True, ahora, config.VERSION_POLITICA, bool(historial_agente),
             ahora, ahora),
        )
        cursor.ejecutar("INSERT INTO correos (token, indice, cifrado) VALUES (?, ?, ?)",
                        (token, indice, bd.cifrar(normalizar_correo(correo))))
        cursor.ejecutar("INSERT INTO portafolios (token, efectivo, saldo_inicial, inicio_en) VALUES (?, ?, ?, ?)",
                        (token, config.SALDO_INICIAL, config.SALDO_INICIAL, ahora))
    return {"token": token, "alias": alias.strip()}


def iniciar_sesion(bd, correo, contrasena, ahora=None):
    """Devuelve {'token', 'alias'} o lanza ErrorCuenta. Cuenta los intentos fallidos y bloquea la cuenta."""
    ahora = ahora or ahora_utc()
    resultado, mensaje = None, MENSAJE_CREDENCIALES
    with bd.transaccion() as cursor:
        usuario = cursor.ejecutar(
            "SELECT u.token, u.alias, u.hash_contrasena, u.intentos_fallidos, u.bloqueado_hasta "
            "FROM correos c JOIN usuarios u ON u.token = c.token WHERE c.indice = ?" + bd.bloqueo,
            (bd.indice_correo(correo),),
        ).uno()
        if usuario is None:
            _coincide(contrasena, _HASH_DE_RELLENO)
        elif usuario["bloqueado_hasta"] and de_texto(usuario["bloqueado_hasta"]) > ahora:
            minutos = math.ceil((de_texto(usuario["bloqueado_hasta"]) - ahora).total_seconds() / 60)
            mensaje = (f"La cuenta está bloqueada por {config.INTENTOS_MAXIMOS} intentos fallidos. "
                       f"Se puede volver a intentar en {minutos} min.")
        elif _coincide(contrasena, usuario["hash_contrasena"]):
            cursor.ejecutar("UPDATE usuarios SET intentos_fallidos = 0, bloqueado_hasta = NULL, ultimo_acceso = ? "
                            "WHERE token = ?", (a_texto(ahora), usuario["token"]))
            resultado = {"token": usuario["token"], "alias": usuario["alias"]}
        else:
            intentos = int(usuario["intentos_fallidos"]) + 1
            if intentos >= config.INTENTOS_MAXIMOS:
                hasta = a_texto(ahora + timedelta(minutes=config.MINUTOS_BLOQUEO))
                cursor.ejecutar("UPDATE usuarios SET intentos_fallidos = 0, bloqueado_hasta = ? WHERE token = ?",
                                (hasta, usuario["token"]))
                mensaje = (f"Se bloqueó la cuenta por {config.MINUTOS_BLOQUEO} minutos después de "
                           f"{config.INTENTOS_MAXIMOS} intentos fallidos.")
            else:
                cursor.ejecutar("UPDATE usuarios SET intentos_fallidos = ? WHERE token = ?",
                                (intentos, usuario["token"]))
    if resultado is None:
        raise ErrorCuenta(mensaje)
    return resultado


def existe(bd, token):
    with bd.transaccion() as cursor:
        return cursor.ejecutar("SELECT 1 AS existe FROM usuarios WHERE token = ?", (token,)).uno() is not None


def datos_del_titular(bd, token):
    """Todo lo que la aplicación guarda sobre el usuario (derecho de acceso)."""
    with bd.transaccion() as cursor:
        usuario = cursor.ejecutar(
            "SELECT alias, mayor_de_edad, autorizacion_en, version_politica, historial_agente, creado_en, "
            "ultimo_acceso FROM usuarios WHERE token = ?", (token,)).uno()
        if usuario is None:
            return None
        correo = cursor.ejecutar("SELECT cifrado FROM correos WHERE token = ?", (token,)).uno()
        portafolio = cursor.ejecutar("SELECT efectivo, saldo_inicial, inicio_en FROM portafolios WHERE token = ?",
                                     (token,)).uno()
        operaciones = cursor.ejecutar(
            "SELECT fecha, ticker, tipo, cantidad, precio, comision, consenso FROM operaciones WHERE token = ? "
            "ORDER BY fecha, id", (token,)).todos()
        uso = cursor.ejecutar("SELECT dia, consultas FROM uso_agente WHERE token = ? ORDER BY dia", (token,)).todos()
        conversaciones = cursor.ejecutar("SELECT fecha, ticker, rol, texto FROM conversaciones WHERE token = ? "
                                         "ORDER BY id", (token,)).todos()
    usuario["mayor_de_edad"] = bool(usuario["mayor_de_edad"])
    usuario["historial_agente"] = bool(usuario["historial_agente"])
    usuario["correo"] = bd.descifrar(correo["cifrado"]) if correo else None
    return {"usuario": usuario, "portafolio": portafolio, "operaciones": operaciones,
            "preguntas_al_agente_por_dia": uso, "historial_del_agente": conversaciones}


def actualizar_alias(bd, token, alias):
    errores = errores_alias(alias)
    if errores:
        raise ErrorCuenta(" ".join(errores))
    with bd.transaccion() as cursor:
        cursor.ejecutar("UPDATE usuarios SET alias = ? WHERE token = ?", (alias.strip(), token))
    return alias.strip()


def actualizar_correo(bd, token, correo):
    errores = errores_correo(correo)
    if errores:
        raise ErrorCuenta(" ".join(errores))
    indice = bd.indice_correo(correo)
    with bd.transaccion() as cursor:
        otro = cursor.ejecutar("SELECT token FROM correos WHERE indice = ?", (indice,)).uno()
        if otro and otro["token"] != token:
            raise ErrorCuenta("Ese correo ya está registrado en otra cuenta.")
        cursor.ejecutar("UPDATE correos SET indice = ?, cifrado = ? WHERE token = ?",
                        (indice, bd.cifrar(normalizar_correo(correo)), token))


def cambiar_contrasena(bd, token, actual, nueva, confirmacion):
    errores = errores_contrasena(nueva, confirmacion)
    if errores:
        raise ErrorCuenta(" ".join(errores))
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar("SELECT hash_contrasena FROM usuarios WHERE token = ?", (token,)).uno()
        if fila is None or not _coincide(actual, fila["hash_contrasena"]):
            raise ErrorCuenta("La contraseña actual no es correcta.")
        cursor.ejecutar("UPDATE usuarios SET hash_contrasena = ? WHERE token = ?", (_hash(nueva), token))


def historial_activo(bd, token):
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar("SELECT historial_agente FROM usuarios WHERE token = ?", (token,)).uno()
    return bool(fila["historial_agente"]) if fila else False


def cambiar_historial_agente(bd, token, activo):
    with bd.transaccion() as cursor:
        cursor.ejecutar("UPDATE usuarios SET historial_agente = ? WHERE token = ?", (bool(activo), token))


def eliminar_cuenta(bd, token):
    """Revoca la autorización: borra la cuenta, el correo, el portafolio, las operaciones y el uso del agente."""
    with bd.transaccion() as cursor:
        for tabla in ("conversaciones", "uso_agente", "operaciones", "portafolios", "correos", "usuarios"):
            cursor.ejecutar(f"DELETE FROM {tabla} WHERE token = ?", (token,))
