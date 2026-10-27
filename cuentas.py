"""Cuentas de usuario: registro con autorización, inicio de sesión seguro y derechos del titular (Ley 1581).

- La contraseña se guarda solo como hash bcrypt.
- Tras INTENTOS_MAXIMOS intentos fallidos la cuenta se bloquea MINUTOS_BLOQUEO minutos. También cuentan los intentos
  fallidos al abrir los datos personales, al autorizar una solicitud o al ver datos compartidos.
- Los mensajes de error de inicio de sesión no dicen si el correo existe.
- Se guarda la fecha y hora de la autorización y la versión de la política aceptada, como prueba.
- El correo, el nombre, la dirección, la ciudad, el teléfono y la fecha de nacimiento van en un sobre cifrado con la
  llave pública del usuario. Solo su contraseña abre la llave privada que lo descifra (llaves.py).
"""

import math
import re
import secrets
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import bcrypt

import config
import llaves
from base_datos import a_texto, ahora_utc, de_texto, normalizar_correo

PATRON_CORREO = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
MENSAJE_CREDENCIALES = "Correo o contraseña incorrectos."
_HASH_DE_RELLENO = bcrypt.hashpw(b"relleno-para-igualar-tiempos", bcrypt.gensalt()).decode("ascii")
EDAD_MINIMA = 18

CAMPOS_PERSONALES = {
    "nombre": "Nombre completo",
    "correo": "Correo",
    "telefono": "Teléfono",
    "direccion": "Dirección",
    "ciudad": "Ciudad",
    "fecha_nacimiento": "Fecha de nacimiento",
}


class ErrorCuenta(Exception):
    """Error con un mensaje que se le puede mostrar al usuario."""


class CuentaBloqueada(ErrorCuenta):
    """La cuenta quedó bloqueada por intentos fallidos: hay que cerrar la sesión."""


def _hash(contrasena):
    return bcrypt.hashpw(contrasena.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def _coincide(contrasena, hash_guardado):
    clave = contrasena.encode("utf-8")
    if len(clave) > 72:
        return False
    return bcrypt.checkpw(clave, hash_guardado.encode("ascii"))


def _contexto_datos(token):
    return f"datos:{token}"


# ------------------------------------------------------------------ validaciones

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
        errores.append(f"La contraseña debe tener al menos {config.CONTRASENA_MINIMA} caracteres, con letras y "
                       "números.")
    if len(contrasena.encode("utf-8")) > 72:
        errores.append("La contraseña puede tener como máximo 72 bytes.")
    if contrasena != confirmacion:
        errores.append("Las contraseñas no coinciden.")
    return errores


def hoy_bogota():
    return datetime.now(ZoneInfo(config.ZONA_USUARIO)).date()


def edad(fecha_nacimiento, hoy=None):
    hoy = hoy or hoy_bogota()
    return hoy.year - fecha_nacimiento.year - ((hoy.month, hoy.day) < (fecha_nacimiento.month, fecha_nacimiento.day))


def _como_fecha(valor):
    if isinstance(valor, date):
        return valor
    try:
        return date.fromisoformat(str(valor))
    except ValueError:
        return None


def normalizar_personales(personales):
    """Datos personales sin espacios sobrantes, con la fecha como texto AAAA-MM-DD."""
    limpios = {campo: " ".join(str(personales.get(campo) or "").split()) for campo in CAMPOS_PERSONALES
               if campo != "fecha_nacimiento"}
    limpios["correo"] = normalizar_correo(limpios["correo"])
    fecha = _como_fecha(personales.get("fecha_nacimiento"))
    limpios["fecha_nacimiento"] = fecha.isoformat() if fecha else ""
    return limpios


def errores_personales(personales, hoy=None):
    """Errores de los datos personales ya normalizados."""
    errores = errores_correo(personales["correo"])
    if not 3 <= len(personales["nombre"]) <= 80 or not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]{2}", personales["nombre"]):
        errores.append("Escribe tu nombre completo (entre 3 y 80 caracteres).")
    if not 5 <= len(personales["direccion"]) <= 120:
        errores.append("Escribe tu dirección (entre 5 y 120 caracteres).")
    if not 2 <= len(personales["ciudad"]) <= 60 or not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", personales["ciudad"]):
        errores.append("Escribe tu ciudad.")
    digitos = re.sub(r"\D", "", personales["telefono"])
    if not re.fullmatch(r"[\d\s()+-]+", personales["telefono"] or "x") or not 7 <= len(digitos) <= 15:
        errores.append("El teléfono debe tener entre 7 y 15 dígitos.")
    fecha = _como_fecha(personales["fecha_nacimiento"])
    if fecha is None:
        errores.append("Escribe tu fecha de nacimiento.")
    elif edad(fecha, hoy) < EDAD_MINIMA:
        errores.append(f"La aplicación es solo para mayores de {EDAD_MINIMA} años.")
    elif edad(fecha, hoy) > 120:
        errores.append("Revisa la fecha de nacimiento.")
    return errores


# ------------------------------------------------------------------ llaves y sobres

def _guardar_llaves(cursor, token, par, ahora):
    cursor.ejecutar(
        "INSERT INTO llaves (token, publica, privada_cifrada, sal, nonce, iteraciones, creada_en) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (token, par["publica"], par["privada_cifrada"], par["sal"], par["nonce"], par["iteraciones"], ahora))


def _guardar_sobre(cursor, token, publica, personales, ahora, nuevo):
    sobre = llaves.cifrar_para(publica, personales, _contexto_datos(token))
    if nuevo:
        cursor.ejecutar("INSERT INTO datos_personales (token, sobre, actualizado_en) VALUES (?, ?, ?)",
                        (token, sobre, ahora))
    else:
        cursor.ejecutar("UPDATE datos_personales SET sobre = ?, actualizado_en = ? WHERE token = ?",
                        (sobre, ahora, token))


def leer_llaves(cursor, token):
    return cursor.ejecutar("SELECT publica, privada_cifrada, sal, nonce, iteraciones, creada_en FROM llaves "
                           "WHERE token = ?", (token,)).uno()


def tiene_llaves(bd, token):
    with bd.transaccion() as cursor:
        return cursor.ejecutar("SELECT 1 AS existe FROM llaves WHERE token = ?", (token,)).uno() is not None


def necesita_completar(bd, token):
    """True si la cuenta se creó antes de los datos personales cifrados y todavía no los ha completado."""
    return not tiene_llaves(bd, token)


# ------------------------------------------------------------------ registro e inicio de sesión

def registrar(bd, alias, contrasena, confirmacion, personales, autoriza, historial_agente=False):
    """Crea la cuenta, su par de llaves, su sobre de datos personales y su portafolio. Devuelve {'token', 'alias'}."""
    personales = normalizar_personales(personales)
    errores = errores_alias(alias) + errores_personales(personales) + errores_contrasena(contrasena, confirmacion)
    if not autoriza:
        errores.append("Para crear la cuenta hay que autorizar el tratamiento de datos personales.")
    if errores:
        raise ErrorCuenta(" ".join(errores))

    token = secrets.token_urlsafe(18)
    ahora = a_texto(ahora_utc())
    indice = bd.indice_correo(personales["correo"])
    hash_contrasena = _hash(contrasena)
    par = llaves.generar_par(contrasena)
    with bd.transaccion() as cursor:
        if cursor.ejecutar("SELECT 1 AS existe FROM correos WHERE indice = ?", (indice,)).uno():
            raise ErrorCuenta("Ya existe una cuenta con ese correo.")
        cursor.ejecutar(
            "INSERT INTO usuarios (token, alias, hash_contrasena, mayor_de_edad, autorizacion_en, version_politica, "
            "historial_agente, creado_en, ultimo_acceso, intentos_fallidos) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (token, alias.strip(), hash_contrasena, True, ahora, config.VERSION_POLITICA, bool(historial_agente),
             ahora, ahora),
        )
        cursor.ejecutar("INSERT INTO correos (token, indice, cifrado) VALUES (?, ?, ?)", (token, indice, ""))
        _guardar_llaves(cursor, token, par, ahora)
        _guardar_sobre(cursor, token, par["publica"], personales, ahora, nuevo=True)
        cursor.ejecutar("INSERT INTO portafolios (token, efectivo, saldo_inicial, inicio_en) VALUES (?, ?, ?, ?)",
                        (token, config.SALDO_INICIAL, config.SALDO_INICIAL, ahora))
    return {"token": token, "alias": alias.strip()}


def _registrar_fallo(cursor, token, intentos_previos, ahora):
    """Suma un intento fallido; al llegar a INTENTOS_MAXIMOS bloquea la cuenta. Devuelve (mensaje, bloqueada)."""
    intentos = int(intentos_previos) + 1
    if intentos >= config.INTENTOS_MAXIMOS:
        hasta = a_texto(ahora + timedelta(minutes=config.MINUTOS_BLOQUEO))
        cursor.ejecutar("UPDATE usuarios SET intentos_fallidos = 0, bloqueado_hasta = ? WHERE token = ?",
                        (hasta, token))
        return (f"Se bloqueó la cuenta por {config.MINUTOS_BLOQUEO} minutos después de {config.INTENTOS_MAXIMOS} "
                "intentos fallidos."), True
    cursor.ejecutar("UPDATE usuarios SET intentos_fallidos = ? WHERE token = ?", (intentos, token))
    return None, False


def _mensaje_bloqueo(bloqueado_hasta, ahora):
    minutos = math.ceil((de_texto(bloqueado_hasta) - ahora).total_seconds() / 60)
    return (f"La cuenta está bloqueada por {config.INTENTOS_MAXIMOS} intentos fallidos. "
            f"Se puede volver a intentar en {minutos} min.")


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
            mensaje = _mensaje_bloqueo(usuario["bloqueado_hasta"], ahora)
        elif _coincide(contrasena, usuario["hash_contrasena"]):
            cursor.ejecutar("UPDATE usuarios SET intentos_fallidos = 0, bloqueado_hasta = NULL, ultimo_acceso = ? "
                            "WHERE token = ?", (a_texto(ahora), usuario["token"]))
            resultado = {"token": usuario["token"], "alias": usuario["alias"]}
        else:
            bloqueo, _ = _registrar_fallo(cursor, usuario["token"], usuario["intentos_fallidos"], ahora)
            mensaje = bloqueo or mensaje
    if resultado is None:
        raise ErrorCuenta(mensaje)
    return resultado


def comprobar_contrasena(bd, token, contrasena, ahora=None):
    """Confirma la contraseña de un usuario con sesión iniciada; los fallos cuentan para el bloqueo."""
    ahora = ahora or ahora_utc()
    with bd.transaccion() as cursor:
        usuario = cursor.ejecutar("SELECT hash_contrasena, intentos_fallidos, bloqueado_hasta FROM usuarios "
                                  "WHERE token = ?" + bd.bloqueo, (token,)).uno()
        if usuario is None:
            raise CuentaBloqueada("La cuenta ya no existe.")
        if usuario["bloqueado_hasta"] and de_texto(usuario["bloqueado_hasta"]) > ahora:
            raise CuentaBloqueada(_mensaje_bloqueo(usuario["bloqueado_hasta"], ahora))
        if _coincide(contrasena, usuario["hash_contrasena"]):
            cursor.ejecutar("UPDATE usuarios SET intentos_fallidos = 0 WHERE token = ?", (token,))
            return
        mensaje, bloqueada = _registrar_fallo(cursor, token, usuario["intentos_fallidos"], ahora)
    if bloqueada:
        raise CuentaBloqueada(mensaje)
    raise ErrorCuenta("La contraseña no es correcta.")


def abrir_llave_privada(bd, token, contrasena):
    """Llave privada del usuario, después de comprobar su contraseña."""
    comprobar_contrasena(bd, token, contrasena)
    with bd.transaccion() as cursor:
        guardada = leer_llaves(cursor, token)
    if guardada is None:
        raise ErrorCuenta("Esta cuenta todavía no tiene llaves: primero hay que completar los datos personales.")
    try:
        return llaves.abrir_privada(guardada, contrasena)
    except llaves.LlaveNoAbre as error:
        raise ErrorCuenta("La llave privada no se pudo abrir con esa contraseña.") from error


# ------------------------------------------------------------------ datos personales

def completar_datos(bd, token, contrasena, personales, autoriza):
    """Cuentas creadas antes de los datos cifrados: genera las llaves, guarda el sobre y borra la copia Fernet."""
    personales = normalizar_personales(personales)
    errores = errores_personales(personales)
    if not autoriza:
        errores.append("Para continuar hay que autorizar el tratamiento de datos de la política actual.")
    if errores:
        raise ErrorCuenta(" ".join(errores))
    comprobar_contrasena(bd, token, contrasena)
    par = llaves.generar_par(contrasena)
    ahora = a_texto(ahora_utc())
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar("SELECT indice FROM correos WHERE token = ?" + bd.bloqueo, (token,)).uno()
        if fila is None or fila["indice"] != bd.indice_correo(personales["correo"]):
            raise ErrorCuenta("Escribe el mismo correo con el que inicias sesión.")
        if leer_llaves(cursor, token) is not None:
            raise ErrorCuenta("Los datos de esta cuenta ya están completos.")
        _guardar_llaves(cursor, token, par, ahora)
        _guardar_sobre(cursor, token, par["publica"], personales, ahora, nuevo=True)
        cursor.ejecutar("UPDATE correos SET cifrado = ? WHERE token = ?", ("", token))
        cursor.ejecutar("UPDATE usuarios SET mayor_de_edad = ?, autorizacion_en = ?, version_politica = ? "
                        "WHERE token = ?", (True, ahora, config.VERSION_POLITICA, token))


def correo_anterior(bd, token):
    """Correo de una cuenta creada antes de los datos cifrados (copia Fernet), o None."""
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar("SELECT cifrado FROM correos WHERE token = ?", (token,)).uno()
    return bd.descifrar(fila["cifrado"]) if fila and fila["cifrado"] else None


def abrir_datos_personales(bd, token, contrasena):
    """Datos personales descifrados con la llave privada del titular."""
    privada = abrir_llave_privada(bd, token, contrasena)
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar("SELECT sobre FROM datos_personales WHERE token = ?", (token,)).uno()
    if fila is None:
        raise ErrorCuenta("No hay datos personales guardados.")
    return llaves.abrir_sobre(privada, fila["sobre"], _contexto_datos(token))


def actualizar_datos_personales(bd, token, personales):
    """Reemplaza el sobre con los datos nuevos. Solo hace falta la llave pública: escribir no requiere poder leer."""
    personales = normalizar_personales(personales)
    errores = errores_personales(personales)
    if errores:
        raise ErrorCuenta(" ".join(errores))
    indice = bd.indice_correo(personales["correo"])
    ahora = a_texto(ahora_utc())
    with bd.transaccion() as cursor:
        otro = cursor.ejecutar("SELECT token FROM correos WHERE indice = ?", (indice,)).uno()
        if otro and otro["token"] != token:
            raise ErrorCuenta("Ese correo ya está registrado en otra cuenta.")
        guardada = leer_llaves(cursor, token)
        if guardada is None:
            raise ErrorCuenta("Esta cuenta todavía no tiene llaves: primero hay que completar los datos personales.")
        cursor.ejecutar("UPDATE correos SET indice = ? WHERE token = ?", (indice, token))
        _guardar_sobre(cursor, token, guardada["publica"], personales, ahora, nuevo=False)
    return personales


# ------------------------------------------------------------------ derechos del titular

def existe(bd, token):
    with bd.transaccion() as cursor:
        return cursor.ejecutar("SELECT 1 AS existe FROM usuarios WHERE token = ?", (token,)).uno() is not None


def datos_del_titular(bd, token):
    """Todo lo que la aplicación guarda sobre el usuario (derecho de acceso). Los datos personales siguen cifrados."""
    with bd.transaccion() as cursor:
        usuario = cursor.ejecutar(
            "SELECT alias, mayor_de_edad, autorizacion_en, version_politica, historial_agente, creado_en, "
            "ultimo_acceso FROM usuarios WHERE token = ?", (token,)).uno()
        if usuario is None:
            return None
        llave = cursor.ejecutar("SELECT publica, creada_en FROM llaves WHERE token = ?", (token,)).uno()
        sobre = cursor.ejecutar("SELECT actualizado_en FROM datos_personales WHERE token = ?", (token,)).uno()
        portafolio = cursor.ejecutar("SELECT efectivo, saldo_inicial, inicio_en FROM portafolios WHERE token = ?",
                                     (token,)).uno()
        operaciones = cursor.ejecutar(
            "SELECT fecha, ticker, tipo, cantidad, precio, comision, consenso FROM operaciones WHERE token = ? "
            "ORDER BY fecha, id", (token,)).todos()
        uso = cursor.ejecutar("SELECT dia, consultas FROM uso_agente WHERE token = ? ORDER BY dia", (token,)).todos()
        conversaciones = cursor.ejecutar("SELECT fecha, ticker, rol, texto FROM conversaciones WHERE token = ? "
                                         "ORDER BY id", (token,)).todos()
        solicitudes = cursor.ejecutar(
            "SELECT s.id, u.alias AS solicitante, s.motivo, s.campos, s.minutos, s.estado, s.creada_en, "
            "s.decidida_en, s.expira_en, s.campos_autorizados FROM solicitudes_acceso s "
            "JOIN usuarios u ON u.token = s.solicitante WHERE s.titular = ? ORDER BY s.id", (token,)).todos()
        accesos = cursor.ejecutar(
            "SELECT a.solicitud_id, a.visto_en FROM accesos_datos a JOIN solicitudes_acceso s "
            "ON s.id = a.solicitud_id WHERE s.titular = ? ORDER BY a.id", (token,)).todos()
    usuario["mayor_de_edad"] = bool(usuario["mayor_de_edad"])
    usuario["historial_agente"] = bool(usuario["historial_agente"])
    return {
        "usuario": usuario,
        "llave": ({"huella_llave_publica": llaves.huella(llave["publica"]), "creada_en": llave["creada_en"]}
                  if llave else None),
        "datos_personales": ({"estado": "cifrados con la llave pública del titular", "actualizados_en":
                              sobre["actualizado_en"]} if sobre else None),
        "portafolio": portafolio,
        "operaciones": operaciones,
        "preguntas_al_agente_por_dia": uso,
        "historial_del_agente": conversaciones,
        "solicitudes_de_acceso": solicitudes,
        "accesos_a_datos_compartidos": accesos,
    }


def actualizar_alias(bd, token, alias):
    errores = errores_alias(alias)
    if errores:
        raise ErrorCuenta(" ".join(errores))
    with bd.transaccion() as cursor:
        cursor.ejecutar("UPDATE usuarios SET alias = ? WHERE token = ?", (alias.strip(), token))
    return alias.strip()


def cambiar_contrasena(bd, token, actual, nueva, confirmacion):
    """Cambia el hash y vuelve a cifrar la llave privada con la contraseña nueva."""
    errores = errores_contrasena(nueva, confirmacion)
    if errores:
        raise ErrorCuenta(" ".join(errores))
    comprobar_contrasena(bd, token, actual)
    with bd.transaccion() as cursor:
        guardada = leer_llaves(cursor, token)
    protegida = None
    if guardada is not None:
        try:
            protegida = llaves.proteger_privada(llaves.abrir_privada(guardada, actual), nueva)
        except llaves.LlaveNoAbre as error:
            raise ErrorCuenta("La llave privada no se pudo abrir con la contraseña actual.") from error
    with bd.transaccion() as cursor:
        cursor.ejecutar("UPDATE usuarios SET hash_contrasena = ? WHERE token = ?", (_hash(nueva), token))
        if protegida is not None:
            cursor.ejecutar("UPDATE llaves SET privada_cifrada = ?, sal = ?, nonce = ?, iteraciones = ? "
                            "WHERE token = ?", (protegida["privada_cifrada"], protegida["sal"], protegida["nonce"],
                                                protegida["iteraciones"], token))


def historial_activo(bd, token):
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar("SELECT historial_agente FROM usuarios WHERE token = ?", (token,)).uno()
    return bool(fila["historial_agente"]) if fila else False


def cambiar_historial_agente(bd, token, activo):
    with bd.transaccion() as cursor:
        cursor.ejecutar("UPDATE usuarios SET historial_agente = ? WHERE token = ?", (bool(activo), token))


def eliminar_cuenta(bd, token):
    """Revoca la autorización: borra la cuenta, sus llaves, sus datos personales, sus solicitudes, el portafolio, las
    operaciones y el uso del agente."""
    with bd.transaccion() as cursor:
        cursor.ejecutar("DELETE FROM accesos_datos WHERE solicitud_id IN (SELECT id FROM solicitudes_acceso "
                        "WHERE titular = ? OR solicitante = ?)", (token, token))
        cursor.ejecutar("DELETE FROM solicitudes_acceso WHERE titular = ? OR solicitante = ?", (token, token))
        for tabla in ("datos_personales", "llaves", "conversaciones", "uso_agente", "operaciones", "portafolios",
                      "correos", "usuarios"):
            cursor.ejecutar(f"DELETE FROM {tabla} WHERE token = ?", (token,))