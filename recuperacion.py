"""Recuperar la cuenta cuando se olvida la contraseña, con un código al correo y una llave de emergencia (break-glass).

1. Código al correo. El usuario escribe su correo; si coincide con el índice HMAC de una cuenta, se le envía a ese
   mismo correo un código de 6 dígitos que vence en MINUTOS_CODIGO minutos. Así demuestra que es el dueño de la cuenta.
   La aplicación no necesita descifrar ningún correo: lo escribe el propio usuario. Del código solo se guarda su hash
   bcrypt, y se descarta tras INTENTOS_CODIGO intentos fallidos.
2. Contraseña nueva. Con el código correcto, el usuario elige una contraseña nueva y vuelve a entrar a su cuenta. Sus
   datos personales siguen cifrados con su llave privada, que estaba protegida con la contraseña olvidada.
3. Llave de emergencia (opcional). Si el usuario la activó, existe una copia de su llave privada cifrada con la llave
   pública de emergencia. La llave privada de emergencia no está en el servidor: el administrador la guarda fuera de la
   aplicación, protegida con una frase. Para "romper el vidrio", la sube solo en ese momento; la aplicación abre la
   copia de rescate y la vuelve a cerrar con una llave temporal que solo abre la contraseña nueva del usuario. El
   administrador no ve la llave privada, ni la contraseña nueva, ni los datos, y la llave de emergencia no se guarda.
   La próxima vez que el usuario escribe su contraseña, la aplicación protege su llave privada original con ella.
4. Sin llave de emergencia, los datos personales cifrados con la contraseña olvidada no se pueden abrir: se borran y el
   usuario los vuelve a escribir al entrar.

Cada recuperación queda registrada, y el usuario la ve en Mis datos.
"""

import json
import secrets
from datetime import timedelta

import bcrypt

import cuentas
import llaves
from base_datos import a_texto, ahora_utc, de_texto

DIGITOS_CODIGO = 6
MINUTOS_CODIGO = 15
INTENTOS_CODIGO = 5
SEGUNDOS_ENTRE_CODIGOS = 60
MENSAJE_CODIGO_ENVIADO = (f"Si el correo está registrado, te llegó un código de {DIGITOS_CODIGO} dígitos. "
                          f"Vence en {MINUTOS_CODIGO} minutos.")
MENSAJE_CODIGO_INVALIDO = "El código no es válido o ya venció. Pide uno nuevo."
ESTADOS = {
    "pendiente": "Esperando la llave de emergencia",
    "lista": "Llave lista: se completa al escribir la contraseña nueva",
    "completada": "Datos recuperados",
    "sin_rescate": "Contraseña nueva sin llave de emergencia: los datos anteriores se borraron",
    "cancelada": "Cancelada",
}


class ErrorRecuperacion(Exception):
    """Error con un mensaje que se le puede mostrar al usuario."""


def _contexto_rescate(token):
    return f"rescate:{token}"


def _contexto_recuperacion(recuperacion_id):
    return f"recuperacion:{recuperacion_id}"


# ------------------------------------------------------------------ copia de rescate (opcional)

def sobre_rescate(token, privada, publica_emergencia):
    """(sobre, huella): copia de la llave privada cifrada con la llave pública de emergencia."""
    sobre = llaves.cifrar_para(publica_emergencia, {"privada": llaves.exportar_privada(privada)},
                               _contexto_rescate(token))
    return sobre, llaves.huella(publica_emergencia)


def guardar_rescate(cursor, token, privada, publica_emergencia, ahora):
    sobre, huella = sobre_rescate(token, privada, publica_emergencia)
    cursor.ejecutar("DELETE FROM llaves_rescate WHERE token = ?", (token,))
    cursor.ejecutar("INSERT INTO llaves_rescate (token, sobre, huella_emergencia, creada_en) VALUES (?, ?, ?, ?)",
                    (token, sobre, huella, ahora))


def rescate(bd, token):
    """{'huella_emergencia', 'creada_en'} si el usuario activó la copia de rescate, o None."""
    with bd.transaccion() as cursor:
        return cursor.ejecutar("SELECT huella_emergencia, creada_en FROM llaves_rescate WHERE token = ?",
                               (token,)).uno()


def activar_rescate(bd, token, contrasena, publica_emergencia):
    if not publica_emergencia:
        raise ErrorRecuperacion("La llave de emergencia no está configurada en esta aplicación.")
    privada = cuentas.abrir_llave_privada(bd, token, contrasena)
    with bd.transaccion() as cursor:
        guardar_rescate(cursor, token, privada, publica_emergencia, a_texto(ahora_utc()))


def desactivar_rescate(bd, token):
    with bd.transaccion() as cursor:
        cursor.ejecutar("DELETE FROM llaves_rescate WHERE token = ?", (token,))


# ------------------------------------------------------------------ código al correo y contraseña nueva

def pedir_codigo(bd, correo, enviar, ahora=None):
    """Envía un código al correo si está registrado. Responde igual exista o no, para no revelar qué correos hay.

    `enviar(destino, codigo)` manda el correo; si es None, el envío no está configurado.
    """
    if enviar is None:
        raise ErrorRecuperacion("El envío de correos no está configurado en esta aplicación. Escribe a "
                                "la dirección de contacto de la política.")
    if not cuentas.PATRON_CORREO.match(correo.strip()):
        raise ErrorRecuperacion("Escribe un correo válido.")
    ahora = ahora or ahora_utc()
    codigo = f"{secrets.randbelow(10 ** DIGITOS_CODIGO):0{DIGITOS_CODIGO}d}"
    with bd.transaccion() as cursor:
        cuenta = cursor.ejecutar("SELECT token FROM correos WHERE indice = ?", (bd.indice_correo(correo),)).uno()
        if cuenta is None:
            return
        previo = cursor.ejecutar("SELECT creado_en FROM codigos_recuperacion WHERE token = ?" + bd.bloqueo,
                                 (cuenta["token"],)).uno()
        if previo and (ahora - de_texto(previo["creado_en"])).total_seconds() < SEGUNDOS_ENTRE_CODIGOS:
            return
        cursor.ejecutar("DELETE FROM codigos_recuperacion WHERE token = ?", (cuenta["token"],))
        cursor.ejecutar(
            "INSERT INTO codigos_recuperacion (token, hash_codigo, creado_en, expira_en, intentos) "
            "VALUES (?, ?, ?, ?, 0)",
            (cuenta["token"], bcrypt.hashpw(codigo.encode("ascii"), bcrypt.gensalt()).decode("ascii"),
             a_texto(ahora), a_texto(ahora + timedelta(minutes=MINUTOS_CODIGO))))
    enviar(correo.strip(), codigo)


def _consumir_codigo(bd, correo, codigo, ahora):
    """Token de la cuenta si el código es correcto (y lo borra), o ErrorRecuperacion. Los fallos se registran."""
    error = None
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar(
            "SELECT c.token, k.hash_codigo, k.expira_en, k.intentos FROM correos c "
            "JOIN codigos_recuperacion k ON k.token = c.token WHERE c.indice = ?" + bd.bloqueo,
            (bd.indice_correo(correo),)).uno()
        if fila is None or de_texto(fila["expira_en"]) <= ahora:
            error = MENSAJE_CODIGO_INVALIDO
        elif not bcrypt.checkpw(codigo.strip().encode("ascii", "ignore"), fila["hash_codigo"].encode("ascii")):
            intentos = int(fila["intentos"]) + 1
            if intentos >= INTENTOS_CODIGO:
                cursor.ejecutar("DELETE FROM codigos_recuperacion WHERE token = ?", (fila["token"],))
                error = "Demasiados intentos con este código. Pide uno nuevo."
            else:
                cursor.ejecutar("UPDATE codigos_recuperacion SET intentos = ? WHERE token = ?",
                                (intentos, fila["token"]))
                restantes = INTENTOS_CODIGO - intentos
                error = (f"El código no es correcto. Te {'queda' if restantes == 1 else 'quedan'} {restantes} "
                         f"{'intento' if restantes == 1 else 'intentos'}.")
        else:
            cursor.ejecutar("DELETE FROM codigos_recuperacion WHERE token = ?", (fila["token"],))
            token = fila["token"]
    if error:
        raise ErrorRecuperacion(error)
    return token


def restablecer(bd, correo, codigo, nueva, confirmacion, ahora=None):
    """Cambia la contraseña con el código del correo. Devuelve 'pendiente' (hay llave de emergencia) o 'sin_rescate'."""
    errores = cuentas.errores_contrasena(nueva, confirmacion)
    if errores:
        raise ErrorRecuperacion(" ".join(errores))
    ahora = ahora or ahora_utc()
    token = _consumir_codigo(bd, correo, codigo, ahora)
    with bd.transaccion() as cursor:
        con_rescate = cursor.ejecutar("SELECT 1 AS existe FROM llaves_rescate WHERE token = ?", (token,)).uno()
        con_llaves = cursor.ejecutar("SELECT 1 AS existe FROM llaves WHERE token = ?", (token,)).uno()
    temporal = llaves.generar_par(nueva) if con_rescate and con_llaves else None
    marca = a_texto(ahora)
    with bd.transaccion() as cursor:
        cursor.ejecutar("UPDATE usuarios SET hash_contrasena = ?, intentos_fallidos = 0, bloqueado_hasta = NULL "
                        "WHERE token = ?", (cuentas.hash_contrasena(nueva), token))
        cursor.ejecutar("UPDATE recuperaciones SET estado = 'cancelada', llave_temporal = NULL, sobre_llave = NULL "
                        "WHERE token = ? AND estado IN ('pendiente', 'lista')", (token,))
        if temporal is not None:
            cursor.ejecutar("INSERT INTO recuperaciones (token, estado, creada_en, llave_temporal) "
                            "VALUES (?, 'pendiente', ?, ?)", (token, marca, json.dumps(temporal)))
            return "pendiente"
        cursor.ejecutar("DELETE FROM datos_personales WHERE token = ?", (token,))
        cursor.ejecutar("DELETE FROM llaves_rescate WHERE token = ?", (token,))
        cursor.ejecutar("DELETE FROM llaves WHERE token = ?", (token,))
        cursor.ejecutar("INSERT INTO recuperaciones (token, estado, creada_en, completada_en) "
                        "VALUES (?, 'sin_rescate', ?, ?)", (token, marca, marca))
    return "sin_rescate"


# ------------------------------------------------------------------ romper el vidrio (administrador)

def pendientes(bd):
    """Recuperaciones que esperan la llave de emergencia (sin datos personales: alias y fechas)."""
    with bd.transaccion() as cursor:
        return cursor.ejecutar(
            "SELECT r.id, u.alias, r.creada_en, s.huella_emergencia FROM recuperaciones r "
            "JOIN usuarios u ON u.token = r.token JOIN llaves_rescate s ON s.token = r.token "
            "WHERE r.estado = 'pendiente' ORDER BY r.id").todos()


def romper_vidrio(bd, recuperacion_id, pem_emergencia, frase, ahora=None):
    """Vuelve a cerrar la llave privada del usuario para su contraseña nueva, con la llave de emergencia.

    La copia de rescate se abre y su contenido pasa directo a un sobre nuevo para la llave temporal del usuario: la
    llave privada del usuario no se muestra ni se guarda en claro, y la de emergencia no se guarda.
    """
    try:
        emergencia = llaves.cargar_llave_emergencia(pem_emergencia, frase)
    except llaves.LlaveNoAbre as error:
        raise ErrorRecuperacion(str(error)) from error
    huella = llaves.huella(llaves.publica_pem(emergencia))
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar(
            "SELECT r.token, r.llave_temporal, s.sobre, s.huella_emergencia FROM recuperaciones r "
            "JOIN llaves_rescate s ON s.token = r.token WHERE r.id = ? AND r.estado = 'pendiente'",
            (recuperacion_id,)).uno()
    if fila is None:
        raise ErrorRecuperacion("La recuperación ya no está pendiente.")
    if fila["huella_emergencia"] != huella:
        raise ErrorRecuperacion("Esa no es la llave de emergencia con la que se guardó la copia de esta cuenta.")
    try:
        contenido = llaves.abrir_sobre(emergencia, fila["sobre"], _contexto_rescate(fila["token"]))
    except llaves.LlaveNoAbre as error:
        raise ErrorRecuperacion("La copia de rescate no se pudo abrir con esta llave.") from error
    publica_temporal = json.loads(fila["llave_temporal"])["publica"]
    sobre_llave = llaves.cifrar_para(publica_temporal, contenido, _contexto_recuperacion(recuperacion_id))
    with bd.transaccion() as cursor:
        cursor.ejecutar("UPDATE recuperaciones SET estado = 'lista', vidrio_roto_en = ?, sobre_llave = ? "
                        "WHERE id = ? AND estado = 'pendiente'",
                        (a_texto(ahora or ahora_utc()), sobre_llave, recuperacion_id))
        if cursor.filas_afectadas != 1:
            raise ErrorRecuperacion("La recuperación cambió mientras tanto. Recarga la página.")


# ------------------------------------------------------------------ el usuario termina la recuperación

def en_curso(bd, token):
    with bd.transaccion() as cursor:
        return cursor.ejecutar("SELECT 1 AS existe FROM recuperaciones WHERE token = ? AND estado IN "
                               "('pendiente', 'lista')", (token,)).uno() is not None


def terminar(bd, token, contrasena, ahora=None):
    """Si el administrador ya rompió el vidrio, protege la llave privada original con la contraseña nueva.

    Devuelve la llave privada, o None si no hay una recuperación lista para esta contraseña.
    """
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar("SELECT id, llave_temporal, sobre_llave FROM recuperaciones WHERE token = ? "
                               "AND estado = 'lista' ORDER BY id DESC", (token,)).uno()
        guardada = cuentas.leer_llaves(cursor, token)
    if fila is None or guardada is None:
        return None
    try:
        temporal = llaves.abrir_privada(json.loads(fila["llave_temporal"]), contrasena)
        contenido = llaves.abrir_sobre(temporal, fila["sobre_llave"], _contexto_recuperacion(fila["id"]))
    except llaves.LlaveNoAbre:
        return None
    privada = llaves.importar_privada(contenido["privada"])
    if llaves.publica_pem(privada) != guardada["publica"]:
        return None
    protegida = llaves.proteger_privada(privada, contrasena)
    with bd.transaccion() as cursor:
        cursor.ejecutar("UPDATE llaves SET privada_cifrada = ?, sal = ?, nonce = ?, iteraciones = ? WHERE token = ?",
                        (protegida["privada_cifrada"], protegida["sal"], protegida["nonce"], protegida["iteraciones"],
                         token))
        cursor.ejecutar("UPDATE recuperaciones SET estado = 'completada', completada_en = ?, llave_temporal = NULL, "
                        "sobre_llave = NULL WHERE id = ?", (a_texto(ahora or ahora_utc()), fila["id"]))
    return privada


def del_titular(bd, token):
    with bd.transaccion() as cursor:
        return cursor.ejecutar("SELECT id, estado, creada_en, vidrio_roto_en, completada_en FROM recuperaciones "
                               "WHERE token = ? ORDER BY id", (token,)).todos()
