"""Solicitudes de acceso a los datos personales: el usuario gobierna sus datos.

1. El administrador pide ver algunos datos de un usuario, con un motivo y por un tiempo.
2. El titular ve la solicitud y decide: la rechaza, o la autoriza con su contraseña, eligiendo qué campos comparte y
   por cuánto tiempo.
3. Al autorizar, su llave privada abre su sobre y crea uno nuevo, solo con esos campos, cifrado con la llave pública del
   administrador. Nada más puede abrir ese sobre.
4. Mientras la autorización esté vigente, el administrador lo abre con su propia contraseña. Cada consulta queda
   registrada y el titular la ve.
5. Al vencer, o si el titular la revoca antes, el sobre compartido se borra.
"""

import json
from datetime import timedelta

import cuentas
import llaves
from base_datos import a_texto, ahora_utc

DURACIONES = {15: "15 minutos", 60: "1 hora", 1440: "24 horas"}
ESTADOS = {
    "pendiente": "Pendiente",
    "aprobada": "Autorizada",
    "rechazada": "Rechazada",
    "revocada": "Revocada por el titular",
    "expirada": "Vencida",
    "cancelada": "Cancelada por el solicitante",
}
MOTIVO_MINIMO, MOTIVO_MAXIMO = 15, 300


class ErrorSolicitud(Exception):
    """Error con un mensaje que se le puede mostrar al usuario."""


def _contexto(solicitud_id):
    return f"solicitud:{solicitud_id}"


def nombres_campos(campos):
    return ", ".join(cuentas.CAMPOS_PERSONALES.get(campo, campo) for campo in campos)


def es_admin(bd, token, correo_admin):
    """True si la cuenta de `token` es la del correo definido como administrador en los Secrets."""
    if not token or not correo_admin:
        return False
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar("SELECT 1 AS existe FROM correos WHERE token = ? AND indice = ?",
                               (token, bd.indice_correo(correo_admin))).uno()
    return fila is not None


def purgar_vencidas(bd, ahora=None):
    """Las autorizaciones vencidas pasan a 'expirada' y su sobre compartido se borra."""
    ahora = a_texto(ahora or ahora_utc())
    with bd.transaccion() as cursor:
        cursor.ejecutar("UPDATE solicitudes_acceso SET estado = 'expirada', sobre = NULL "
                        "WHERE estado = 'aprobada' AND expira_en <= ?", (ahora,))


def usuarios_con_datos(bd, excluir):
    """Cuentas que tienen datos personales cifrados (alias y fecha de creación, nada más)."""
    with bd.transaccion() as cursor:
        return cursor.ejecutar(
            "SELECT u.token, u.alias, u.creado_en FROM usuarios u JOIN datos_personales d ON d.token = u.token "
            "WHERE u.token <> ? ORDER BY u.alias", (excluir,)).todos()


def crear(bd, solicitante, titular, campos, motivo, minutos):
    campos = [campo for campo in cuentas.CAMPOS_PERSONALES if campo in set(campos)]
    motivo = " ".join((motivo or "").split())
    errores = []
    if titular == solicitante:
        errores.append("No se puede pedir acceso a los datos propios.")
    if not campos:
        errores.append("Elige al menos un dato.")
    if not MOTIVO_MINIMO <= len(motivo) <= MOTIVO_MAXIMO:
        errores.append(f"El motivo debe tener entre {MOTIVO_MINIMO} y {MOTIVO_MAXIMO} caracteres.")
    if minutos not in DURACIONES:
        errores.append("Elige una duración válida.")
    if errores:
        raise ErrorSolicitud(" ".join(errores))
    purgar_vencidas(bd)
    with bd.transaccion() as cursor:
        if cuentas.leer_llaves(cursor, solicitante) is None:
            raise ErrorSolicitud("Tu cuenta necesita sus propias llaves: primero completa tus datos personales.")
        if cursor.ejecutar("SELECT 1 AS existe FROM datos_personales WHERE token = ?", (titular,)).uno() is None:
            raise ErrorSolicitud("Esa cuenta no tiene datos personales guardados.")
        abierta = cursor.ejecutar(
            "SELECT 1 AS existe FROM solicitudes_acceso WHERE titular = ? AND solicitante = ? "
            "AND estado IN ('pendiente', 'aprobada')", (titular, solicitante)).uno()
        if abierta:
            raise ErrorSolicitud("Ya hay una solicitud pendiente o vigente para esa cuenta.")
        cursor.ejecutar(
            "INSERT INTO solicitudes_acceso (titular, solicitante, motivo, campos, minutos, estado, creada_en) "
            "VALUES (?, ?, ?, ?, ?, 'pendiente', ?)",
            (titular, solicitante, motivo, json.dumps(campos), int(minutos), a_texto(ahora_utc())))


def _filas(cursor, condicion, parametros):
    filas = cursor.ejecutar(
        "SELECT s.id, s.titular, s.solicitante, t.alias AS alias_titular, p.alias AS alias_solicitante, s.motivo, "
        "s.campos, s.minutos, s.estado, s.creada_en, s.decidida_en, s.expira_en, s.campos_autorizados, "
        "(SELECT COUNT(*) FROM accesos_datos a WHERE a.solicitud_id = s.id) AS consultas, "
        "(SELECT MAX(a.visto_en) FROM accesos_datos a WHERE a.solicitud_id = s.id) AS ultima_consulta "
        "FROM solicitudes_acceso s JOIN usuarios t ON t.token = s.titular JOIN usuarios p ON p.token = s.solicitante "
        f"WHERE {condicion} ORDER BY s.id DESC", parametros).todos()
    for fila in filas:
        fila["campos"] = json.loads(fila["campos"])
        fila["campos_autorizados"] = json.loads(fila["campos_autorizados"]) if fila["campos_autorizados"] else []
        fila["consultas"] = int(fila["consultas"] or 0)
    return filas


def del_titular(bd, titular):
    purgar_vencidas(bd)
    with bd.transaccion() as cursor:
        return _filas(cursor, "s.titular = ?", (titular,))


def del_solicitante(bd, solicitante):
    purgar_vencidas(bd)
    with bd.transaccion() as cursor:
        return _filas(cursor, "s.solicitante = ?", (solicitante,))


def pendientes(bd, titular):
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar("SELECT COUNT(*) AS n FROM solicitudes_acceso WHERE titular = ? "
                               "AND estado = 'pendiente'", (titular,)).uno()
    return int(fila["n"]) if fila else 0


def autorizar(bd, titular, solicitud_id, contrasena, campos, minutos, ahora=None):
    """El titular abre su sobre y comparte solo `campos`, durante `minutos`, con la llave pública del solicitante."""
    if minutos not in DURACIONES:
        raise ErrorSolicitud("Elige una duración válida.")
    with bd.transaccion() as cursor:
        solicitud = cursor.ejecutar("SELECT solicitante, campos FROM solicitudes_acceso WHERE id = ? AND titular = ? "
                                    "AND estado = 'pendiente'", (solicitud_id, titular)).uno()
    if solicitud is None:
        raise ErrorSolicitud("La solicitud ya no está pendiente.")
    pedidos = json.loads(solicitud["campos"])
    campos = [campo for campo in pedidos if campo in set(campos)]
    if not campos:
        raise ErrorSolicitud("Elige al menos un dato para compartir, o rechaza la solicitud.")

    datos = cuentas.abrir_datos_personales(bd, titular, contrasena)
    ahora = ahora or ahora_utc()
    with bd.transaccion() as cursor:
        destino = cuentas.leer_llaves(cursor, solicitud["solicitante"])
        if destino is None:
            raise ErrorSolicitud("La cuenta que pidió los datos ya no tiene llaves.")
        sobre = llaves.cifrar_para(destino["publica"], {campo: datos.get(campo, "") for campo in campos},
                                   _contexto(solicitud_id))
        cursor.ejecutar(
            "UPDATE solicitudes_acceso SET estado = 'aprobada', decidida_en = ?, expira_en = ?, "
            "campos_autorizados = ?, sobre = ? WHERE id = ? AND titular = ? AND estado = 'pendiente'",
            (a_texto(ahora), a_texto(ahora + timedelta(minutes=minutos)), json.dumps(campos), sobre, solicitud_id,
             titular))
        if cursor.filas_afectadas != 1:
            raise ErrorSolicitud("La solicitud ya no está pendiente.")
    return campos


def _cerrar(bd, condicion, parametros, estado):
    with bd.transaccion() as cursor:
        cursor.ejecutar(f"UPDATE solicitudes_acceso SET estado = ?, decidida_en = COALESCE(decidida_en, ?), "
                        f"sobre = NULL WHERE {condicion}", (estado, a_texto(ahora_utc())) + tuple(parametros))
        if cursor.filas_afectadas != 1:
            raise ErrorSolicitud("La solicitud cambió mientras tanto. Recarga la página.")


def rechazar(bd, titular, solicitud_id):
    _cerrar(bd, "id = ? AND titular = ? AND estado = 'pendiente'", (solicitud_id, titular), "rechazada")


def revocar(bd, titular, solicitud_id):
    """El titular termina antes de tiempo una autorización vigente; el sobre compartido se borra."""
    _cerrar(bd, "id = ? AND titular = ? AND estado = 'aprobada'", (solicitud_id, titular), "revocada")


def cancelar(bd, solicitante, solicitud_id):
    _cerrar(bd, "id = ? AND solicitante = ? AND estado IN ('pendiente', 'aprobada')", (solicitud_id, solicitante),
            "cancelada")


def ver(bd, solicitante, solicitud_id, contrasena, ahora=None):
    """Datos compartidos, abiertos con la llave privada del solicitante. Cada consulta queda registrada."""
    purgar_vencidas(bd, ahora)
    with bd.transaccion() as cursor:
        solicitud = cursor.ejecutar(
            "SELECT sobre, expira_en, campos_autorizados FROM solicitudes_acceso WHERE id = ? AND solicitante = ? "
            "AND estado = 'aprobada'", (solicitud_id, solicitante)).uno()
    if solicitud is None or not solicitud["sobre"]:
        raise ErrorSolicitud("La autorización ya no está vigente.")
    privada = cuentas.abrir_llave_privada(bd, solicitante, contrasena)
    datos = llaves.abrir_sobre(privada, solicitud["sobre"], _contexto(solicitud_id))
    with bd.transaccion() as cursor:
        cursor.ejecutar("INSERT INTO accesos_datos (solicitud_id, visto_en) VALUES (?, ?)",
                        (solicitud_id, a_texto(ahora or ahora_utc())))
    return datos, solicitud["expira_en"]
