"""Uso del agente: cuota diaria por usuario, historial opcional de conversaciones y auditoría anónima.

- La cuota se cuenta por token y por día (hora de Bogotá).
- El historial solo se guarda si el usuario lo activó, y siempre con las preguntas ya anonimizadas.
- La auditoría guarda fecha, modelo, resultado, tipos de datos detectados y motivo de un bloqueo: no guarda el token
  ni el texto de las preguntas o respuestas.
"""

import json

import pandas as pd

import config
from base_datos import a_texto, ahora_utc

RESULTADOS = ("respondida", "bloqueada_entrada", "bloqueada_salida", "cifras_ocultas", "cuota_agotada", "error")


def hoy():
    return pd.Timestamp.now(tz=config.ZONA_USUARIO).date().isoformat()


def consultas_hoy(bd, token, dia=None):
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar("SELECT consultas FROM uso_agente WHERE token = ? AND dia = ?",
                               (token, dia or hoy())).uno()
    return int(fila["consultas"]) if fila else 0


def consumir_cuota(bd, token, limite=config.AGENTE_CONSULTAS_DIARIAS, dia=None):
    """Descuenta una pregunta de la cuota del día. Devuelve (permitida, preguntas usadas hoy)."""
    dia = dia or hoy()
    with bd.transaccion() as cursor:
        fila = cursor.ejecutar("SELECT consultas FROM uso_agente WHERE token = ? AND dia = ?" + bd.bloqueo,
                               (token, dia)).uno()
        usadas = int(fila["consultas"]) if fila else 0
        if usadas >= limite:
            return False, usadas
        if fila:
            cursor.ejecutar("UPDATE uso_agente SET consultas = ? WHERE token = ? AND dia = ?", (usadas + 1, token, dia))
        else:
            cursor.ejecutar("INSERT INTO uso_agente (token, dia, consultas) VALUES (?, ?, 1)", (token, dia))
    return True, usadas + 1


def guardar_mensajes(bd, token, ticker, mensajes):
    """Guarda mensajes [{'rol', 'texto'}] del historial. El texto del usuario ya debe venir anonimizado."""
    fecha = a_texto(ahora_utc())
    with bd.transaccion() as cursor:
        for mensaje in mensajes:
            cursor.ejecutar("INSERT INTO conversaciones (token, fecha, ticker, rol, texto) VALUES (?, ?, ?, ?, ?)",
                            (token, fecha, ticker, mensaje["rol"], mensaje["texto"]))


def historial(bd, token, ticker=None, limite=None):
    """Mensajes guardados, del más antiguo al más reciente (los últimos `limite` si se indica)."""
    consulta = "SELECT fecha, ticker, rol, texto FROM conversaciones WHERE token = ?"
    parametros = [token]
    if ticker:
        consulta += " AND ticker = ?"
        parametros.append(ticker)
    with bd.transaccion() as cursor:
        filas = cursor.ejecutar(consulta + " ORDER BY id", parametros).todos()
    return filas[-limite:] if limite else filas


def borrar_historial(bd, token):
    with bd.transaccion() as cursor:
        cursor.ejecutar("DELETE FROM conversaciones WHERE token = ?", (token,))


def auditar(bd, modelo, resultado, datos_detectados=None, motivo=None):
    if resultado not in RESULTADOS:
        raise ValueError(f"Resultado de auditoría desconocido: {resultado}")
    with bd.transaccion() as cursor:
        cursor.ejecutar(
            "INSERT INTO auditoria_agente (fecha, modelo, resultado, datos_detectados, motivo) VALUES (?, ?, ?, ?, ?)",
            (a_texto(ahora_utc()), modelo, resultado,
             json.dumps(datos_detectados, ensure_ascii=False) if datos_detectados else None, motivo),
        )


def resumen_auditoria(bd, dias=7):
    """Conteo de preguntas por resultado en los últimos `dias` días, sin datos de ningún usuario."""
    desde = a_texto(ahora_utc() - pd.Timedelta(days=dias))
    with bd.transaccion() as cursor:
        filas = cursor.ejecutar("SELECT resultado, COUNT(*) AS preguntas FROM auditoria_agente WHERE fecha >= ? "
                                "GROUP BY resultado ORDER BY resultado", (desde,)).todos()
    return pd.DataFrame(filas, columns=["resultado", "preguntas"])
