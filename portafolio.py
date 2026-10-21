"""Portafolio simulado con dinero ficticio.

Reglas del juego, iguales para todos los usuarios:
- Cada cuenta empieza con SALDO_INICIAL dólares ficticios y puede reiniciar su portafolio cuando quiera.
- Solo se opera con acciones que tienen modelos entrenados, con el mercado abierto y al último precio de 2 minutos.
- Se compran y venden acciones enteras, con una comisión simulada de COMISION_SIMULADA_PCT % por operación.
- No se puede comprar sin saldo ni vender acciones que no se tienen.

Las validaciones se hacen en el servidor dentro de la misma transacción que registra la operación, así que el efectivo
solo cambia por una operación válida y registrada. Las acciones que tiene cada usuario no se guardan aparte: se calculan
sumando sus compras y restando sus ventas.
"""

import math

import numpy as np
import pandas as pd

import config
import datos
from base_datos import a_texto, ahora_utc, de_texto

TIPOS = ("compra", "venta")
COLUMNAS_OPERACIONES = ["id", "fecha", "ticker", "tipo", "cantidad", "precio", "comision", "consenso"]


class OperacionRechazada(Exception):
    """La operación no cumple una regla del juego; el mensaje explica cuál."""


def comision(monto):
    return round(monto * config.COMISION_SIMULADA_PCT / 100, 2)


def horario_bogota(ahora=None):
    """Horario de la bolsa de EE. UU. en hora de Bogotá para el día de `ahora` (cambia con el horario de verano)."""
    dia = (ahora or datos.ahora_mercado()).tz_convert(config.ZONA_MERCADO).normalize()
    apertura = dia + pd.Timedelta(hours=config.APERTURA[0], minutes=config.APERTURA[1])
    cierre = dia + pd.Timedelta(hours=config.CIERRE[0], minutes=config.CIERRE[1])
    return (f"{apertura.tz_convert(config.ZONA_USUARIO):%H:%M} a "
            f"{cierre.tz_convert(config.ZONA_USUARIO):%H:%M}")


def cotizacion(df_2m):
    """(precio, inicio de la vela) del último dato de 2 minutos, o (None, None) si no hay datos."""
    if df_2m is None or df_2m.empty:
        return None, None
    return float(df_2m["Close"].iloc[-1]), df_2m.index[-1]


def texto_consenso(consenso):
    if not consenso:
        return None
    total = consenso["suben"] + consenso["bajan"]
    if consenso["direccion"] == "Empate":
        return f"Empate ({consenso['suben']} de {total} modelos predicen que sube)"
    a_favor = consenso["suben"] if consenso["direccion"] == "Sube" else consenso["bajan"]
    return f"{consenso['direccion']} ({a_favor} de {total} modelos)"


def maximo_comprable(efectivo, precio):
    if not precio or precio <= 0 or efectivo <= 0:
        return 0
    cantidad = int(efectivo // (precio * (1 + config.COMISION_SIMULADA_PCT / 100)))
    while cantidad > 0:
        monto = round(cantidad * precio, 2)
        if monto + comision(monto) <= efectivo + 1e-9:
            break
        cantidad -= 1
    return cantidad


def validar(tipo, ticker, cantidad, precio, marca, efectivo, acciones, permitidos, ahora):
    """Devuelve (monto, comisión) o lanza OperacionRechazada con el motivo."""
    if tipo not in TIPOS:
        raise OperacionRechazada("Tipo de operación no válido.")
    if ticker not in permitidos:
        raise OperacionRechazada(f"{ticker} no tiene modelos entrenados; por ahora solo se opera con "
                                 f"{', '.join(permitidos)}.")
    if (isinstance(cantidad, bool) or not isinstance(cantidad, (int, np.integer))
            or not 1 <= int(cantidad) <= config.CANTIDAD_MAXIMA_OPERACION):
        raise OperacionRechazada("La cantidad debe ser un número entero de acciones mayor que cero.")
    momento = pd.Timestamp(ahora).tz_convert(config.ZONA_MERCADO)
    if not datos.mercado_abierto(momento):
        raise OperacionRechazada(f"El mercado está cerrado. Se opera de 9:30 a 16:00, hora de Nueva York "
                                 f"({horario_bogota(momento)} en Bogotá).")
    if precio is None or marca is None or not precio > 0:
        raise OperacionRechazada("No hay un precio disponible para operar.")
    antiguedad = momento - pd.Timestamp(marca).tz_convert(config.ZONA_MERCADO)
    if antiguedad > pd.Timedelta(minutes=config.PRECIO_MAX_ANTIGUEDAD_MIN):
        raise OperacionRechazada(f"El último precio es de hace {int(antiguedad.total_seconds() // 60)} minutos y para "
                                 f"operar debe tener menos de {config.PRECIO_MAX_ANTIGUEDAD_MIN}. Hay que actualizar "
                                 "los datos e intentar de nuevo.")
    monto = round(int(cantidad) * precio, 2)
    costo = comision(monto)
    if tipo == "compra" and monto + costo > efectivo + 1e-9:
        raise OperacionRechazada(f"Saldo insuficiente: la compra cuesta {monto + costo:,.2f} USD con la comisión y hay "
                                 f"{efectivo:,.2f} USD.")
    if tipo == "venta" and int(cantidad) > acciones.get(ticker, 0):
        raise OperacionRechazada(f"Solo hay {acciones.get(ticker, 0)} acciones de {ticker} para vender.")
    return monto, costo


def _acciones(cursor, token):
    filas = cursor.ejecutar(
        "SELECT ticker, SUM(CASE WHEN tipo = 'compra' THEN cantidad ELSE -cantidad END) AS acciones "
        "FROM operaciones WHERE token = ? GROUP BY ticker", (token,)).todos()
    return {fila["ticker"]: int(fila["acciones"]) for fila in filas if int(fila["acciones"]) != 0}


def ejecutar(bd, token, tipo, ticker, cantidad, precio, marca, consenso, permitidos, ahora=None):
    """Valida y registra la operación en una sola transacción. Devuelve el detalle con el efectivo resultante."""
    ahora = ahora or ahora_utc()
    precio = round(float(precio), 4) if precio is not None else None
    with bd.transaccion() as cursor:
        portafolio = cursor.ejecutar("SELECT efectivo FROM portafolios WHERE token = ?" + bd.bloqueo, (token,)).uno()
        if portafolio is None:
            raise OperacionRechazada("La cuenta ya no existe.")
        efectivo = float(portafolio["efectivo"])
        monto, costo = validar(tipo, ticker, cantidad, precio, marca, efectivo, _acciones(cursor, token),
                               permitidos, ahora)
        nuevo = round(efectivo - monto - costo if tipo == "compra" else efectivo + monto - costo, 2)
        cursor.ejecutar(
            "INSERT INTO operaciones (token, fecha, ticker, tipo, cantidad, precio, comision, consenso) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (token, a_texto(ahora), ticker, tipo, int(cantidad), precio, costo, consenso),
        )
        cursor.ejecutar("UPDATE portafolios SET efectivo = ? WHERE token = ?", (nuevo, token))
    return {"tipo": tipo, "ticker": ticker, "cantidad": int(cantidad), "precio": precio, "monto": monto,
            "comision": costo, "efectivo": nuevo}


def estado(bd, token):
    """Efectivo, saldo inicial, fecha de inicio y operaciones del portafolio, o None si la cuenta no existe."""
    with bd.transaccion() as cursor:
        portafolio = cursor.ejecutar("SELECT efectivo, saldo_inicial, inicio_en FROM portafolios WHERE token = ?",
                                     (token,)).uno()
        if portafolio is None:
            return None
        filas = cursor.ejecutar(
            "SELECT id, fecha, ticker, tipo, cantidad, precio, comision, consenso FROM operaciones "
            "WHERE token = ? ORDER BY fecha, id", (token,)).todos()
    operaciones = pd.DataFrame(filas, columns=COLUMNAS_OPERACIONES)
    if not operaciones.empty:
        operaciones["fecha"] = pd.to_datetime(operaciones["fecha"], utc=True)
        operaciones = operaciones.astype({"cantidad": int, "precio": float, "comision": float})
    return {
        "efectivo": float(portafolio["efectivo"]),
        "saldo_inicial": float(portafolio["saldo_inicial"]),
        "inicio_en": de_texto(portafolio["inicio_en"]),
        "operaciones": operaciones,
    }


def efectivo_segun_operaciones(saldo_inicial, operaciones):
    """Recalcula el efectivo solo con las operaciones registradas, para verificar el saldo guardado."""
    total = saldo_inicial
    for operacion in operaciones.itertuples():
        monto = round(operacion.cantidad * operacion.precio, 2)
        total += (-monto if operacion.tipo == "compra" else monto) - operacion.comision
    return round(total, 2)


def posiciones(operaciones, precios):
    """Acciones que se tienen, con costo promedio (comisiones incluidas), valor actual y ganancia no realizada."""
    filas = []
    if operaciones.empty:
        return pd.DataFrame(filas)
    for ticker, grupo in operaciones.groupby("ticker", sort=True):
        cantidad, costo_total = 0, 0.0
        for operacion in grupo.itertuples():
            if operacion.tipo == "compra":
                costo_total += operacion.cantidad * operacion.precio + operacion.comision
                cantidad += operacion.cantidad
            elif cantidad:
                costo_total -= costo_total / cantidad * operacion.cantidad
                cantidad -= operacion.cantidad
        if cantidad <= 0:
            continue
        precio = precios.get(ticker)
        valor = cantidad * precio if precio else math.nan
        filas.append({
            "acción": ticker,
            "acciones": cantidad,
            "costo promedio": costo_total / cantidad,
            "precio actual": precio if precio else math.nan,
            "valor": valor,
            "ganancia": valor - costo_total,
            "ganancia %": 100 * (valor / costo_total - 1) if costo_total else math.nan,
        })
    return pd.DataFrame(filas)


def comprar_y_mantener(saldo_inicial, inicio_en, series, precio_actual):
    """Cómo iría el portafolio si al empezar se hubiera comprado todo lo posible de la acción sin vender nunca.

    `series` son velas (de la más fina a la más gruesa); se usa la primera que cubra la fecha de inicio y el precio
    de apertura de la primera vela desde esa fecha.
    """
    inicio = pd.Timestamp(inicio_en)
    for serie in series:
        if serie is None or serie.empty or serie.index[0] > inicio:
            continue
        posteriores = serie[serie.index >= inicio]
        if posteriores.empty:
            return None
        precio_inicial = float(posteriores["Open"].iloc[0])
        cantidad = maximo_comprable(saldo_inicial, precio_inicial)
        monto = round(cantidad * precio_inicial, 2)
        efectivo = saldo_inicial - monto - comision(monto)
        return {"precio_inicial": precio_inicial, "marca": posteriores.index[0], "acciones": cantidad,
                "valor": efectivo + cantidad * precio_actual}
    return None


def frente_al_consenso(operaciones):
    """Cuántas operaciones siguieron al consenso de los modelos (comprar si decía Sube, vender si decía Baja)."""
    cuenta = {"a favor": 0, "en contra": 0, "sin consenso": 0}
    for operacion in operaciones.itertuples():
        consenso = operacion.consenso or ""
        if consenso.startswith("Sube"):
            cuenta["a favor" if operacion.tipo == "compra" else "en contra"] += 1
        elif consenso.startswith("Baja"):
            cuenta["a favor" if operacion.tipo == "venta" else "en contra"] += 1
        else:
            cuenta["sin consenso"] += 1
    return cuenta


def reiniciar(bd, token):
    """Borra las operaciones y devuelve el portafolio al saldo inicial, con nueva fecha de inicio."""
    with bd.transaccion() as cursor:
        cursor.ejecutar("DELETE FROM operaciones WHERE token = ?", (token,))
        cursor.ejecutar("UPDATE portafolios SET efectivo = ?, saldo_inicial = ?, inicio_en = ? WHERE token = ?",
                        (config.SALDO_INICIAL, config.SALDO_INICIAL, a_texto(ahora_utc()), token))
