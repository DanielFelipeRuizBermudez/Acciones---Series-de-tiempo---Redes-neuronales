"""Almacenamiento de usuarios, portafolios simulados y operaciones.

En el equipo local usa un archivo SQLite (base_datos/usuarios.db). Si se define DATABASE_URL, usa esa base Postgres,
que es lo que permite que las cuentas sobrevivan a los reinicios de Streamlit Community Cloud.

El correo se guarda cifrado (Fernet) en una tabla aparte y se busca por un índice HMAC, así que la base nunca tiene el
correo en texto plano. El portafolio, las operaciones, el uso del agente y su historial solo se relacionan con el
usuario por su token aleatorio. La auditoría del agente no guarda ni el token ni el texto de las preguntas.
"""

import base64
import hashlib
import hmac
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

import config

_TABLAS = (
    """CREATE TABLE IF NOT EXISTS usuarios (
        token TEXT PRIMARY KEY,
        alias TEXT NOT NULL,
        hash_contrasena TEXT NOT NULL,
        mayor_de_edad BOOLEAN NOT NULL,
        autorizacion_en TEXT NOT NULL,
        version_politica TEXT NOT NULL,
        historial_agente BOOLEAN NOT NULL,
        creado_en TEXT NOT NULL,
        ultimo_acceso TEXT,
        intentos_fallidos INTEGER NOT NULL DEFAULT 0,
        bloqueado_hasta TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS correos (
        token TEXT PRIMARY KEY REFERENCES usuarios (token) ON DELETE CASCADE,
        indice TEXT NOT NULL UNIQUE,
        cifrado TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS portafolios (
        token TEXT PRIMARY KEY REFERENCES usuarios (token) ON DELETE CASCADE,
        efectivo DOUBLE PRECISION NOT NULL,
        saldo_inicial DOUBLE PRECISION NOT NULL,
        inicio_en TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS operaciones (
        id {clave_autonumerica},
        token TEXT NOT NULL REFERENCES usuarios (token) ON DELETE CASCADE,
        fecha TEXT NOT NULL,
        ticker TEXT NOT NULL,
        tipo TEXT NOT NULL CHECK (tipo IN ('compra', 'venta')),
        cantidad INTEGER NOT NULL CHECK (cantidad > 0),
        precio DOUBLE PRECISION NOT NULL CHECK (precio > 0),
        comision DOUBLE PRECISION NOT NULL CHECK (comision >= 0),
        consenso TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS operaciones_por_token ON operaciones (token, fecha)",
    """CREATE TABLE IF NOT EXISTS uso_agente (
        token TEXT NOT NULL REFERENCES usuarios (token) ON DELETE CASCADE,
        dia TEXT NOT NULL,
        consultas INTEGER NOT NULL CHECK (consultas >= 0),
        PRIMARY KEY (token, dia)
    )""",
    """CREATE TABLE IF NOT EXISTS conversaciones (
        id {clave_autonumerica},
        token TEXT NOT NULL REFERENCES usuarios (token) ON DELETE CASCADE,
        fecha TEXT NOT NULL,
        ticker TEXT NOT NULL,
        rol TEXT NOT NULL CHECK (rol IN ('usuario', 'modelo')),
        texto TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS conversaciones_por_token ON conversaciones (token, ticker, id)",
    """CREATE TABLE IF NOT EXISTS auditoria_agente (
        id {clave_autonumerica},
        fecha TEXT NOT NULL,
        modelo TEXT,
        resultado TEXT NOT NULL,
        datos_detectados TEXT,
        motivo TEXT
    )""",
)


def ahora_utc():
    return datetime.now(timezone.utc)


def a_texto(momento):
    return momento.astimezone(timezone.utc).isoformat(timespec="seconds")


def de_texto(texto):
    return datetime.fromisoformat(texto) if texto else None


def normalizar_correo(correo):
    return correo.strip().lower()


class _Cursor:
    """Cursor con la misma interfaz para SQLite y Postgres: las consultas se escriben con `?`."""

    def __init__(self, cursor, postgres):
        self._cursor = cursor
        self._postgres = postgres

    def ejecutar(self, sql, parametros=()):
        self._cursor.execute(sql.replace("?", "%s") if self._postgres else sql, tuple(parametros))
        return self

    def uno(self):
        fila = self._cursor.fetchone()
        return dict(fila) if fila is not None else None

    def todos(self):
        return [dict(fila) for fila in self._cursor.fetchall()]

    @property
    def filas_afectadas(self):
        return self._cursor.rowcount


class BaseDatos:
    def __init__(self, clave, url=None, ruta_sqlite=None):
        if not clave:
            raise ValueError("Falta la clave de cifrado de los datos (CLAVE_DATOS).")
        self.url = url or None
        self.postgres = self.url is not None
        self.ruta_sqlite = Path(ruta_sqlite or config.RUTA_BASE_LOCAL)
        material = clave.encode("utf-8")
        self._fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256(b"cifrado|" + material).digest()))
        self._clave_indice = hashlib.sha256(b"indice|" + material).digest()
        self.bloqueo = " FOR UPDATE" if self.postgres else ""
        self._crear_tablas()

    @property
    def descripcion(self):
        return "Postgres (DATABASE_URL)" if self.postgres else f"SQLite local ({self.ruta_sqlite.name})"

    def _conectar(self):
        if self.postgres:
            import psycopg2
            import psycopg2.extras

            conexion = psycopg2.connect(self.url, connect_timeout=10)
            return conexion, conexion.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        self.ruta_sqlite.parent.mkdir(parents=True, exist_ok=True)
        conexion = sqlite3.connect(self.ruta_sqlite, timeout=15, isolation_level=None)
        conexion.row_factory = sqlite3.Row
        conexion.execute("PRAGMA foreign_keys = ON")
        conexion.execute("BEGIN IMMEDIATE")
        return conexion, conexion.cursor()

    @contextmanager
    def transaccion(self):
        """Todo lo que se ejecuta dentro queda guardado junto o no queda nada (commit o rollback)."""
        conexion, cursor = self._conectar()
        try:
            yield _Cursor(cursor, self.postgres)
            if self.postgres:
                conexion.commit()
            else:
                conexion.execute("COMMIT")
        except BaseException:
            if self.postgres:
                conexion.rollback()
            elif conexion.in_transaction:
                conexion.execute("ROLLBACK")
            raise
        finally:
            conexion.close()

    def _crear_tablas(self):
        autonumerica = "BIGSERIAL PRIMARY KEY" if self.postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
        with self.transaccion() as cursor:
            for sentencia in _TABLAS:
                cursor.ejecutar(sentencia.replace("{clave_autonumerica}", autonumerica))

    def indice_correo(self, correo):
        return hmac.new(self._clave_indice, normalizar_correo(correo).encode("utf-8"), hashlib.sha256).hexdigest()

    def cifrar(self, texto):
        return self._fernet.encrypt(texto.encode("utf-8")).decode("ascii")

    def descifrar(self, texto):
        try:
            return self._fernet.decrypt(texto.encode("ascii")).decode("utf-8")
        except (InvalidToken, ValueError):
            return None


def clave_local():
    """Clave para la base SQLite local; se crea una sola vez y queda en base_datos/, fuera del repositorio."""
    ruta = config.RUTA_CLAVE_LOCAL
    if ruta.exists():
        return ruta.read_text(encoding="utf-8").strip()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    clave = secrets.token_urlsafe(48)
    ruta.write_text(clave, encoding="utf-8")
    return clave


def crear(url=None, clave=None):
    """Base Postgres si hay URL (exige CLAVE_DATOS); si no, SQLite local con su clave guardada en el equipo."""
    if url:
        if not clave:
            raise ValueError("Con DATABASE_URL también hay que definir CLAVE_DATOS en los secretos de la app.")
        return BaseDatos(clave, url=url)
    return BaseDatos(clave or clave_local())
