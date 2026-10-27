from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DIR_DATOS = RAIZ / "datos"
DIR_MODELOS = RAIZ / "modelos_entrenados"
DIR_BASE_LOCAL = RAIZ / "base_datos"
RUTA_BASE_LOCAL = DIR_BASE_LOCAL / "usuarios.db"
RUTA_CLAVE_LOCAL = DIR_BASE_LOCAL / "clave_local.txt"

TICKER_POR_DEFECTO = "AAPL"

INTERVALO_MODELO = "1h"
INTERVALO_VIVO = "2m"
SEGUNDOS_ACTUALIZACION = 120

ZONA_MERCADO = "America/New_York"
ZONA_USUARIO = "America/Bogota"
APERTURA = (9, 30)
CIERRE = (16, 0)

MINUTOS_INTERVALO = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60, "90m": 90}

HISTORIA_MAXIMA_DIAS = {"1m": 6, "2m": 58, "5m": 58, "15m": 58, "30m": 58, "60m": 720, "1h": 720, "90m": 58}
SOLAPAMIENTO_DIAS = 3

CARACTERISTICAS = ["log_ret"]
VENTANA_VOLUMEN = 35

PROPORCION_VALIDACION = 0.15
PROPORCION_PRUEBA = 0.15
PLIEGUES_CV = 3
PLIEGUES_CV_RAPIDO = 2

AR_MAX_P = 10

PROPHET_REJILLA = {
    "changepoint_prior_scale": [0.01, 0.05, 0.3],
    "seasonality_prior_scale": [0.1, 1.0, 10.0],
}
PROPHET_PARAMETROS_POR_DEFECTO = {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0}
PROPHET_DIAS_ENTRE_REAJUSTES_VALIDACION = 5
PROPHET_DIAS_ENTRE_REAJUSTES_PRUEBA = 1

REJILLA_REDES = {
    "ventana": [7, 14, 35],
    "unidades": [32, 64],
    "capas": [1, 2],
    "dropout": [0.2],
    "tasa": [0.01, 0.05, 0.1],
}
REJILLA_REDES_RAPIDA = {
    "ventana": [7, 14],
    "unidades": [32],
    "capas": [1],
    "dropout": [0.2],
    "tasa": [0.05],
}
REJILLA_CNN = {
    "velas": [10, 20],
    "filtros": [16, 32],
    "tasa": [0.01, 0.05],
}
REJILLA_CNN_RAPIDA = {
    "velas": [20],
    "filtros": [16],
    "tasa": [0.05],
}
CNN_ALTO_IMAGEN = 40
CNN_ANCHO_VELA = 4
CNN_DROPOUT = 0.2
CNN_UMBRAL = 0.5

VELAS_SEGUIDAS = 3
COSTO_OPERACION_PCT = 0.05
VELAS_POR_ANO = 7 * 252

GEMINI_MODELOS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]

NOMBRE_APP = "Predicción de acciones"
CORREO_CONTACTO = "acciones1stdg2appr3espd@gmail.com"
RESPONSABLE_CONSULTAS = "Daniel Ruiz"
VERSION_POLITICA = "2026-10-27"

SALDO_INICIAL = 10_000.0
COMISION_SIMULADA_PCT = 0.1
PRECIO_MAX_ANTIGUEDAD_MIN = 10
CANTIDAD_MAXIMA_OPERACION = 100_000

INTENTOS_MAXIMOS = 5
MINUTOS_BLOQUEO = 15
MINUTOS_SESION = 30
CONTRASENA_MINIMA = 8

AGENTE_CONSULTAS_DIARIAS = 20
AGENTE_MAX_CARACTERES = 500
AGENTE_MENSAJES_HISTORIAL = 20

EPOCAS = 100
PACIENCIA = 10
TAMANO_LOTE = 32
OPTIMIZADOR = "SGD"
TASA_APRENDIZAJE = 0.05
SEMILLA = 42


def directorio_modelos(ticker, intervalo):
    return DIR_MODELOS / f"{ticker.upper()}_{intervalo}"