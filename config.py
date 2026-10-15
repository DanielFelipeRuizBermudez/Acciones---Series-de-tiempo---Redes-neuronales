from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DIR_DATOS = RAIZ / "datos"
DIR_MODELOS = RAIZ / "modelos_entrenados"

TICKER_POR_DEFECTO = "AAPL"
TICKERS_SUGERIDOS = ["AAPL", "MSFT", "NVDA", "SPY", "EC"]

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

CARACTERISTICAS = ["log_ret", "rango", "vol_rel"]
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
    "tasa": [1e-3],
}
REJILLA_REDES_RAPIDA = {
    "ventana": [7, 14],
    "unidades": [32],
    "capas": [1],
    "dropout": [0.2],
    "tasa": [1e-3],
}
EPOCAS = 100
PACIENCIA = 10
TAMANO_LOTE = 32
TASA_APRENDIZAJE = 1e-3
SEMILLA = 42


def directorio_modelos(ticker, intervalo):
    return DIR_MODELOS / f"{ticker.upper()}_{intervalo}"