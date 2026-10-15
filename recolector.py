"""Recolección automática de precios.

Uso:
    python recolector.py                       # AAPL cada 2 minutos
    python recolector.py --tickers AAPL MSFT   # varias acciones
    python recolector.py --una-vez             # una sola actualización (útil con el Programador de tareas)
"""

import argparse
import time

import pandas as pd

import config
import datos

MINUTOS_ENTRE_ACTUALIZACIONES_CERRADO = 30


def actualizar_todo(tickers, intervalos):
    marca = pd.Timestamp.now(tz=config.ZONA_USUARIO).strftime("%Y-%m-%d %H:%M:%S")
    for ticker in tickers:
        for intervalo in intervalos:
            df, error = datos.actualizar(ticker, intervalo)
            if error:
                print(f"[{marca}] {ticker} {intervalo}: {error}")
                continue
            ultima = df.index[-1].tz_convert(config.ZONA_USUARIO).strftime("%Y-%m-%d %H:%M")
            print(f"[{marca}] {ticker} {intervalo}: {len(df):>6} velas guardadas | última vela {ultima} (Bogotá) "
                  f"| cierre {df['Close'].iloc[-1]:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Recolector de precios con yfinance")
    parser.add_argument("--tickers", nargs="+", default=[config.TICKER_POR_DEFECTO])
    parser.add_argument("--intervalos", nargs="+", default=[config.INTERVALO_VIVO, config.INTERVALO_MODELO])
    parser.add_argument("--cada", type=int, default=config.SEGUNDOS_ACTUALIZACION, help="segundos entre consultas")
    parser.add_argument("--una-vez", action="store_true")
    args = parser.parse_args()
    tickers = [t.upper() for t in args.tickers]

    actualizar_todo(tickers, args.intervalos)
    if args.una_vez:
        return

    ultima_actualizacion = time.time()
    print(f"Recolectando {', '.join(tickers)} cada {args.cada} s con el mercado abierto "
          f"(cada {MINUTOS_ENTRE_ACTUALIZACIONES_CERRADO} min con el mercado cerrado). Ctrl+C para detener.")
    try:
        while True:
            time.sleep(args.cada)
            cerrado = not datos.mercado_abierto()
            if cerrado and time.time() - ultima_actualizacion < MINUTOS_ENTRE_ACTUALIZACIONES_CERRADO * 60:
                continue
            actualizar_todo(tickers, args.intervalos)
            ultima_actualizacion = time.time()
    except KeyboardInterrupt:
        print("Recolector detenido.")


if __name__ == "__main__":
    main()
