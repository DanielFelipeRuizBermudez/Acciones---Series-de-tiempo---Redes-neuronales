"""Entrena y evalúa AR(p), Prophet, RNN y LSTM para una acción.

Uso:
    python entrenar.py                         # AAPL con velas de 1 hora
    python entrenar.py --ticker MSFT
    python entrenar.py --rapido                # rejilla pequeña, para probar que todo funciona
"""

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

import config
import datos
import evaluacion
import modelos_redes
import modelos_series
import preparacion


def _fecha(marca):
    return marca.tz_convert(config.ZONA_MERCADO).strftime("%Y-%m-%d %H:%M")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de predicción de acciones")
    parser.add_argument("--ticker", default=config.TICKER_POR_DEFECTO)
    parser.add_argument("--intervalo", default=config.INTERVALO_MODELO)
    parser.add_argument("--rapido", action="store_true", help="rejilla pequeña y Prophet sin búsqueda")
    parser.add_argument("--sin-prophet", action="store_true")
    parser.add_argument("--sin-descarga", action="store_true", help="usar solo los datos ya guardados")
    args = parser.parse_args()
    ticker, intervalo = args.ticker.upper(), args.intervalo
    inicio_total = time.time()

    print(f"== {ticker} | velas de {intervalo} ==")
    if args.sin_descarga:
        df = datos.leer_local(ticker, intervalo)
    else:
        df, error = datos.actualizar(ticker, intervalo)
        if error:
            print(f"Aviso: {error}")
    df = datos.velas_completas(df, intervalo)
    if len(df) < 300:
        raise SystemExit(f"Solo hay {len(df)} velas de {ticker}; se necesitan al menos 300 para entrenar.")

    f = preparacion.construir_caracteristicas(df)
    cortes = preparacion.calcular_cortes(f)
    escalado = preparacion.ajustar_escalado(f, preparacion.cortes_desarrollo(cortes))
    k = config.PLIEGUES_CV_RAPIDO if args.rapido else config.PLIEGUES_CV
    pliegues = preparacion.pliegues_temporales(cortes, k)
    marcas = f.index
    for nombre in ("entrenamiento", "validacion", "prueba"):
        a, b = cortes.tramo(nombre)
        print(f"  {nombre:<13} {b - a:>5} velas  {_fecha(marcas[a])} -> {_fecha(marcas[b - 1])} (hora NY)")
    print(f"  Validación cruzada temporal para AR y redes ({k} pliegues sobre entrenamiento + validación):")
    for numero, pliegue in enumerate(pliegues, start=1):
        a, b = pliegue.tramo("validacion")
        print(f"    pliegue {numero}: entrena con {pliegue.fin_entrenamiento:>5} velas, "
              f"valida con {b - a} ({_fecha(marcas[a])} -> {_fecha(marcas[b - 1])})")

    a, b = cortes.tramo("prueba")
    r_real = f["objetivo"].to_numpy()[a:b]
    r_actual = f["log_ret"].to_numpy()[a:b]
    cierre_actual = f["cierre"].to_numpy()[a:b]
    predicciones = pd.DataFrame({
        "marca_referencia": marcas[a:b],
        "marca_objetivo": marcas[a + 1: b + 1],
        "cierre_referencia": cierre_actual,
        "retorno_real": r_real,
        "precio_real": cierre_actual * np.exp(r_real),
    })
    filas_metricas = evaluacion.lineas_base(r_real, r_actual, cierre_actual)
    resumen = {
        "ticker": ticker,
        "intervalo": intervalo,
        "entrenado_en": pd.Timestamp.now(tz=config.ZONA_USUARIO).strftime("%Y-%m-%d %H:%M"),
        "caracteristicas": config.CARACTERISTICAS,
        "velas": len(f),
        "tramos": {n: {"desde": _fecha(marcas[cortes.tramo(n)[0]]), "hasta": _fecha(marcas[cortes.tramo(n)[1] - 1]),
                       "velas": cortes.tramo(n)[1] - cortes.tramo(n)[0]}
                   for n in ("entrenamiento", "validacion", "prueba")},
        "validacion_cruzada": {
            "pliegues": k,
            "tramos": [{"entrena_hasta": _fecha(marcas[p.fin_entrenamiento - 1]),
                        "valida_desde": _fecha(marcas[p.fin_entrenamiento]),
                        "valida_hasta": _fecha(marcas[p.fin_validacion - 1])} for p in pliegues],
        },
    }
    salida = config.directorio_modelos(ticker, intervalo)
    salida.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] AR(p) sobre log-retornos")
    p, modelo_ar, tabla_ar = modelos_series.buscar_ar(f, cortes, k=k)
    pred_ar = modelos_series.predecir_ar(modelo_ar, f, p, a, b)
    predicciones[f"AR({p})"] = pred_ar
    filas_metricas.append(evaluacion.metricas(f"AR({p})", r_real, cierre_actual, r_pred=pred_ar))
    resumen["ar"] = modelos_series.ar_a_dict(modelo_ar, p)
    tabla_ar.to_csv(salida / "busqueda_ar.csv", index=False)
    print(f"  p elegido = {p}")

    if not args.sin_prophet:
        print("\n[2/4] Prophet sobre log(precio)")
        if args.rapido:
            parametros = modelos_series.parametros_prophet()
            dias = config.PROPHET_DIAS_ENTRE_REAJUSTES_VALIDACION
        else:
            print("  búsqueda de hiperparámetros:")

            def informar_prophet(numero, total, fila):
                print(f"  {numero:>2}/{total} changepoint_prior_scale={fila['changepoint_prior_scale']:<5} "
                      f"seasonality_prior_scale={fila['seasonality_prior_scale']:<5} -> mse_val={fila['mse_validacion']:.3e}")

            parametros, tabla_prophet = modelos_series.buscar_prophet(f, cortes, intervalo, al_terminar=informar_prophet)
            tabla_prophet.to_csv(salida / "busqueda_prophet.csv", index=False)
            dias = config.PROPHET_DIAS_ENTRE_REAJUSTES_PRUEBA
        cada = datos.velas_por_dia(intervalo) * dias
        pred_prophet = modelos_series.backtest_prophet(f, a, b, parametros, cada)
        predicciones["Prophet"] = pred_prophet
        filas_metricas.append(evaluacion.metricas("Prophet", r_real, cierre_actual, r_pred=pred_prophet))
        resumen["prophet"] = {**parametros, "reajuste_cada_velas": cada}
        print("  elegido: " + ", ".join(f"{k} = {v}" for k, v in parametros.items()))

    rejilla = config.REJILLA_REDES_RAPIDA if args.rapido else config.REJILLA_REDES
    resumen["redes"] = {}
    for paso, tipo in ((3, "rnn"), (4, "lstm")):
        nombre = modelos_redes.TIPOS[tipo]
        print(f"\n[{paso}/4] {nombre}: búsqueda de hiperparámetros")

        def informar(numero, total, fila):
            print(f"  {numero:>2}/{total} ventana={fila['ventana']:<3} unidades={fila['unidades']:<3} "
                  f"capas={fila['capas']} dropout={fila['dropout']} tasa={fila['tasa']} "
                  f"-> rmse_cv={fila['rmse_cv_pct']:.4f}% ±{fila['desv_pliegues_pct']:.4f} "
                  f"({fila['epocas_promedio']:.0f} épocas en promedio)")

        mejor, tabla = modelos_redes.buscar_hiperparametros(tipo, f, cortes, escalado, rejilla, k, informar)
        x_prueba, _ = mejor["conjuntos"]["prueba"]
        pred = modelos_redes.predecir(mejor["modelo"], x_prueba, escalado)
        predicciones[nombre] = pred
        filas_metricas.append(evaluacion.metricas(nombre, r_real, cierre_actual, r_pred=pred))
        modelos_redes.guardar_modelo(mejor["modelo"], salida / f"{tipo}.keras")
        mejor["historial"].to_csv(salida / f"historial_{tipo}.csv", index=False)
        elegido = mejor["fila"]
        print(f"  elegido: ventana={elegido['ventana']}, unidades={elegido['unidades']}, capas={elegido['capas']}, "
              f"dropout={elegido['dropout']}, tasa={elegido['tasa']}; reentrenado con todo el desarrollo "
              f"durante {elegido['epocas_finales']} épocas")
        tabla.to_csv(salida / f"busqueda_{tipo}.csv", index=False)
        resumen["redes"][tipo] = mejor["fila"]

    joblib.dump(escalado, salida / "escalado.pkl")
    tabla = evaluacion.tabla_metricas(filas_metricas)
    tabla.to_csv(salida / "metricas.csv", index=False)
    predicciones.to_csv(salida / "predicciones_prueba.csv", index=False)
    with open(salida / "resumen.json", "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=2)

    print("\n== Resultados en el conjunto de prueba ==")
    vista = tabla.copy()
    vista["acierto_direccion"] = vista["acierto_direccion"].map(lambda v: "" if pd.isna(v) else f"{100 * v:.1f}%")
    vista["margen_95"] = vista["margen_95"].map(lambda v: "" if pd.isna(v) else f"±{100 * v:.1f}%")
    print(vista.to_string(index=False, na_rep=""))
    print(f"\nModelos guardados en {salida} ({(time.time() - inicio_total) / 60:.1f} min)")


if __name__ == "__main__":
    main()