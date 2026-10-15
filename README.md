# Predicción de acciones: series de tiempo (stdg) + aprendizaje profundo (appr)

Dashboard en Streamlit que descarga precios con **yfinance**, los va guardando automáticamente y predice si
la **próxima vela de 1 hora** de una acción cierra arriba o abajo, con cuatro modelos:

| Modelo | Materia | Qué hace | Archivo |
|---|---|---|---|
| AR(p) | Series de tiempo | Regresión lineal del retorno sobre sus `p` retornos anteriores | `modelos_series.py` |
| Prophet | Series de tiempo | Tendencia + estacionalidad diaria y semanal sobre log(precio) | `modelos_series.py` |
| RNN simple | Aprendizaje profundo | Red recurrente sobre ventanas de las últimas velas | `modelos_redes.py` |
| LSTM | Aprendizaje profundo | Igual, con celdas LSTM (memoria de largo plazo) | `modelos_redes.py` |

Se comparan contra tres líneas base: *precio sin cambio*, *siempre sube* y *repite la última vela*.

## Instalación (Windows)

Se recomienda Python 3.11 o 3.12 (TensorFlow y Prophet todavía pueden fallar con versiones más nuevas).

```powershell
cd acciones_appr_stdg
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> `fbprophet` ya no existe con ese nombre: el paquete actual es `prophet` (`from prophet import Prophet`).

## Uso

```powershell
python recolector.py --una-vez      # 1. descarga inicial: ~60 días de velas de 2 min y ~2 años de velas de 1 h
python entrenar.py --rapido         # 2a. prueba rápida de que todo funciona (1-2 min)
python entrenar.py                  # 2b. entrenamiento completo con búsqueda de hiperparámetros (5-15 min)
streamlit run app.py                # 3. dashboard
```

Para otra acción: `python entrenar.py --ticker MSFT`. En el dashboard se elige en la barra lateral, donde
también hay un buscador de símbolos.

Para seguir recolectando datos aunque el dashboard esté cerrado, deja corriendo en otra terminal:

```powershell
python recolector.py --tickers AAPL MSFT
```

Consulta cada 2 minutos con el mercado abierto y cada 30 minutos con el mercado cerrado.

## Cómo funciona

### Los datos

- yfinance **no es un flujo en vivo**: cada consulta devuelve velas ya formadas (apertura, máximo, mínimo, cierre,
  volumen). "Cada 2 minutos" significa que se piden velas de 2 minutos y se vuelve a consultar cada 2 minutos.
- Yahoo solo guarda unos **60 días** de velas de 2 minutos y unos **2 años** de velas de 1 hora. Cada consulta se
  une con lo que ya está en `datos/` (CSV por acción e intervalo), así que la historia de 2 minutos se va
  acumulando mientras el dashboard o el recolector estén corriendo.
- La bolsa de EE. UU. abre de 9:30 a 16:00, hora de Nueva York (8:30 a 15:00 en Bogotá mientras EE. UU. está en
  horario de verano; 9:30 a 16:00 en Bogotá de noviembre a marzo). Fuera de ese horario no llegan datos nuevos.
- La última vela que entrega Yahoo puede estar incompleta (la hora en curso). Para predecir y entrenar solo se
  usan velas completas.

### Qué se predice

Con las velas de **1 hora**: dado el cierre de la última vela completa, ¿la vela siguiente cerrará arriba o abajo?
Todos los modelos predicen el **log-retorno** de la vela siguiente, `ln(P_{t+1}) - ln(P_t)`, y de ahí salen el
precio predicho y la dirección. Se usan retornos y no precios porque el precio no es estacionario (la pestaña
*Análisis de la serie* lo muestra con la prueba ADF) y una red que predice precios aprende a copiar el último valor.

Las velas de 2 minutos se usan para la vista en vivo: permiten ver, durante la hora predicha, si el precio va en la
dirección que dijeron los modelos.

### Entrenamiento (`entrenar.py`)

1. **Características** por vela: log-retorno, rango `ln(High/Low)` y volumen relativo a su promedio reciente.
2. **División temporal** 70 / 15 / 15 (entrenamiento / validación / prueba), sin mezclar el orden. Entrenamiento +
   validación forman el tramo de *desarrollo*; la prueba solo se usa al final.
3. **Ventanas deslizantes**: para las redes, cada muestra son las últimas `ventana` velas; el objetivo es el
   retorno de la vela siguiente. Es la misma idea de los rezagos de un AR(p).
4. **Hiperparámetros**. AR y redes usan **validación cruzada temporal** (`TimeSeriesSplit`, 3 pliegues sobre el
   desarrollo): cada pliegue entrena con un bloque inicial y valida con el bloque siguiente, así nunca se entrena con
   datos posteriores a los de validación. Cada combinación se califica con el RMSE promedio de los pliegues y la
   mejor se reentrena con todo el desarrollo (las redes, durante el promedio de épocas que necesitaron en los
   pliegues). El escalado se ajusta en cada pliegue solo con su parte de entrenamiento. Prophet usa el tramo de
   validación, porque cada pliegue le exigiría decenas de ajustes más.
   - AR: `p` de 1 a 10.
   - Prophet: `changepoint_prior_scale` {0.01, 0.05, 0.3} × `seasonality_prior_scale` {0.1, 1, 10}.
   - RNN y LSTM: ventana {7, 14, 35 velas ≈ 1 día, 2 días, 1 semana} × unidades {32, 64} × capas {1, 2},
     con *early stopping*.
   Las rejillas (`REJILLA_REDES`, `PROPHET_REJILLA`) se cambian en `config.py`; en las redes también se puede
   agregar `dropout` y `tasa` de aprendizaje. Cada valor nuevo multiplica el tiempo de entrenamiento.
5. **Evaluación en prueba**: acierto de dirección con su margen de error al 95%, error medio del precio y RMSE del
   retorno. Prophet se evalúa reajustándose una vez por día de prueba, sin ver nunca datos futuros.

Todo queda en `modelos_entrenados/<TICKER>_1h/` y el dashboard lo carga automáticamente.

### Prophet en una línea

`log(precio) = tendencia(t) + efecto de la hora del día + efecto del día de la semana + ruido`. Es una regresión
sobre el tiempo: no usa los valores anteriores como AR o las redes. La pestaña *Prophet* muestra el pronóstico y
cada componente por separado.

### Pronóstico a varias horas (pestaña *Varias horas*)

Además de la próxima hora, el dashboard pronostica de 1 a 35 horas de mercado hacia adelante con el modelo ganador
de cada tipo. AR, RNN y LSTM lo hacen de forma **recursiva**: predicen una hora, toman esa predicción como dato y
siguen con la siguiente (en las redes, el rango y el volumen futuros se fijan en su promedio reciente, porque no los
predicen). Prophet evalúa su curva en cada hora futura. Se puede pronosticar desde la última vela o desde un momento
de las últimas 10 sesiones para compararlo con lo que pasó, y calcular el error promedio y el acierto de dirección
según el horizonte. La franja gris es el rango del 80% de una caminata aleatoria con la volatilidad reciente.

## Cómo leer los resultados

En acciones líquidas y a 1 hora, lo normal es que ningún modelo supere con claridad a *siempre sube*. Un acierto de
52% con un margen de ±4 puntos no es evidencia de poder predictivo. Ese resultado también es válido para el informe:
se puede discutir por qué (eficiencia del mercado, ruido) y comparar con otros horizontes.

## Limitaciones conocidas

- El horario está configurado para bolsas de EE. UU. (NYSE/Nasdaq); acciones de la BVC (`.CL`) tienen otro horario
  y muy poco volumen en velas de 2 minutos.
- No se descuentan los festivos de EE. UU. al calcular la "siguiente vela" (solo afecta la etiqueta de la hora).
- yfinance usa la web de Yahoo, que no es una API oficial: puede limitar las consultas (error 429) o cambiar sin
  aviso. El dashboard sigue mostrando los datos guardados cuando falla una descarga.

## Próximos pasos

- Comparar horizontes: `python entrenar.py --intervalo 30m` (solo hay ~60 días de historia a 30 minutos).
- Reglas de inversión (umbral de retorno predicho, confirmación entre modelos) y backtest contra comprar y mantener.
- Agente de Gemini: `prediccion.resumen_para_agente()` ya arma el texto con el estado de la acción y las
  predicciones; el dashboard lo muestra en *En vivo → Resumen de texto*.

*Proyecto académico. Nada de esto es una recomendación de inversión.*