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
python entrenar.py --rapido         # 2a. prueba rápida de que todo funciona (~2 min)
python entrenar.py                  # 2b. entrenamiento completo con búsqueda de hiperparámetros (1 hora o más)
python -m streamlit run app.py      # 3. dashboard
```

Para otra acción: `python entrenar.py --ticker MSFT`. La lista de la barra lateral solo muestra las acciones que ya
tienen modelos entrenados, así que la acción nueva aparece ahí después de entrenarla.

Para seguir recolectando datos con el dashboard cerrado, el recolector puede quedar corriendo en otra terminal:

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
   - RNN y LSTM: ventana {7, 14, 35 velas ≈ 1 día, 2 días, 1 semana} × unidades {32, 64} × capas {1, 2} ×
     tasa de aprendizaje {0.01, 0.05, 0.1}. Los pesos se ajustan con **descenso de gradiente estocástico** (SGD,
     sin momento) sobre lotes de 32 ventanas, minimizando el MSE, con dropout de 0.2 y *early stopping*
     (paciencia de 10 épocas, máximo 100).
   Las rejillas (`REJILLA_REDES`, `PROPHET_REJILLA`) y el optimizador se cambian en `config.py`. Cada valor nuevo
   multiplica el tiempo de entrenamiento.
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

### Estrategias de compra y venta (pestaña *Estrategias*)

Cada estrategia decide al cierre de cada hora si queda **comprada** o **fuera del mercado** durante la hora
siguiente, solo con información disponible hasta ese momento (no hay ventas en corto):

| Estrategia | Regla |
|---|---|
| Comprar y mantener | Compra al inicio y no vende: la referencia |
| Cruce de medias 20/50 | Comprada mientras el promedio de 20 horas está por encima del de 50 horas |
| RSI(14) 30/70 | Compra cuando el RSI baja de 30 y vende cuando supera 70 |
| Señal de cada modelo | Comprada cuando el retorno predicho para la próxima hora supera un umbral |
| Consenso de modelos | Comprada cuando más de la mitad de los modelos predice subida |

El backtest usa el periodo de prueba del entrenamiento, donde las predicciones de los modelos son fuera de muestra,
cobra un costo en cada compra y en cada venta (0,05% por defecto, ajustable en la pestaña) y reporta el retorno total,
la diferencia frente a comprar y mantener, la máxima caída, el número de operaciones, el porcentaje de operaciones
ganadoras, el tiempo invertido y el Sharpe anualizado. Los parámetros de las reglas están en `config.py`.

### Agente (pestaña *Agente*)

Chat que explica el estado de la acción. Los números los calcula el dashboard (predicción de la próxima hora,
pronóstico a 7 horas, métricas de prueba, posición y backtest de cada estrategia) y se envían a Gemini como contexto
en cada pregunta; sus instrucciones (`agente.py`) le piden usar solo esos datos y no dar recomendaciones personales.
La llamada al modelo se hace con **LiteLLM** (`litellm.completion` con el modelo `gemini/...`). Para usar el agente
hay que iniciar sesión, y cada cuenta tiene 20 preguntas por día.

**Guardrails (`guardrails.py`).**

| Momento | Qué revisa | Qué hace |
|---|---|---|
| Entrada | Nombres (Microsoft Presidio con spaCy en español, y frases como "me llamo…"), cédulas, correos, teléfonos y números de cuenta o tarjeta (expresiones regulares) | Los reemplaza por [NOMBRE], [DOCUMENTO], [CORREO], [TELEFONO] o [CUENTA] antes de enviar la pregunta |
| Entrada | Intentos de *prompt injection*: ignorar o revelar las instrucciones, cambiar el papel del agente, "modo desarrollador", marcas de mensajes de sistema | Bloquea la pregunta sin enviarla |
| Entrada | Preguntas de más de 500 caracteres | Las bloquea |
| Salida | Recomendaciones personales de inversión ("te recomiendo comprar", "deberías vender", "es buen momento para…") | Bloquea la respuesta y explica por qué |
| Salida | Cifras que no aparecen en los datos de la app ni en la pregunta | Las reemplaza por "[cifra no verificada]" |

Si Presidio o el modelo `es_core_news_sm` no están instalados, la detección de nombres sigue funcionando solo con las
frases y la pestaña lo indica. `python guardrails.py` muestra qué detector está activo y cómo anonimiza unos ejemplos.

**Uso, historial y auditoría (`registro_agente.py`).** La cuota se cuenta por cuenta y por día (hora de Bogotá). El
historial de conversaciones solo se guarda si el usuario lo activa en *Mis datos*, y siempre con las preguntas ya
anonimizadas. De cada pregunta se registra una auditoría con la fecha, el modelo, el resultado (respondida, bloqueada,
sin cuota, error) y el tipo de datos personales detectados, sin el texto y sin el usuario.

La clave de la API se guarda en `.streamlit/secrets.toml`, dentro de la carpeta del proyecto, o en los *Secrets* de
la app publicada (también sirve la variable de entorno `GEMINI_API_KEY`). Ese archivo está en `.gitignore`:

```toml
GEMINI_API_KEY = "clave-de-gemini"
```

La clave solo existe en el servidor: los usuarios nunca la ven ni la usan directamente. Si se llegara a exponer, se
crea una nueva en Google AI Studio, se reemplaza en los *Secrets* y se borra la anterior. Los modelos de Gemini
disponibles en la pestaña se definen en `GEMINI_MODELOS` (`config.py`).

### Cuentas y portafolio simulado (pestañas *Portafolio simulado* y *Mis datos*)

Cada usuario se registra con alias, correo y contraseña y recibe 10.000 USD ficticios para comprar y vender las
acciones que tienen modelos entrenados. Las reglas están en `portafolio.py` y en la pestaña:

- Solo con el mercado abierto y al último precio de 2 minutos, que debe tener menos de 10 minutos.
- Acciones enteras, comisión simulada del 0,1% por operación, sin comprar sin saldo ni vender lo que no se tiene.
- Antes de confirmar se muestran la predicción de cada modelo y el consenso; la decisión es del usuario.
- El rendimiento se compara con comprar y mantener desde el inicio del portafolio y se cuenta cuántas operaciones
  siguieron al consenso de los modelos.

Cada operación se valida en el servidor dentro de la misma transacción que la registra, así que el efectivo solo
cambia por una operación válida; además, la pestaña verifica que el efectivo guardado coincida con la suma de las
operaciones. En *Mis datos* el usuario ve todo lo que se guarda sobre él, lo descarga, corrige su alias y su correo,
cambia su contraseña, reinicia el portafolio o elimina su cuenta (Ley 1581). La política de tratamiento de datos está
en `politica_datos.md` y se muestra en el registro y al pie de la aplicación.

**Base de datos (`base_datos.py`).** En el equipo local se usa SQLite (`base_datos/usuarios.db`, fuera del
repositorio). En Streamlit Community Cloud el disco se borra con cada reinicio, así que allá se usa una base Postgres
gratuita (por ejemplo, de [Neon](https://neon.com)): basta con agregar estas líneas en los *Secrets* de la app:

```toml
DATABASE_URL = "postgresql://usuario:contraseña@servidor/basededatos?sslmode=require"
CLAVE_DATOS = "una frase larga y aleatoria"
```

`CLAVE_DATOS` cifra los correos; si se cambia o se pierde, los correos guardados ya no se pueden leer y nadie podría
iniciar sesión, así que se define una sola vez. Las contraseñas se guardan con hash bcrypt, la cuenta se bloquea 15
minutos tras 5 intentos fallidos y la sesión se cierra tras 30 minutos sin actividad.

## Cómo leer los resultados

En acciones líquidas y a 1 hora, lo normal es que ningún modelo supere con claridad a *siempre sube*. Un acierto de
52% con un margen de ±4 puntos no es evidencia de poder predictivo. Es lo esperable en un mercado eficiente, donde el
movimiento de una hora a otra es casi todo ruido.

## Limitaciones conocidas

- El horario está configurado para bolsas de EE. UU. (NYSE/Nasdaq); acciones de la BVC (`.CL`) tienen otro horario
  y muy poco volumen en velas de 2 minutos.
- No se descuentan los festivos de EE. UU. al calcular la "siguiente vela" (solo afecta la etiqueta de la hora).
- yfinance usa la web de Yahoo, que no es una API oficial: puede limitar las consultas (error 429) o cambiar sin
  aviso. El dashboard sigue mostrando los datos guardados cuando falla una descarga.

## Próximos pasos

- Comparar horizontes: `python entrenar.py --intervalo 30m` (solo hay ~60 días de historia a 30 minutos).
- Prueba de McNemar para comparar el acierto de cada modelo con "Siempre sube" sobre las mismas velas.
- Variables de contexto del mercado (retorno de SPY o QQQ, VIX) como entradas adicionales de los modelos.

*Proyecto académico. Nada de esto es una recomendación de inversión.*
