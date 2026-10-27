"""Red convolucional (CNN) que mira "fotos" de las últimas velas y clasifica si la vela siguiente sube o baja.

Cada imagen es un gráfico de velas de las últimas N horas, como el que vería una persona: verde si la vela cerró por
encima de su apertura y roja si cerró por debajo, con el cuerpo (de la apertura al cierre) y la mecha (del mínimo al
máximo). La escala vertical va del mínimo al máximo de esas N velas, así que la red ve la forma del movimiento y no el
nivel del precio.

A diferencia de la RNN y la LSTM, que hacen regresión del log-retorno, la CNN es un modelo de clasificación: su salida
(sigmoide) es la probabilidad de que la vela siguiente cierre por encima del cierre actual. Se entrena con descenso de
gradiente estocástico (SGD), entropía cruzada binaria y la misma validación cruzada temporal que las demás redes.
"""

import itertools
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import config
import preparacion

NOMBRE = "CNN de velas"
COLUMNAS = ["Open", "High", "Low", "Close"]
VERDE = (0, 200, 0)
ROJO = (220, 0, 0)


def _keras():
    from tensorflow import keras
    return keras


# ------------------------------------------------------------------ imágenes

def dibujar(ohlc, alto=config.CNN_ALTO_IMAGEN, ancho_vela=config.CNN_ANCHO_VELA):
    """Imagen RGB (alto × velas·ancho_vela × 3, uint8) de las velas dadas como filas [apertura, máximo, mínimo, cierre]."""
    ohlc = np.asarray(ohlc, dtype=float)
    imagen = np.zeros((alto, len(ohlc) * ancho_vela, 3), dtype=np.uint8)
    minimo, maximo = ohlc[:, 2].min(), ohlc[:, 1].max()
    escala = (alto - 1) / (maximo - minimo) if maximo > minimo else 0.0

    def fila(precio):
        if escala == 0:
            return alto // 2
        return int(alto - 1 - round((precio - minimo) * escala))

    for i, (apertura, techo, piso, cierre) in enumerate(ohlc):
        color = VERDE if cierre >= apertura else ROJO
        x = i * ancho_vela
        imagen[fila(techo):fila(piso) + 1, x + (ancho_vela - 1) // 2] = color
        arriba, abajo = sorted((fila(apertura), fila(cierre)))
        imagen[arriba:abajo + 1, x:x + ancho_vela - 1] = color
    return imagen


def imagenes(ohlc, velas):
    """Una imagen por cada posición j con al menos `velas` velas: la que termina en la vela j."""
    ohlc = np.asarray(ohlc, dtype=float)
    return np.stack([dibujar(ohlc[j - velas + 1: j + 1]) for j in range(velas - 1, len(ohlc))])


def ampliar(imagen, factor=6):
    """Imagen agrandada sin suavizar, para mostrarla en el dashboard."""
    return np.repeat(np.repeat(imagen, factor, axis=0), factor, axis=1)


class Conjunto:
    """Imágenes y etiquetas alineadas con las filas de las características `f` (las mismas que usan los demás modelos).

    La etiqueta de la fila t es 1 si el log-retorno de la vela siguiente es positivo.
    """

    def __init__(self, velas_df, f, velas):
        self.velas = velas
        self.imagenes = imagenes(velas_df[COLUMNAS].to_numpy(dtype=float), velas)
        self.posiciones = velas_df.index.get_indexer(f.index) - (velas - 1)
        self.y = (f["objetivo"].to_numpy(dtype=float) > 0).astype("float32")

    def tramo(self, a, b):
        filas = np.arange(a, b)
        filas = filas[self.posiciones[filas] >= 0]
        return self.imagenes[self.posiciones[filas]], self.y[filas], filas


# ------------------------------------------------------------------ modelo

def construir_modelo(alto, ancho, filtros, tasa, dropout=0.2):
    keras = _keras()
    modelo = keras.Sequential(name="cnn_velas")
    modelo.add(keras.Input(shape=(alto, ancho, 3)))
    modelo.add(keras.layers.Rescaling(1 / 255))
    modelo.add(keras.layers.Conv2D(filtros, 3, padding="same", activation="relu"))
    modelo.add(keras.layers.MaxPooling2D(2))
    modelo.add(keras.layers.Conv2D(2 * filtros, 3, padding="same", activation="relu"))
    modelo.add(keras.layers.MaxPooling2D(2))
    modelo.add(keras.layers.Flatten())
    modelo.add(keras.layers.Dropout(dropout))
    modelo.add(keras.layers.Dense(1, activation="sigmoid"))
    modelo.compile(optimizer=keras.optimizers.SGD(learning_rate=tasa), loss="binary_crossentropy",
                   metrics=["accuracy"])
    return modelo


def _entrenar(modelo, x_tr, y_tr, x_va, y_va):
    keras = _keras()
    parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PACIENCIA, restore_best_weights=True)
    historial = modelo.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=config.EPOCAS,
                           batch_size=config.TAMANO_LOTE, callbacks=[parada], shuffle=True, verbose=0)
    return pd.DataFrame(historial.history)


def _puntaje(fila):
    valor = fila["perdida_cv"]
    return valor if np.isfinite(valor) else np.inf


def buscar_hiperparametros(velas_df, f, cortes, rejilla=config.REJILLA_CNN, k=config.PLIEGUES_CV, al_terminar=None):
    """Búsqueda en rejilla con validación cruzada temporal (TimeSeriesSplit), como en las redes recurrentes.

    Cada combinación se califica con la entropía cruzada promedio en validación. La mejor se reentrena con todo el
    tramo de desarrollo durante el número de épocas promedio de sus pliegues.
    """
    keras = _keras()
    pliegues = preparacion.pliegues_temporales(cortes, k)
    conjuntos = {velas: Conjunto(velas_df, f, velas) for velas in rejilla["velas"]}
    combinaciones = list(itertools.product(rejilla["velas"], rejilla["filtros"], rejilla["tasa"]))
    filas, mejor_fila, historiales_mejor = [], None, None
    for numero, (velas, filtros, tasa) in enumerate(combinaciones, start=1):
        conjunto = conjuntos[velas]
        perdidas, aciertos, epocas, historiales = [], [], [], []
        for numero_pliegue, pliegue in enumerate(pliegues, start=1):
            keras.utils.set_random_seed(config.SEMILLA)
            x_tr, y_tr, _ = conjunto.tramo(0, pliegue.fin_entrenamiento)
            x_va, y_va, _ = conjunto.tramo(*pliegue.tramo("validacion"))
            modelo = construir_modelo(x_tr.shape[1], x_tr.shape[2], filtros, tasa, config.CNN_DROPOUT)
            historial = _entrenar(modelo, x_tr, y_tr, x_va, y_va)
            perdida, acierto = modelo.evaluate(x_va, y_va, verbose=0)
            perdidas.append(float(perdida))
            aciertos.append(float(acierto))
            valores = historial["val_loss"].to_numpy(dtype=float)
            epocas.append(int(np.nanargmin(valores)) + 1 if np.isfinite(valores).any() else 1)
            historiales.append(historial.assign(pliegue=str(numero_pliegue)))
        fila = {
            "velas": velas, "filtros": filtros, "tasa": tasa, "dropout": config.CNN_DROPOUT,
            "perdida_cv": float(np.mean(perdidas)), "desv_pliegues": float(np.std(perdidas)),
            "acierto_cv_pct": 100 * float(np.mean(aciertos)), "epocas_promedio": float(np.mean(epocas)),
        }
        filas.append(fila)
        if al_terminar:
            al_terminar(numero, len(combinaciones), fila)
        if mejor_fila is None or _puntaje(fila) < _puntaje(mejor_fila):
            mejor_fila, historiales_mejor = fila, historiales

    keras.utils.set_random_seed(config.SEMILLA)
    conjunto = conjuntos[mejor_fila["velas"]]
    x_dev, y_dev, _ = conjunto.tramo(0, cortes.fin_validacion)
    modelo = construir_modelo(x_dev.shape[1], x_dev.shape[2], mejor_fila["filtros"], mejor_fila["tasa"],
                              config.CNN_DROPOUT)
    epocas_finales = max(1, int(round(mejor_fila["epocas_promedio"])))
    final = modelo.fit(x_dev, y_dev, epochs=epocas_finales, batch_size=config.TAMANO_LOTE, shuffle=True, verbose=0)
    historial = pd.concat(historiales_mejor + [pd.DataFrame(final.history).assign(pliegue="final")])
    historial = historial.rename_axis("epoca").reset_index()
    mejor = {"fila": {**mejor_fila, "epocas_finales": epocas_finales, "optimizador": config.OPTIMIZADOR,
                      "alto": config.CNN_ALTO_IMAGEN, "ancho_vela": config.CNN_ANCHO_VELA},
             "modelo": modelo, "historial": historial, "conjunto": conjunto}
    tabla = pd.DataFrame(filas).sort_values("perdida_cv").reset_index(drop=True)
    return mejor, tabla


def probabilidades(modelo, conjunto, a, b):
    """Probabilidad de que suba la vela siguiente para cada fila [a, b) de f (NaN si no hay suficientes velas)."""
    x, _, filas = conjunto.tramo(a, b)
    salida = np.full(b - a, np.nan)
    if len(filas):
        salida[filas - a] = modelo.predict(x, verbose=0).ravel()
    return salida


def foto_actual(velas_df, velas):
    """Imagen de las últimas `velas` velas completas: lo que mira la CNN para predecir la hora siguiente."""
    return dibujar(velas_df[COLUMNAS].to_numpy(dtype=float)[-velas:])


def probabilidad_actual(modelo, velas_df, velas):
    imagen = foto_actual(velas_df, velas)
    return float(modelo.predict(imagen[None, ...], verbose=0).ravel()[0]), imagen


def guardar_modelo(modelo, ruta):
    modelo.save(str(ruta))


def cargar_modelo(ruta):
    return _keras().models.load_model(str(ruta), compile=False)
