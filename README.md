# video_sentiment_analysis
The aim of this exercise is to show that classical Convolutional Neural Networks (CNN) are not efficient enough
(in terms of accuracy) for video sentiment analysis. Although these networks can use residual datas,
they do not analyze the input on a recurring basis.

## Estado

Se implementó y probó la parte de CNN: entrenamiento, evaluación y reconocimiento de emociones en vivo con webcam.
La comparación con un modelo recurrente (CNN + LSTM/GRU) **no se realizó**, así que la hipótesis de arriba queda sin
contrastar. Lo único medido es el rendimiento de la CNN por imagen.

## Qué contiene

| Archivo | Función |
| --- | --- |
| `data.py` | Crea `sentiment_dataset_clean` a partir de `sentiment_dataset`, sin duplicados ni fugas entre train y validation |
| `model.py` | Define, entrena y evalúa la CNN (48x48, escala de grises, 6 clases) |
| `emotion.py` | Preprocesado de caras, seguimiento entre cuadros y suavizado temporal de las predicciones |
| `video_reader.py` | Reconocimiento en tiempo real con webcam o sobre un archivo de video |
| `tests/` | Pruebas unitarias |

Clases: Angry, Fear, Happy, Neutral, Sad, Surprise.

## Uso

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python data.py                                  # crea sentiment_dataset_clean
.venv/bin/python model.py train                           # entrena y guarda EmotionDetectionModel.keras
.venv/bin/python model.py evaluate EmotionDetectionModel.keras
.venv/bin/python video_reader.py                          # webcam en vivo (q o Esc para salir)
.venv/bin/python video_reader.py --source video.mp4 --output salida.mp4 --no-display
.venv/bin/python -m unittest discover -s tests -t .       # pruebas
```

En macOS, la terminal necesita permiso de cámara para usar la webcam.

## Problemas del conjunto de datos original

La validación original no es fiable: hay imágenes repetidas, imágenes de validación que también están en entrenamiento
y `validation/Neutral` está formada casi por completo por copias de entrenamiento de otras emociones. `data.py` genera una
copia limpia y reparte de nuevo las imágenes únicas (estratificado por clase, semilla fija). Detalles en el docstring del módulo.

## Resultados

Validación sobre el conjunto limpio (2936 imágenes):

- Exactitud **0.626** (la clase mayoritaria daría 0.266), F1 macro **0.603**.

| Clase | Recall | Precisión |
| --- | --- | --- |
| Angry | 0.632 | 0.510 |
| Fear | 0.340 | 0.482 |
| Happy | 0.791 | 0.896 |
| Neutral | 0.689 | 0.510 |
| Sad | 0.459 | 0.592 |
| Surprise | 0.790 | 0.651 |

Happy y Surprise son las más fiables. Fear es la peor y se confunde sobre todo con Sad, Neutral y Surprise; Sad también
se confunde con Neutral.

## Limitaciones

- La exactitud por imagen (~63%) es modesta; en la webcam se nota en Fear y Sad.
- Cada cuadro se clasifica de forma independiente. El suavizado de `emotion.py` solo estabiliza la etiqueta con una media
  móvil, no modela la dinámica temporal.
- Falta la comparación con un modelo recurrente.
