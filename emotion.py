"""
Piezas de la deteccion de emociones en video: preparar una cara para el modelo, seguir las caras entre cuadros y suavizar
sus predicciones, y dibujar el resultado. No dependen de TensorFlow, asi que se pueden probar por separado.
"""
import numpy as np
import cv2

from data import CLASSES

IMG_SIZE = 48
COLOR = (0, 255, 0)


def preprocessFace(gray, box):
    """Recorta la cara (x, y, w, h) de una imagen en gris y la deja como el modelo la espera: (48, 48, 1) en [0, 1]."""
    x, y, w, h = box
    face = cv2.resize(gray[y:y + h, x:x + w], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return (face.astype('float32') / 255.0)[:, :, None]


class FaceTracker:
    """
    Asocia las caras de cada cuadro con las del anterior (por cercania de los centros) y suaviza sus probabilidades con una
    media movil exponencial: p = alpha * nueva + (1 - alpha) * anterior. Sin esto la etiqueta parpadea de un cuadro a otro.
    """

    def __init__(self, alpha=0.4, max_distance=0.75, max_missed=8):
        self.alpha = alpha
        self.max_distance = max_distance  # distancia maxima entre centros, en tamanos de cara
        self.max_missed = max_missed      # cuadros sin ver una cara antes de olvidarla
        self.tracks = {}
        self.next_id = 0

    @staticmethod
    def _center(box):
        x, y, w, h = box
        return np.array([x + w / 2.0, y + h / 2.0])

    def update(self, boxes, probabilities):
        """
        boxes: lista de (x, y, w, h); probabilities: matriz (n_caras, n_clases) con la salida del modelo.
        Devuelve, por cada cara, (id, caja, probabilidades suavizadas).
        """
        unmatched = set(self.tracks)
        results = []
        for box, probs in zip(boxes, probabilities):
            center, size = self._center(box), max(box[2], box[3])
            best, best_distance = None, self.max_distance * size
            for track_id in unmatched:
                distance = np.linalg.norm(center - self.tracks[track_id]['center'])
                if distance < best_distance:
                    best, best_distance = track_id, distance
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {'probs': np.asarray(probs, dtype=float)}
            else:
                unmatched.discard(best)
                old = self.tracks[best]['probs']
                self.tracks[best]['probs'] = self.alpha * np.asarray(probs, dtype=float) + (1 - self.alpha) * old
            self.tracks[best]['center'] = center
            self.tracks[best]['missed'] = 0
            results.append((best, box, self.tracks[best]['probs']))
        for track_id in unmatched:
            self.tracks[track_id]['missed'] += 1
            if self.tracks[track_id]['missed'] > self.max_missed:
                del self.tracks[track_id]
        return results


def label(probs):
    """Texto 'Emocion 83%' de un vector de probabilidades."""
    k = int(np.argmax(probs))
    return f"{CLASSES[k]} {probs[k] * 100:.0f}%"


def drawProbabilities(frame, probs, origin=(10, 10), width=150, bar_height=14):
    """Panel con una barra por emocion, para ver como se reparte la prediccion."""
    x0, y0 = origin
    height = len(CLASSES) * (bar_height + 4) + 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width + 90, y0 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for k, cls in enumerate(CLASSES):
        y = y0 + 6 + k * (bar_height + 4)
        cv2.rectangle(frame, (x0 + 80, y), (x0 + 80 + int(width * float(probs[k])), y + bar_height),
                      COLOR if k == int(np.argmax(probs)) else (180, 180, 180), -1)
        cv2.putText(frame, cls, (x0 + 4, y + bar_height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return frame


def annotateFrame(frame, detector, predict, tracker, min_face=60):
    """
    Detecta las caras de un cuadro (BGR), las clasifica y dibuja el resultado.
    detector: algo con detectMultiScale (un CascadeClassifier); predict: funcion (n, 48, 48, 1) -> (n, n_clases).
    Devuelve la lista de (id, caja, probabilidades suavizadas) de las caras de este cuadro.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face))
    boxes = [tuple(int(v) for v in box) for box in boxes]
    if boxes:
        probabilities = predict(np.stack([preprocessFace(gray, box) for box in boxes]))
    else:
        probabilities = np.zeros((0, len(CLASSES)))
    results = tracker.update(boxes, probabilities)
    for _, (x, y, w, h), probs in results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR, 2)
        text = label(probs)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        top = max(y - th - baseline - 6, 0)
        cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + baseline + 6), COLOR, -1)  # fondo para que se lea
        cv2.putText(frame, text, (x + 4, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    if results:  # panel de barras de la cara mas grande
        drawProbabilities(frame, max(results, key=lambda r: r[1][2] * r[1][3])[2])
    return results
