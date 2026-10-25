"""
Reconocimiento de emociones en tiempo real con la webcam (o sobre un archivo de video).

  python video_reader.py                       # webcam 0
  python video_reader.py --source video.mp4 --output salida.mp4 --no-display
  python video_reader.py --model EmotionDetectionModel.h5

Teclas: q o Esc para salir.
"""
import argparse
import os
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import cv2
import keras

from emotion import FaceTracker, annotateFrame

HERE = os.path.dirname(os.path.abspath(__file__))


def defaultModel():
    for name in ('EmotionDetectionModel.keras', 'EmotionDetectionModel.h5'):
        if os.path.exists(os.path.join(HERE, name)):
            return os.path.join(HERE, name)
    raise SystemExit('No hay modelo: entrene uno con "python model.py train" o indique --model')


def loadDetector():
    path = os.path.join(HERE, 'modelo', 'haarcascade_frontalface_default.xml')
    if not os.path.exists(path):
        path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise SystemExit(f'No se pudo cargar el detector de caras: {path}')
    return detector


def main():
    parser = argparse.ArgumentParser(description='Emociones en tiempo real')
    parser.add_argument('--model', default=None, help='modelo .keras o .h5 (por defecto el que haya en esta carpeta)')
    parser.add_argument('--source', default='0', help='indice de la camara (0, 1...) o ruta de un archivo de video')
    parser.add_argument('--output', help='guardar el video con las anotaciones')
    parser.add_argument('--no-display', action='store_true', help='no abrir ventana (util con --output)')
    parser.add_argument('--no-mirror', action='store_true', help='con webcam, no voltear la imagen como un espejo')
    parser.add_argument('--smooth', type=float, default=0.4,
                        help='peso de cada cuadro nuevo en la media movil (1 = sin suavizado, menor = mas estable)')
    parser.add_argument('--min-face', type=int, default=60, help='tamano minimo de cara en pixeles')
    parser.add_argument('--max-frames', type=int, default=0, help='parar tras N cuadros (0 = sin limite)')
    args = parser.parse_args()

    model = keras.models.load_model(args.model or defaultModel(), compile=False)
    detector = loadDetector()
    tracker = FaceTracker(alpha=args.smooth)
    predict = lambda batch: model(batch, training=False).numpy()

    is_camera = args.source.isdigit()
    capture = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    if not capture.isOpened():
        raise SystemExit('No se pudo abrir la camara/video. En macOS: Ajustes del Sistema > Privacidad y seguridad > '
                         'Camara, y permita el acceso a la terminal o al editor desde el que ejecuta esto.')
    writer = None
    frames, started = 0, time.time()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:  # fin del video o la camara dejo de responder
                break
            if is_camera and not args.no_mirror:
                frame = cv2.flip(frame, 1)
            annotateFrame(frame, detector, predict, tracker, args.min_face)
            frames += 1
            fps = frames / max(time.time() - started, 1e-6)
            cv2.putText(frame, f"{fps:.0f} fps", (frame.shape[1] - 90, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if args.output:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'),
                                             capture.get(cv2.CAP_PROP_FPS) or 20.0, (width, height))
                writer.write(frame)
            if not args.no_display:
                cv2.imshow('Emociones (q para salir)', frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print(f"{frames} cuadros procesados ({frames / max(time.time() - started, 1e-6):.1f} cuadros por segundo)")


if __name__ == '__main__':
    main()
