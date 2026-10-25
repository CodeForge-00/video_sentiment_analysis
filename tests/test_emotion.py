"""Pruebas de la logica de video (sin TensorFlow). Ejecutar con:  python -m unittest discover -s tests -t ."""
import unittest

import cv2
import numpy as np

from data import CLASSES
from emotion import FaceTracker, annotateFrame, drawProbabilities, label, preprocessFace


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.last_gray = None

    def detectMultiScale(self, gray, **kwargs):
        self.last_gray = gray
        self.kwargs = kwargs
        return np.array(self.boxes, dtype=int).reshape(-1, 4)


def onehot(k, n=len(CLASSES)):
    v = np.zeros(n)
    v[k] = 1.0
    return v


class PreprocessTest(unittest.TestCase):

    def test_shape_range_and_crop(self):
        gray = np.zeros((200, 300), dtype=np.uint8)
        gray[50:150, 100:200] = 255          # la "cara"
        face = preprocessFace(gray, (100, 50, 100, 100))
        self.assertEqual(face.shape, (48, 48, 1))
        self.assertEqual(face.dtype, np.float32)
        self.assertTrue(np.allclose(face, 1.0))
        self.assertTrue(np.allclose(preprocessFace(gray, (0, 0, 40, 40)), 0.0))


class TrackerTest(unittest.TestCase):

    def test_same_face_keeps_id_and_smooths(self):
        tracker = FaceTracker(alpha=0.5)
        (id1, _, p1), = tracker.update([(100, 100, 80, 80)], [onehot(0)])
        (id2, _, p2), = tracker.update([(104, 102, 80, 80)], [onehot(1)])  # se movio un poco
        self.assertEqual(id1, id2)
        np.testing.assert_allclose(p1, onehot(0))
        np.testing.assert_allclose(p2, 0.5 * onehot(1) + 0.5 * onehot(0))

    def test_smoothing_stops_flicker(self):
        """Un cuadro aislado con otra emocion no cambia la etiqueta mostrada."""
        tracker = FaceTracker(alpha=0.3)
        for _ in range(10):
            tracker.update([(50, 50, 90, 90)], [onehot(2)])
        (_, _, probs), = tracker.update([(50, 50, 90, 90)], [onehot(4)])
        self.assertEqual(int(np.argmax(probs)), 2)

    def test_two_faces_are_tracked_independently(self):
        tracker = FaceTracker(alpha=1.0)
        a, b = (50, 50, 80, 80), (400, 60, 80, 80)
        first = tracker.update([a, b], [onehot(0), onehot(1)])
        second = tracker.update([b, a], [onehot(1), onehot(0)])   # distinto orden de deteccion
        ids = {box: i for i, box, _ in first}
        self.assertEqual({box: i for i, box, _ in second}, ids)
        self.assertNotEqual(ids[a], ids[b])

    def test_new_face_far_away_gets_new_id(self):
        tracker = FaceTracker()
        (i1, _, _), = tracker.update([(10, 10, 60, 60)], [onehot(0)])
        (i2, _, _), = tracker.update([(500, 300, 60, 60)], [onehot(0)])
        self.assertNotEqual(i1, i2)

    def test_lost_faces_are_forgotten(self):
        tracker = FaceTracker(max_missed=2)
        tracker.update([(10, 10, 60, 60)], [onehot(0)])
        for _ in range(3):
            tracker.update([], np.zeros((0, len(CLASSES))))
        self.assertEqual(tracker.tracks, {})

    def test_face_reappearing_soon_keeps_its_track(self):
        tracker = FaceTracker(max_missed=5)
        (i1, _, _), = tracker.update([(10, 10, 60, 60)], [onehot(0)])
        tracker.update([], np.zeros((0, len(CLASSES))))
        (i2, _, _), = tracker.update([(12, 11, 60, 60)], [onehot(0)])
        self.assertEqual(i1, i2)


class DrawingAndPipelineTest(unittest.TestCase):

    def test_label_text(self):
        probs = np.array([0.1, 0.05, 0.8, 0.02, 0.02, 0.01])
        self.assertEqual(label(probs), 'Happy 80%')

    def test_draw_probabilities_changes_only_the_panel(self):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        before = frame.copy()
        drawProbabilities(frame, np.full(len(CLASSES), 1 / len(CLASSES)))
        self.assertTrue((frame != before).any())
        self.assertTrue((frame[200:, 250:] == 0).all())   # lejos del panel no cambia nada

    def test_frame_without_faces_is_left_alone(self):
        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        before = frame.copy()
        calls = []
        results = annotateFrame(frame, FakeDetector([]), lambda b: calls.append(b) or np.zeros((len(b), 6)), FaceTracker())
        self.assertEqual(results, [])
        self.assertEqual(calls, [])                  # el modelo no se llama sin caras
        np.testing.assert_array_equal(frame, before)

    def test_frame_with_faces_is_classified_in_one_batch_and_annotated(self):
        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        batches = []

        def predict(batch):
            batches.append(batch.shape)
            return np.stack([onehot(2), onehot(5)])

        detector = FakeDetector([(20, 40, 80, 80), (200, 50, 70, 70)])
        results = annotateFrame(frame, detector, predict, FaceTracker(), min_face=60)
        self.assertEqual(batches, [(2, 48, 48, 1)])   # una sola llamada para las dos caras
        self.assertEqual([int(np.argmax(p)) for _, _, p in results], [2, 5])
        self.assertEqual(detector.kwargs['minSize'], (60, 60))
        self.assertTrue((frame != 100).any())


if __name__ == '__main__':
    unittest.main()
