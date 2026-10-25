"""Pruebas de las funciones de model.py. Ejecutar con:  python -m unittest discover -s tests -t ."""
import unittest

import numpy as np

try:
    import model
except ImportError:  # sin TensorFlow/Keras
    model = None


@unittest.skipIf(model is None, 'TensorFlow/Keras no esta instalado')
class ModelFunctionsTest(unittest.TestCase):

    def test_class_weights_are_inverse_to_frequency(self):
        weights = model.classWeights([100, 200, 400, 100, 100, 100])
        self.assertAlmostEqual(weights[2], 1000 / (6 * 400))
        self.assertGreater(weights[0], weights[2])
        # el peso total ponderado por clase es el mismo para todas las clases
        counts = np.array([100, 200, 400, 100, 100, 100])
        np.testing.assert_allclose([weights[k] * counts[k] for k in range(6)], 1000 / 6)

    def test_metrics_from_known_confusion(self):
        y_true = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
        y_pred = [0, 0, 0, 1, 1, 0, 2, 2, 2, 2]
        m = model.metricsFrom(y_true, y_pred, n_classes=3)
        self.assertAlmostEqual(m['accuracy'], 0.8)
        self.assertAlmostEqual(m['majority_class_accuracy'], 0.4)
        self.assertEqual(m['confusion_matrix'], [[3, 1, 0], [1, 1, 0], [0, 0, 4]])

    def test_empty_class_does_not_divide_by_zero(self):
        m = model.metricsFrom([0, 0, 1], [0, 0, 1], n_classes=3)   # la clase 2 no aparece
        self.assertEqual(m['accuracy'], 1.0)

    def test_model_outputs_probabilities(self):
        net = model.buildModel()
        out = net(np.random.rand(3, 48, 48, 1).astype('float32'), training=False).numpy()
        self.assertEqual(out.shape, (3, 6))
        np.testing.assert_allclose(out.sum(axis=1), 1, atol=1e-5)

    def test_augmentation_is_only_active_in_training(self):
        net = model.buildModel(augment=True)
        x = np.random.rand(2, 48, 48, 1).astype('float32')
        a = net(x, training=False).numpy()
        b = net(x, training=False).numpy()
        np.testing.assert_allclose(a, b)           # determinista en inferencia


if __name__ == '__main__':
    unittest.main()
