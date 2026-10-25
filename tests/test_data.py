"""Pruebas de la limpieza del conjunto de datos. Ejecutar con:  python -m unittest discover -s tests -t ."""
import os
import tempfile
import unittest

import cv2
import numpy as np

from data import buildClean, countImages, scan, hashFile, MARKER


def makeImage(path, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.random.RandomState(seed).randint(0, 255, (48, 48), dtype=np.uint8))


class CleanDatasetTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, 'src')
        self.dst = os.path.join(self.tmp.name, 'dst')
        p = lambda split, cls, name: os.path.join(self.src, split, cls, name)
        # contenido 1: una copia en train (Happy)
        makeImage(p('train', 'Happy', 'a.png'), 1)
        # contenido 2: duplicado dentro de train con la misma etiqueta
        makeImage(p('train', 'Sad', 'b1.png'), 2)
        makeImage(p('train', 'Sad', 'b2.png'), 2)
        # contenido 3: en train (Angry) y en validacion con OTRA etiqueta (Neutral): se confia en train
        makeImage(p('train', 'Angry', 'c.png'), 3)
        makeImage(p('validation', 'Neutral', 'c_val.png'), 3)
        # contenido 4: en train (Fear) y en validacion con la misma etiqueta: fuga, se quita de validacion
        makeImage(p('train', 'Fear', 'd.png'), 4)
        makeImage(p('validation', 'Fear', 'd_val.png'), 4)
        # contenido 5: dentro de train con etiquetas distintas: ambiguo, se descarta
        makeImage(p('train', 'Happy', 'e1.png'), 5)
        makeImage(p('train', 'Sad', 'e2.png'), 5)
        # contenido 6: solo en validacion, duplicado: se conserva una copia
        makeImage(p('validation', 'Happy', 'f1.png'), 6)
        makeImage(p('validation', 'Happy', 'f2.png'), 6)
        # contenido 7: solo en validacion con etiquetas distintas: ambiguo
        makeImage(p('validation', 'Happy', 'g1.png'), 7)
        makeImage(p('validation', 'Fear', 'g2.png'), 7)
        # contenido 8: solo en validacion, unico
        makeImage(p('validation', 'Surprise', 'h.png'), 8)

    def tearDown(self):
        self.tmp.cleanup()

    def files(self, root):
        return sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs if f != MARKER)

    def test_cleaning_without_resplit(self):
        removed = buildClean(self.src, self.dst, val_fraction=None)
        kept = self.files(self.dst)
        self.assertEqual(len(kept), 6)
        self.assertEqual(removed, {'duplicate_in_train': 1, 'validation_copy_of_train': 2, 'ambiguous_label': 4,
                                   'duplicate_in_validation': 1})
        counts = countImages(self.dst)
        self.assertEqual(counts['train'], {'Angry': 1, 'Fear': 1, 'Happy': 1, 'Sad': 1})
        self.assertEqual(counts['validation'], {'Happy': 1, 'Surprise': 1})
        # ningun contenido aparece dos veces y ninguno esta en las dos particiones
        contents = scan(self.dst)
        self.assertTrue(all(len(entries) == 1 for entries in contents.values()))
        # el contenido 3 conserva la etiqueta de entrenamiento
        self.assertTrue(os.path.exists(os.path.join(self.dst, 'train', 'Angry', 'c.png')))
        self.assertFalse(os.path.exists(os.path.join(self.dst, 'validation', 'Neutral')))

    def test_original_is_not_modified(self):
        before = self.files(self.src)
        buildClean(self.src, self.dst, val_fraction=None)
        self.assertEqual(before, self.files(self.src))

    def test_resplit_is_stratified_deterministic_and_disjoint(self):
        for k in range(20):  # mas imagenes de una clase para poder estratificar
            makeImage(os.path.join(self.src, 'train', 'Happy', f'x{k}.png'), 100 + k)
        buildClean(self.src, self.dst, val_fraction=0.25, seed=3)
        first = self.files(self.dst)
        buildClean(self.src, self.dst, val_fraction=0.25, seed=3, overwrite=True)
        self.assertEqual(first, self.files(self.dst))
        counts = countImages(self.dst)
        self.assertEqual(counts['validation']['Happy'], round((21 + 1) * 0.25))  # a + 20 x + una copia de f
        hashes = {h: {s for s, _, _ in entries} for h, entries in scan(self.dst).items()}
        self.assertTrue(all(len(splits) == 1 for splits in hashes.values()))

    def test_refuses_to_overwrite_a_folder_it_did_not_create(self):
        os.makedirs(self.dst)
        with open(os.path.join(self.dst, 'importante.txt'), 'w') as f:
            f.write('no borrar')
        with self.assertRaises(FileExistsError):
            buildClean(self.src, self.dst, val_fraction=None, overwrite=True)
        self.assertTrue(os.path.exists(os.path.join(self.dst, 'importante.txt')))

    def test_hash_depends_only_on_content(self):
        a = os.path.join(self.tmp.name, 'x.bin')
        b = os.path.join(self.tmp.name, 'y.bin')
        for path in (a, b):
            with open(path, 'wb') as f:
                f.write(b'mismo contenido')
        self.assertEqual(hashFile(a), hashFile(b))


if __name__ == '__main__':
    unittest.main()
