"""
Limpieza del conjunto de datos de emociones.

El conjunto original tiene problemas que invalidan la evaluacion:

  * Imagenes repetidas dentro de entrenamiento.
  * Imagenes de validacion que tambien estan en entrenamiento (fuga de datos).
  * El mismo contenido con etiquetas distintas. En particular, la carpeta validation/Neutral esta formada casi por
    completo por copias de imagenes de entrenamiento de OTRAS emociones (Happy, Sad, Fear...), asi que no sirve para
    evaluar esa clase.

Este modulo construye una copia limpia sin tocar la original:

  * contenido repetido dentro de entrenamiento con etiquetas distintas -> se descarta (ambiguo)
  * contenido de entrenamiento repetido con la misma etiqueta          -> se conserva una copia
  * contenido de validacion que ya esta en entrenamiento               -> se descarta de validacion (se confia en
                                                                          la etiqueta de entrenamiento)
  * contenido solo de validacion, repetido con etiquetas distintas     -> se descarta; con la misma etiqueta, una copia
  * Con `val_fraction`, todas las imagenes unicas que quedan se reparten de nuevo en entrenamiento y validacion, de
    forma estratificada por clase y con semilla fija (la validacion original ya no es fiable, ver arriba).

Uso:  python data.py [--src sentiment_dataset] [--dst sentiment_dataset_clean] [--val-fraction 0.1] [--seed 0]
"""
import argparse
import collections
import hashlib
import os
import random
import shutil

CLASSES = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
SPLITS = ['train', 'validation']
MARKER = '.generated'  # marca las carpetas creadas por este modulo, las unicas que se pueden sobrescribir


def hashFile(path):
    with open(path, 'rb') as file:
        return hashlib.md5(file.read()).hexdigest()


def scan(root):
    """Devuelve {contenido (md5): [(particion, clase, ruta), ...]} para todas las imagenes de root."""
    by_content = collections.defaultdict(list)
    for split in SPLITS:
        for cls in sorted(os.listdir(os.path.join(root, split))):
            folder = os.path.join(root, split, cls)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                path = os.path.join(folder, name)
                by_content[hashFile(path)].append((split, cls, path))
    return by_content


def countImages(root):
    """Imagenes por particion y clase."""
    return {split: {cls: len(os.listdir(os.path.join(root, split, cls)))
                    for cls in sorted(os.listdir(os.path.join(root, split)))
                    if os.path.isdir(os.path.join(root, split, cls))} for split in SPLITS}


def buildClean(src, dst, val_fraction=0.1, seed=0, overwrite=False):
    """
    Crea en dst la copia limpia de src y devuelve un diccionario con lo que se descarto y por que.
    val_fraction: fraccion de cada clase que va a validacion al repartir de nuevo las imagenes unicas; None conserva las
    particiones originales (ya sin duplicados ni fugas).
    """
    if os.path.exists(dst):
        if not (overwrite and os.path.exists(os.path.join(dst, MARKER))):
            raise FileExistsError(f"{dst} ya existe; use --overwrite (solo se sobrescriben carpetas generadas por este modulo)")
        shutil.rmtree(dst)

    removed = collections.Counter()
    kept = []  # (particion, clase, ruta), una por contenido distinto
    for content, entries in sorted(scan(src).items()):
        train_entries = [e for e in entries if e[0] == 'train']
        validation_entries = [e for e in entries if e[0] == 'validation']
        if train_entries:
            if len({cls for _, cls, _ in train_entries}) > 1:
                removed['ambiguous_label'] += len(entries)
                continue
            kept.append(train_entries[0])
            removed['duplicate_in_train'] += len(train_entries) - 1
            removed['validation_copy_of_train'] += len(validation_entries)
        else:
            if len({cls for _, cls, _ in validation_entries}) > 1:
                removed['ambiguous_label'] += len(validation_entries)
                continue
            kept.append(validation_entries[0])
            removed['duplicate_in_validation'] += len(validation_entries) - 1

    if val_fraction is not None:
        rng = random.Random(seed)
        by_class = collections.defaultdict(list)
        for _, cls, path in kept:
            by_class[cls].append(path)
        kept = []
        for cls in sorted(by_class):
            paths = sorted(by_class[cls])
            rng.shuffle(paths)
            n_validation = round(len(paths) * val_fraction)
            kept += [('validation', cls, p) for p in paths[:n_validation]]
            kept += [('train', cls, p) for p in paths[n_validation:]]

    for split, cls, path in kept:
        target = os.path.join(dst, split, cls)
        os.makedirs(target, exist_ok=True)
        shutil.copy2(path, os.path.join(target, os.path.basename(path)))
    with open(os.path.join(dst, MARKER), 'w') as file:
        file.write('Generado por data.py; se puede borrar y regenerar.\n')
    return dict(removed)


def main():
    parser = argparse.ArgumentParser(description='Crea una copia limpia del conjunto de datos de emociones')
    parser.add_argument('--src', default='sentiment_dataset')
    parser.add_argument('--dst', default='sentiment_dataset_clean')
    parser.add_argument('--val-fraction', type=float, default=0.1,
                        help='fraccion de cada clase para validacion al repartir de nuevo (0 conserva las particiones originales)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true', help='regenerar dst si ya fue creado por este script')
    args = parser.parse_args()

    before = countImages(args.src)
    removed = buildClean(args.src, args.dst, args.val_fraction or None, args.seed, args.overwrite)
    after = countImages(args.dst)
    total = lambda counts, split: sum(counts[split].values())
    print(f"Original:  entrenamiento {total(before, 'train')}, validacion {total(before, 'validation')}")
    print(f"Limpio:    entrenamiento {total(after, 'train')}, validacion {total(after, 'validation')}")
    names = {'ambiguous_label': 'contenido repetido con etiquetas distintas',
             'duplicate_in_train': 'duplicados dentro de entrenamiento',
             'validation_copy_of_train': 'imagenes de validacion que ya estaban en entrenamiento',
             'duplicate_in_validation': 'duplicados dentro de validacion'}
    for key, label in names.items():
        print(f"  quitadas por {label}: {removed.get(key, 0)}")
    print('Imagenes por clase (entrenamiento / validacion):')
    for cls in CLASSES:
        print(f"  {cls:<9} {after['train'].get(cls, 0):>5} / {after['validation'].get(cls, 0):>4}")


if __name__ == '__main__':
    main()
