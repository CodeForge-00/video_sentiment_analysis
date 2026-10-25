"""
CNN para reconocer emociones en rostros (48x48, escala de grises, 6 clases).

  python model.py train    [--data sentiment_dataset_clean] [--epochs 40] [--out EmotionDetectionModel.keras]
  python model.py evaluate MODELO [--data sentiment_dataset_clean]

Antes de entrenar, `python data.py` crea la copia limpia del conjunto de datos (sin duplicados ni fugas).
Las imagenes entran al modelo con valores en [0, 1].
"""
import argparse
import json
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import keras
from keras import layers

from data import CLASSES

IMG_SIZE = 48
DEFAULT_DATA = 'sentiment_dataset_clean'


def buildModel(num_classes=len(CLASSES), augment=True):
    """Ocho convoluciones con normalizacion por lotes (Conv -> ELU -> BN) y dos capas densas."""
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 1))
    x = inputs
    if augment:  # solo se aplica durante el entrenamiento
        x = layers.RandomFlip('horizontal')(x)
        x = layers.RandomRotation(0.04)(x)              # ~ +-14 grados
        x = layers.RandomTranslation(0.1, 0.1)(x)
        x = layers.RandomZoom(0.1)(x)
    for filters in (32, 64, 128, 256):
        for _ in range(2):
            x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
            x = layers.Activation('elu')(x)
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    for _ in range(2):
        x = layers.Dense(64, kernel_initializer='he_normal')(x)
        x = layers.Activation('elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, kernel_initializer='he_normal', activation='softmax')(x)
    return keras.Model(inputs, outputs)


def loadDataset(directory, split, batch_size, shuffle, seed=0):
    """Dataset de tf.data con imagenes en [0, 1] y etiquetas one-hot, y el numero de imagenes de cada clase."""
    path = os.path.join(directory, split)
    dataset = keras.utils.image_dataset_from_directory(
        path, labels='inferred', label_mode='categorical', class_names=CLASSES, color_mode='grayscale',
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size, shuffle=shuffle, seed=seed)
    counts = [len(os.listdir(os.path.join(path, cls))) for cls in CLASSES]
    return dataset.map(lambda x, y: (x / 255.0, y)).prefetch(2), counts


def classWeights(counts):
    """Peso inverso a la frecuencia de cada clase: n_total / (n_clases * n_clase)."""
    counts = np.asarray(counts, dtype=float)
    return {k: float(counts.sum() / (len(counts) * c)) for k, c in enumerate(counts)}


def metricsFrom(y_true, y_pred, n_classes=len(CLASSES)):
    """Exactitud, recall y precision por clase y matriz de confusion (filas = real, columnas = predicha)."""
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    support, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    recall = np.divide(np.diag(confusion), support, out=np.zeros(n_classes), where=support > 0)
    precision = np.divide(np.diag(confusion), predicted, out=np.zeros(n_classes), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n_classes), where=(precision + recall) > 0)
    return {'accuracy': float(np.trace(confusion) / confusion.sum()),
            'majority_class_accuracy': float(support.max() / confusion.sum()),
            'macro_f1': float(f1.mean()),
            'recall': {c: float(r) for c, r in zip(CLASSES, recall)},
            'precision': {c: float(p) for c, p in zip(CLASSES, precision)},
            'confusion_matrix': confusion.tolist()}


def evaluate(model, directory=DEFAULT_DATA, split='validation', batch_size=128):
    dataset, _ = loadDataset(directory, split, batch_size, shuffle=False)
    y_true = np.concatenate([y.numpy().argmax(1) for _, y in dataset])
    y_pred = model.predict(dataset, verbose=0).argmax(1)
    return metricsFrom(y_true, y_pred)


def printMetrics(m):
    print(f"Exactitud: {m['accuracy']:.4f} (clase mayoritaria: {m['majority_class_accuracy']:.4f}), F1 macro: {m['macro_f1']:.4f}")
    print(f"{'clase':<10}{'recall':>8}{'precision':>11}")
    for c in CLASSES:
        print(f"{c:<10}{m['recall'][c]:>8.3f}{m['precision'][c]:>11.3f}")
    print('Matriz de confusion (filas = real, columnas = predicha):', ' '.join(CLASSES))
    for row in m['confusion_matrix']:
        print('  ', ' '.join(f'{v:>5}' for v in row))


def train(args):
    keras.utils.set_random_seed(args.seed)
    train_ds, counts = loadDataset(args.data, 'train', args.batch_size, shuffle=True, seed=args.seed)
    validation_ds, _ = loadDataset(args.data, 'validation', args.batch_size, shuffle=False)
    model = buildModel()
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    callbacks = [
        keras.callbacks.ModelCheckpoint(args.out, monitor='val_loss', mode='min', save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_delta=0.0001, verbose=1),
    ]
    history = model.fit(train_ds, epochs=args.epochs, validation_data=validation_ds, callbacks=callbacks,
                        class_weight=None if args.no_class_weights else classWeights(counts))
    metrics = evaluate(model, args.data)
    printMetrics(metrics)
    with open(args.results, 'w') as file:
        json.dump({'epochs_run': len(history.history['loss']), 'seed': args.seed, 'data': args.data,
                   'class_weights': not args.no_class_weights, 'validation': metrics,
                   'history': {k: [float(v) for v in vs] for k, vs in history.history.items()}}, file, indent=2)
    print(f"Modelo guardado en {args.out} y resultados en {args.results}")


def main():
    parser = argparse.ArgumentParser(description='Entrena o evalua la CNN de emociones')
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', default=DEFAULT_DATA)
    t.add_argument('--epochs', type=int, default=40)
    t.add_argument('--batch-size', type=int, default=64)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--no-class-weights', action='store_true', help='no compensar el desequilibrio de clases')
    t.add_argument('--out', default='EmotionDetectionModel.keras')
    t.add_argument('--results', default='results.json')
    e = sub.add_parser('evaluate')
    e.add_argument('model')
    e.add_argument('--data', default=DEFAULT_DATA)
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    else:
        printMetrics(evaluate(keras.models.load_model(args.model, compile=False), args.data))


if __name__ == '__main__':
    main()
