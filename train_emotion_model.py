"""
models/train_emotion_model.py
==============================
Trains a CNN on FER2013 dataset for facial emotion detection.

Architecture: Mini-VGG style (lightweight, runs on CPU)
Input:  48×48 grayscale face images
Output: 7 emotion classes

Usage:
    python models/train_emotion_model.py \
        --data_dir /path/to/fer2013 \
        --epochs 30 \
        --output models/emotion_model.h5

FER2013 folder structure expected:
    data_dir/
        train/
            angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
        test/
            angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Emotion labels ────────────────────────────────────────────────
EMOTIONS   = ['angry','disgust','fear','happy','neutral','sad','surprise']
IMG_SIZE   = 48
BATCH_SIZE = 64

def build_emotion_cnn(num_classes=7):
    """
    Mini-VGG CNN for emotion detection.
    Designed for 48×48 grayscale input (FER2013).

    Architecture:
    Block 1: Conv(32) → Conv(32) → Pool → Dropout
    Block 2: Conv(64) → Conv(64) → Pool → Dropout
    Block 3: Conv(128) → Conv(128) → Pool → Dropout
    Head:    Flatten → Dense(256) → Dropout → Softmax(7)
    """
    model = models.Sequential([
        # Input
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # ── Block 1 ────────────────────────────────────────────
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # ── Block 2 ────────────────────────────────────────────
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # ── Block 3 ────────────────────────────────────────────
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # ── Classification head ────────────────────────────────
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def create_data_generators(data_dir):
    """Create augmented data generators for FER2013."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        shear_range=0.1,
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=EMOTIONS,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=EMOTIONS,
        shuffle=False,
    )
    return train_gen, val_gen


def train(data_dir, epochs=30, output_path='models/emotion_model.h5'):
    print(f'\n{"="*55}')
    print('  Emotion Detection Model Training — FER2013')
    print(f'{"="*55}\n')

    # ── Build model ───────────────────────────────────────────────
    model = build_emotion_cnn(num_classes=len(EMOTIONS))
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # ── Data generators ───────────────────────────────────────────
    train_gen, val_gen = create_data_generators(data_dir)
    print(f'\nTrain samples: {train_gen.samples}')
    print(f'Val   samples: {val_gen.samples}')
    print(f'Classes: {train_gen.class_indices}\n')

    # ── Callbacks ─────────────────────────────────────────────────
    cb = [
        callbacks.ModelCheckpoint(
            output_path, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        callbacks.CSVLogger('models/training_log.csv'),
    ]

    # ── Train ─────────────────────────────────────────────────────
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=cb,
        verbose=1,
    )

    # ── Save final model ──────────────────────────────────────────
    model.save(output_path)
    print(f'\n✅ Model saved to: {output_path}')

    # Best val accuracy
    best_acc = max(history.history['val_accuracy'])
    print(f'✅ Best validation accuracy: {best_acc*100:.2f}%')

    # ── Plot training curves ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0a0a0f')
    for ax in axes:
        ax.set_facecolor('#16161f')
        ax.tick_params(colors='#8b8b9e')
        ax.spines[:].set_color('#2a2a3a')

    axes[0].plot(history.history['accuracy'],     color='#7c3aed', label='Train')
    axes[0].plot(history.history['val_accuracy'], color='#06b6d4', label='Val')
    axes[0].set_title('Accuracy', color='#f1f0ff')
    axes[0].legend()

    axes[1].plot(history.history['loss'],     color='#7c3aed', label='Train')
    axes[1].plot(history.history['val_loss'], color='#06b6d4', label='Val')
    axes[1].set_title('Loss', color='#f1f0ff')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=150, bbox_inches='tight')
    print('✅ Training curves saved to: models/training_curves.png\n')

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FER2013 emotion model')
    parser.add_argument('--data_dir', required=True, help='Path to FER2013 dataset root')
    parser.add_argument('--epochs',   type=int, default=30)
    parser.add_argument('--output',   default='models/emotion_model.h5')
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    train(args.data_dir, args.epochs, args.output)
