"""
models/train_blink_model.py
============================
Trains a binary CNN classifier on MRL Eye Dataset.
Classifies eye images as: open (0) or closed (1=blink).

Usage:
    python models/train_blink_model.py \
        --data_dir /path/to/mrl_eye \
        --epochs 20 \
        --output models/blink_model.h5

MRL Dataset folder structure expected:
    data_dir/
        data/
            train/
                open/   (or folder names containing 'open')
                closed/ (or folder names containing 'close')
            test/
                open/
                closed/
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE   = 64
BATCH_SIZE = 32

def build_blink_cnn():
    """
    Lightweight binary CNN for blink/eye-open detection.
    Input:  64×64 grayscale eye-region image
    Output: [open, closed] probabilities
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.2),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(2, activation='softmax'),  # [open, closed]
    ])
    return model


def compute_ear_from_dataset(data_dir, sample_n=1000):
    """
    Compute EAR statistics from dataset for threshold calibration.
    Returns mean EAR for open/closed eyes.
    """
    import cv2, random
    ear_open, ear_closed = [], []

    for split in ['train', 'test']:
        for state in ['open', 'closed']:
            state_dir = None
            for root, dirs, _ in os.walk(os.path.join(data_dir, split if os.path.exists(
                    os.path.join(data_dir, split)) else '')):
                for d in dirs:
                    if state in d.lower():
                        state_dir = os.path.join(root, d)
                        break

            if not state_dir or not os.path.exists(state_dir): continue
            imgs = [f for f in os.listdir(state_dir) if f.lower().endswith(('.jpg','.png','.bmp'))]
            sample = random.sample(imgs, min(sample_n//4, len(imgs)))

            for img_name in sample:
                img = cv2.imread(os.path.join(state_dir, img_name), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                h, w = img.shape
                # Simplified EAR proxy: height/width ratio of eye region
                ear = h / (w + 1e-6)
                if state == 'open':
                    ear_open.append(ear)
                else:
                    ear_closed.append(ear)

    print(f'EAR open:   mean={np.mean(ear_open):.3f} ± {np.std(ear_open):.3f}' if ear_open else 'EAR open: N/A')
    print(f'EAR closed: mean={np.mean(ear_closed):.3f} ± {np.std(ear_closed):.3f}' if ear_closed else 'EAR closed: N/A')


def train(data_dir, epochs=20, output_path='models/blink_model.h5'):
    print(f'\n{"="*55}')
    print('  Blink Detection Model Training — MRL Eye Dataset')
    print(f'{"="*55}\n')

    model = build_blink_cnn()
    model.summary()
    model.compile(
        optimizer=optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # ── Find data directories ─────────────────────────────────────
    # MRL dataset may have nested structure
    train_path = None
    for candidate in [
        os.path.join(data_dir, 'train'),
        os.path.join(data_dir, 'data', 'train'),
        data_dir,
    ]:
        if os.path.exists(candidate):
            train_path = candidate; break

    if not train_path:
        print(f'❌ Could not find train directory in {data_dir}')
        print('Expected structure: data_dir/train/open/ and data_dir/train/closed/')
        return

    val_path = train_path.replace('train', 'test')
    if not os.path.exists(val_path):
        val_path = None

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        zoom_range=0.1,
        validation_split=0.2 if val_path is None else 0.0,
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training' if val_path is None else None,
        shuffle=True,
    )

    val_gen = (val_datagen if val_path else train_datagen).flow_from_directory(
        val_path or train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset=None if val_path else 'validation',
        shuffle=False,
    )

    print(f'\nClass mapping: {train_gen.class_indices}')
    print(f'Train: {train_gen.samples} | Val: {val_gen.samples}\n')

    cb = [
        callbacks.ModelCheckpoint(output_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ]

    history = model.fit(
        train_gen, epochs=epochs, validation_data=val_gen, callbacks=cb, verbose=1
    )

    model.save(output_path)
    best_acc = max(history.history.get('val_accuracy', [0]))
    print(f'\n✅ Blink model saved: {output_path}')
    print(f'✅ Best val accuracy: {best_acc*100:.2f}%')

    # Compute EAR stats
    print('\nComputing EAR statistics...')
    compute_ear_from_dataset(data_dir)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--epochs',   type=int, default=20)
    parser.add_argument('--output',   default='models/blink_model.h5')
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    train(args.data_dir, args.epochs, args.output)
