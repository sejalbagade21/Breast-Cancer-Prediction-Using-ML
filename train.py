import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


# CONFIG

IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS_INITIAL = 5
EPOCHS_FINE = 3

ORIGINAL_DATASET = "2750"
NEW_DATASET = "land_3class"


# STEP 1: CREATE 3-CLASS DATASET

def create_3class_dataset():
    if os.path.exists(NEW_DATASET):
        print("3-class dataset already exists.")
        return

    os.makedirs(f"{NEW_DATASET}/Agriculture", exist_ok=True)
    os.makedirs(f"{NEW_DATASET}/Forest", exist_ok=True)
    os.makedirs(f"{NEW_DATASET}/Urban", exist_ok=True)

    agri_classes = ["AnnualCrop", "PermanentCrop", "Pasture", "HerbaceousVegetation"]
    urban_classes = ["Residential", "Industrial", "Highway"]

    # Copy Agriculture
    for cls in agri_classes:
        for file in os.listdir(f"{ORIGINAL_DATASET}/{cls}"):
            shutil.copy(
                f"{ORIGINAL_DATASET}/{cls}/{file}",
                f"{NEW_DATASET}/Agriculture"
            )

    # Copy Forest
    for file in os.listdir(f"{ORIGINAL_DATASET}/Forest"):
        shutil.copy(
            f"{ORIGINAL_DATASET}/Forest/{file}",
            f"{NEW_DATASET}/Forest"
        )

    # Copy Urban
    for cls in urban_classes:
        for file in os.listdir(f"{ORIGINAL_DATASET}/{cls}"):
            shutil.copy(
                f"{ORIGINAL_DATASET}/{cls}/{file}",
                f"{NEW_DATASET}/Urban"
            )

    print("3-Class Dataset Created Successfully!")



# STEP 2: BALANCE DATASET

def limit_images(folder, max_images=3000):
    for cls in os.listdir(folder):
        path = os.path.join(folder, cls)
        images = os.listdir(path)

        if len(images) > max_images:
            remove_images = random.sample(images, len(images) - max_images)
            for img in remove_images:
                os.remove(os.path.join(path, img))

    print("Dataset Balanced!")



# STEP 3: DATA GENERATORS

def create_generators():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_data = datagen.flow_from_directory(
        NEW_DATASET,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_data = datagen.flow_from_directory(
        NEW_DATASET,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    return train_data, val_data



# STEP 4: BUILD MODEL

def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model



# STEP 5: TRAIN + FINE-TUNE

def train_model():
    create_3class_dataset()
    limit_images(NEW_DATASET)

    train_data, val_data = create_generators()
    model, base_model = build_model()

    print("Initial Training...")
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS_INITIAL)

    print("Fine-Tuning...")
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_data, validation_data=val_data, epochs=EPOCHS_FINE)

    model.save("land_use_model.h5")
    print("Model saved successfully!")


# ---------------------------
if __name__ == "__main__":
    train_model()
