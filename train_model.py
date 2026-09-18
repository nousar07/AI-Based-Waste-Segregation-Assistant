import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# 1. Project Settings
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "dataset-resized"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


# ==========================================
# 2. Load Training Dataset
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ==========================================
# 3. Load Validation Dataset
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ==========================================
# 4. Waste Categories
# ==========================================

class_names = train_dataset.class_names

print("\nWaste Categories:")
print(class_names)


# ==========================================
# 5. Improve Dataset Performance
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# 6. Data Augmentation
# ==========================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# ==========================================
# 7. Load MobileNetV2
# ==========================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ==========================================
# 8. Build AI Model
# ==========================================

inputs = keras.Input(
    shape=IMG_SIZE + (3,)
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs
)


# ==========================================
# 9. Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 10. Display Model
# ==========================================

model.summary()


# ==========================================
# 11. Train Model
# ==========================================

print("\n========================================")
print("Starting AI model training...")
print("========================================\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ==========================================
# 12. Create Model Folder
# ==========================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==========================================
# 13. Save Trained Model
# ==========================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "waste_classifier.keras"
)

print("\nSaving trained model...")

model.save(MODEL_PATH)


# ==========================================
# 14. Final Message
# ==========================================

print("\n========================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("========================================")
print("Model saved successfully at:")
print(MODEL_PATH)
print("========================================")