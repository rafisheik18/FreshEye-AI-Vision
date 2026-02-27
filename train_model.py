import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Image settings
IMG_SIZE = 224
BATCH_SIZE = 32

# Dataset paths
train_path = "dataset/train"
test_path = "dataset/test"

# ✅ Strong camera-style augmentation (UNCHANGED)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.3,
    horizontal_flip=True,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    brightness_range=[0.6, 1.4]
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# ✅ Freeze MOST layers (UNCHANGED LOGIC)
for layer in base_model.layers[:-4]:
    layer.trainable = False

# ✅ Fine-tune last layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

# ✅ Model (only safe upgrade → BatchNorm)
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),      # 🔥 NEW → Improves stability
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# ✅ Lower LR (UNCHANGED)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Callbacks (MAJOR accuracy upgrade)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.h5",              # 🔥 Saves best model automatically
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',           # 🔥 Smart LR adjustment
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-6
)

# Train model
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=15,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Save final model
model.save("healthy_vs_rotten.h5")

print("MODEL SAVED SUCCESSFULLY ✅")