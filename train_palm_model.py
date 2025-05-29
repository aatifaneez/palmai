from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

# Binary classification settings
IMG_SIZE = 224
BATCH_SIZE = 32

# Data generators - minimal augmentation for binary task
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    'dataset/binary/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    'dataset/binary/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss", 
        patience=7, 
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "palm_leaf_detector.keras", 
        monitor="val_accuracy", 
        save_best_only=True, 
        verbose=1
    )
]

# Build binary model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation="relu")(x)  # Smaller layer for binary task
x = Dropout(0.2)(x)
predictions = Dense(1, activation="sigmoid")(x)  # Single output with sigmoid

model = Model(inputs=base_model.input, outputs=predictions)

# Phase 1
base_model.trainable = False
model.compile(
    optimizer=Adam(learning_rate=1e-3), 
    loss="binary_crossentropy", metrics=["accuracy"]
)
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks
)

# Phase 2
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Freeze more layers for binary task
    layer.trainable = False
model.compile(
    optimizer=Adam(learning_rate=1e-5), 
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)


model.save("palm_leaf_detector_final.keras")
