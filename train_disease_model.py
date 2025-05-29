from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Image settings
IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/diseases/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    'dataset/diseases/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Found {train_generator.samples} training images")
print(f"Found {val_generator.samples} validation images")
print(f"Number of classes: {train_generator.num_classes}")
print(f"Class indices: {train_generator.class_indices}")

# Callbacks
callbacks = [
    EarlyStopping(
        monitor="val_accuracy", 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        "best_palm_model.keras", 
        monitor="val_accuracy", 
        save_best_only=True, 
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Create class weights
labels = train_generator.classes 
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")

# Load MobileNetV2
base_model = MobileNetV2(
    weights="imagenet", 
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Create Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x) 
x = Dropout(0.3)(x)  
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
predictions = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

print(f"Total parameters: {model.count_params():,}")

# Phase 1
print("\n=== Phase 1: Training classifier only ===")
base_model.trainable = False
trainable_params = sum([np.prod(layer.get_weights()[0].shape) for layer in model.layers if layer.trainable and layer.get_weights()])
print(f"Trainable parameters: {trainable_params:,}")

model.compile(
    optimizer=Adam(learning_rate=1e-3),  # Higher initial LR
    loss="categorical_crossentropy", 
    metrics=["accuracy"]
)

history1 = model.fit(
    train_generator, 
    validation_data=val_generator, 
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Phase 2
print("\n=== Phase 2: Fine-tuning ===")
base_model.trainable = True


for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable_params = sum([np.prod(layer.get_weights()[0].shape) for layer in model.layers if layer.trainable and layer.get_weights()])
print(f"Trainable parameters: {trainable_params:,}")

model.compile(
    optimizer=Adam(learning_rate=1e-5),  
    loss="categorical_crossentropy", 
    metrics=["accuracy"]
)

history2 = model.fit(
    train_generator, 
    validation_data=val_generator, 
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)


# Save final model
model.save("final_palm_disease_model.keras")
print("Training completed and model saved!")

# Plot training history
def plot_training_history(history1, history2):
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    phase1_end = len(history1.history['accuracy'])
    
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1 End')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1 End')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot the training history
plot_training_history(history1, history2)

# Evaluate on validation set
print("\n=== Final Evaluation ===")
val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
print(f"Final Validation Accuracy: {val_accuracy:.4f}")
print(f"Final Validation Loss: {val_loss:.4f}")