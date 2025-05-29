from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load the trained model
model = load_model("palm_disease_classifier.keras")

# Create a validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    'val',                
    target_size=(224, 224),       
    batch_size=32,
    class_mode='categorical',      
    shuffle=False                  
)

# Get predictions for all validation data
preds = model.predict(val_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)

# Get true class labels
y_true = val_generator.classes

# Print classification results
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
