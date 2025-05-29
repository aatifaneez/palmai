from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

model = load_model("palm_leaf_detector.keras")


def preprocess_image(image):
    image = Image.open(image).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 224, 224, 3)
    return image_array


def predict(image_data):
    predictions = model.predict(image_data)
    predicted_class_index = np.argmax(predictions)

    confidence = predictions[0][predicted_class_index]
    
    print(f"{predictions}")
    return confidence, predicted_class_index


if __name__ == "__main__":
    image_data = preprocess_image("dataset/binary/train/unrelated/32469_jpg.rf.932e846f4301e65383a16365ecf7b78a.jpg")
    predict(image_data)