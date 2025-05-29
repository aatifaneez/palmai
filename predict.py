import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from collections import Counter
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load models
palm_detector = load_model("palm_leaf_detector.keras")
disease_model = load_model("palm_disease_classifier.keras")

labels = {
    0:"black_scorch", 
    1:"fusarium_wilt", 
    2:"healthy",
    3:"magnesium_deficiency", 
    4:"manganese_deficiency", 
    5:"parlatoria_blanchardi", 
    6:"potassium_deficiency", 
    7:"rachis_blight"
}

def preprocess_image(image):
    image = Image.open(image).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 224, 224, 3)
    return image_array
        

def check_palm_leaf(image_data):
    """Check if image contains a palm leaf using binary classifier - BINARY PREDICTION IS INVERTED"""

    prediction = palm_detector.predict(image_data, verbose=0)[0][0]
    is_palm = prediction < 0.5
    
    logger.info(f"Palm detection: {is_palm}, prediction: {prediction}")
    return is_palm


def get_all_predictions(image_input):
    """
    Main prediction function - returns same format as old_get_all_predictions
    
    Args:
        image_input: Image input (path, file object, or PIL Image)
        use_voting: Whether to use voting system with multiple crops
        num_crops: Number of crops for voting system
        
    Returns:
        list: List of prediction dictionaries with 'disease', 'confidence', and 'class_index'
               Returns empty list if not a palm leaf or on error
    """
    try:
        
        # Preprocess image
        image_data = preprocess_image(image_input)
        
        '''        
        # Stage 1: Palm detection
        is_palm = check_palm_leaf(image_data)
        
        if not is_palm:
            logger.info(f'Image does not appear to be a palm leaf')
            return []
        '''
        predictions = disease_model.predict(image_data)
        
        # Create list of all predictions with their labels and confidence scores
        all_predictions = []
        for class_index, confidence in enumerate(predictions[0]):
            disease_name = labels[class_index]
            all_predictions.append({
                'disease': disease_name,
                'confidence': float(confidence),
                'class_index': class_index
            })
        

        predicted_class_index = np.argmax(predictions)
        predicted_class = labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        print(f"prediction:{predicted_class}, cofidences:{predictions}")

        # Sort by confidence in descending order
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return all_predictions
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return []



'''
def get_prediction_summary(image_data, threshold=0.1):
    """Get a summary of predictions above a certain threshold"""
    all_preds = get_all_predictions(image_data)
    
    # Filter predictions above threshold
    significant_predictions = [
        pred for pred in all_preds 
        if pred['confidence'] >= threshold
    ]
    
    # Get top prediction
    top_prediction = all_preds[0]
    
    return {
        'top_prediction': {
            'disease': top_prediction['disease'],
            'confidence': top_prediction['confidence']
        },
        'all_predictions': all_preds,
        'significant_predictions': significant_predictions,
        'prediction_count': len(significant_predictions)
    }'''