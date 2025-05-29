import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
palm_detector = load_model("palm_leaf_detector.keras")
disease_model = load_model("palm_disease_classifier.keras")

# Disease labels
labels = {
    0: "black_scorch", 
    1: "fusarium_wilt", 
    2: "healthy",
    3: "magnesium_deficiency", 
    4: "manganese_deficiency", 
    5: "parlatoria_blanchardi", 
    6: "potassium_deficiency", 
    7: "rachis_blight"
}


def preprocess_image(image_input):
    """Convert image input to preprocessed numpy array"""
    try:
        # Handle different input types
        if isinstance(image_input, str):
            # File path
            image = Image.open(image_input).convert("RGB")
        elif hasattr(image_input, 'read'):
            # File-like object from Flask
            image_input.seek(0)  # Ensure we're at the beginning
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            # PIL Image
            image = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input type")
        
        # Resize and normalize
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image).astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array, image
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

def check_palm_leaf(image_data):
    """Check if image contains a palm leaf using binary classifier - BINARY PREDICTION IS INVERTED"""

    prediction = palm_detector.predict(image_data, verbose=0)[0][0]
    is_palm = prediction < 0.5
    confidence = 1 - prediction if is_palm else prediction
    
    logger.info(f"Palm detection: {is_palm} (confidence: {confidence:.3f})")
    return is_palm
    
        

def get_all_predictions_from_average(avg_predictions):
    """Convert average predictions to sorted list"""
    try:
        all_predictions = []
        for class_index, confidence in enumerate(avg_predictions):
            disease_name = labels[class_index]
            all_predictions.append({
                'disease': disease_name,
                'confidence': float(confidence),
                'class_index': class_index
            })
        
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return all_predictions
        
    except Exception as e:
        logger.error(f"Failed to process predictions: {e}")
        return []

def predict(image_input, use_voting=True, num_crops=4, palm_threshold=0.7):
    """
    Main prediction function
    
    Args:
        image_input: Image input (path, file object, or PIL Image)
        use_voting: Whether to use voting system with multiple crops
        num_crops: Number of crops for voting system
        palm_threshold: Minimum confidence for palm detection
        
    Returns:
        dict: Complete prediction results
    """
    try:
        
        # Preprocess image
        image_data, pil_image = preprocess_image(image_input)
        
        # Stage 1: Palm detection
        is_palm, palm_confidence = check_palm_leaf(image_data)
        
        if not is_palm or palm_confidence < palm_threshold:
            return {
                'success': False,
                'prediction': 'not_palm_leaf',
                'confidence': palm_confidence,
                'palm_confidence': palm_confidence,
                'message': 'Image does not appear to be a palm leaf',
                'all_predictions': [],
                'metadata': {
                    'stage': 'palm_detection',
                    'use_voting': use_voting
                }
            }
        
        # Stage 2: Disease classification
        if use_voting:
            # Use voting system
            avg_confidence, winner, individual_preds, individual_confs, vote_counts, avg_all_preds = predict_with_voting(
                pil_image, num_crops
            )
            
            all_predictions = get_all_predictions_from_average(avg_all_preds)
            
            metadata = {
                'stage': 'voting_prediction',
                'individual_predictions': individual_preds,
                'individual_confidences': individual_confs,
                'vote_counts': vote_counts,
                'num_crops': num_crops
            }
            
        else:
            # Single prediction
            avg_confidence, winner, single_all_preds = predict_single_image(image_data)
            all_predictions = get_all_predictions_from_average(single_all_preds)
            
            metadata = {
                'stage': 'single_prediction'
            }
        
        return {
            'success': True,
            'prediction': winner,
            'confidence': avg_confidence,
            'palm_confidence': palm_confidence,
            'all_predictions': all_predictions,
            'metadata': metadata
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            'success': False,
            'prediction': 'error',
            'confidence': 0.0,
            'palm_confidence': 0.0,
            'message': f'Prediction error: {str(e)}',
            'all_predictions': [],
            'metadata': {'error': str(e)}
        }
