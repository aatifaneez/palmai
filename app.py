import logging
import json
import time
from datetime import datetime
from collections import defaultdict, deque
from flask import Flask, render_template, redirect, url_for, request, jsonify
from PIL import Image

try:
    import predict as predict
    PREDICT_AVAILABLE = True
except Exception as e:
    logging.warning(f"Predict module not fully available: {e}")
    PREDICT_AVAILABLE = False


app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# In-memory storage for analytics and feedback
analytics_data = {
    'total_analyses': 0,
    'disease_counts': defaultdict(int),
    'recent_analyses': deque(maxlen=50),  # Keep last 50 analyses
    'daily_stats': defaultdict(lambda: {
        'date': None,
        'count': 0,
        'diseases': defaultdict(int),
        'avg_confidence': 0,
        'confidence_sum': 0
    }),
    'feedback_data': [],
    'processing_times': deque(maxlen=100),  # Keep last 100 processing times
    'start_time': datetime.now()
}

with open("static/disease_data.json") as file:
    disease_data = json.load(file)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file):
    """Validate uploaded image file"""
    if not file or file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, "Invalid file type. Please upload JPG or PNG images only."
    
    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        return False, "File too large. Maximum size is 10MB."
    
    # Validate it's actually an image
    try:
        img = Image.open(file)
        img.verify()
        file.seek(0)  # Reset after verify
        return True, None
    except Exception as e:
        logging.error(f"Image validation failed: {e}")
        return False, "Invalid image file. Please upload a valid image."

def update_analytics(prediction, confidence, processing_time, filename):
    """Update in-memory analytics data"""
    now = datetime.now()
    today = now.date()
    
    # Update global stats
    analytics_data['total_analyses'] += 1
    analytics_data['disease_counts'][prediction] += 1
    analytics_data['processing_times'].append(processing_time)
    
    # Add to recent analyses
    analysis_record = {
        'id': f"analysis_{analytics_data['total_analyses']}",
        'disease': prediction,
        'confidence': confidence,
        'timestamp': now.isoformat(),
        'filename': filename,
        'processing_time': processing_time
    }
    analytics_data['recent_analyses'].append(analysis_record)
    
    # Update daily stats
    daily_key = today.isoformat()
    daily = analytics_data['daily_stats'][daily_key]
    daily['date'] = daily_key
    daily['count'] += 1
    daily['diseases'][prediction] += 1
    daily['confidence_sum'] += confidence
    daily['avg_confidence'] = daily['confidence_sum'] / daily['count']


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # Validate file upload
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        is_valid, error_msg = validate_image(file)
        
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        start_time = time.time()
        
        if PREDICT_AVAILABLE:
            # Use the new get_all_predictions function that includes palm detection
            all_predictions = predict.get_all_predictions(file)
            
            # Check if image is a palm leaf
            if not all_predictions:  # Empty list means not a palm leaf or error
                return jsonify({
                    'success': False,
                    'error': 'The uploaded image does not appear to be a palm leaf. Please upload an image of a palm leaf for disease analysis.',
                    'prediction': 'not_palm_leaf',
                    'confidence': 0.0
                }), 400
            
            # Get the top prediction (highest confidence)
            top_prediction = all_predictions[0]
            prediction = top_prediction['disease']
            confidence = top_prediction['confidence']
            
        else:
            # Need the actual AI model - show helpful message
            return jsonify({
                'error': 'AI model not available. Please upload trained .keras model file to the project directory and install the required dependencies (tensorflow, numpy, pillow, scikit-learn).'
            }), 400
        
        processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
        
        # Update analytics
        update_analytics(prediction, confidence, processing_time, file.filename)
        
        # Get disease information
        disease_info = disease_data.get(prediction, disease_data.get('unknown'))
        
        # Prepare response
        result = {
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'disease_info': disease_info,
            'alternatives': all_predictions[:3],  # Top 3 alternatives
            'processing_time_ms': processing_time
        }
        
        return jsonify(result)
            
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({'error': 'An error occurred during analysis. Please try again.'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/analytics')
def analytics():
    """Show analytics dashboard with in-memory data"""
    try:
        # Calculate additional metrics
        total_analyses = analytics_data['total_analyses']
        
        # Convert disease counts to list format
        disease_distribution = [
            {'disease': disease, 'count': count} 
            for disease, count in analytics_data['disease_counts'].items()
        ]
        
        # Get recent analyses (convert deque to list)
        recent_analyses = list(analytics_data['recent_analyses'])
        
        # Calculate average processing time
        processing_times = list(analytics_data['processing_times'])
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Calculate uptime
        uptime_seconds = (datetime.now() - analytics_data['start_time']).total_seconds()
        uptime_hours = uptime_seconds / 3600
        
        # Get daily stats
        daily_stats = [
            {
                'date': stats['date'],
                'count': stats['count'],
                'avg_confidence': round(stats['avg_confidence'], 2),
                'diseases': dict(stats['diseases'])
            }
            for stats in analytics_data['daily_stats'].values()
            if stats['date'] is not None
        ]
        
        response_data = {
            'total_analyses': total_analyses,
            'disease_distribution': disease_distribution,
            'recent_analyses': recent_analyses,
            'avg_processing_time_ms': round(avg_processing_time, 2),
            'uptime_hours': round(uptime_hours, 2),
            'daily_stats': daily_stats,
            'feedback_count': len(analytics_data['feedback_data']),
            'unique_diseases_detected': len(analytics_data['disease_counts']),
            'system_info': {
                'predict_available': PREDICT_AVAILABLE,
                'start_time': analytics_data['start_time'].isoformat(),
                'memory_usage': {
                    'recent_analyses': len(analytics_data['recent_analyses']),
                    'processing_times': len(analytics_data['processing_times']),
                    'daily_stats': len(analytics_data['daily_stats']),
                    'feedback_entries': len(analytics_data['feedback_data'])
                }
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Analytics error: {e}")
        return jsonify({'error': 'Failed to fetch analytics'}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback on analysis results - stored in memory"""
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')
        is_correct = data.get('is_correct')
        actual_disease = data.get('actual_disease')
        feedback_text = data.get('feedback_text')
        rating = data.get('rating')
        
        # Create feedback record
        feedback_record = {
            'id': f"feedback_{len(analytics_data['feedback_data']) + 1}",
            'analysis_id': analysis_id,
            'is_correct': is_correct,
            'actual_disease': actual_disease,
            'feedback_text': feedback_text,
            'rating': rating,
            'timestamp': datetime.now().isoformat(),
            'user_ip': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        }
        
        # Store feedback in memory
        analytics_data['feedback_data'].append(feedback_record)
        
        # Keep only last 1000 feedback entries to prevent memory issues
        if len(analytics_data['feedback_data']) > 1000:
            analytics_data['feedback_data'] = analytics_data['feedback_data'][-1000:]
        
        return jsonify({'success': True, 'message': 'Feedback submitted successfully'})
        
    except Exception as e:
        logging.error(f"Feedback submission error: {e}")
        return jsonify({'error': 'Failed to submit feedback'}), 500

@app.route('/feedback')
def get_feedback():
    """Get feedback data for analytics"""
    try:
        # Calculate feedback statistics
        feedback_stats = {
            'total_feedback': len(analytics_data['feedback_data']),
            'correct_predictions': sum(1 for f in analytics_data['feedback_data'] if f.get('is_correct', False)),
            'average_rating': 0,
            'recent_feedback': analytics_data['feedback_data'][-10:] if analytics_data['feedback_data'] else []
        }
        
        # Calculate average rating
        ratings = [f['rating'] for f in analytics_data['feedback_data'] if f.get('rating') is not None]
        if ratings:
            feedback_stats['average_rating'] = sum(ratings) / len(ratings)
        
        # Calculate accuracy rate
        if feedback_stats['total_feedback'] > 0:
            feedback_stats['accuracy_rate'] = feedback_stats['correct_predictions'] / feedback_stats['total_feedback']
        else:
            feedback_stats['accuracy_rate'] = 0
        
        return jsonify(feedback_stats)
        
    except Exception as e:
        logging.error(f"Get feedback error: {e}")
        return jsonify({'error': 'Failed to fetch feedback data'}), 500

@app.route('/stats/reset', methods=['POST'])
def reset_stats():
    """Reset analytics data (useful for testing)"""
    try:
        global analytics_data
        analytics_data = {
            'total_analyses': 0,
            'disease_counts': defaultdict(int),
            'recent_analyses': deque(maxlen=50),
            'daily_stats': defaultdict(lambda: {
                'date': None,
                'count': 0,
                'diseases': defaultdict(int),
                'avg_confidence': 0,
                'confidence_sum': 0
            }),
            'feedback_data': [],
            'processing_times': deque(maxlen=100),
            'start_time': datetime.now()
        }
        return jsonify({'success': True, 'message': 'Analytics data reset successfully'})
    except Exception as e:
        logging.error(f"Reset stats error: {e}")
        return jsonify({'error': 'Failed to reset analytics data'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)