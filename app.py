from flask import Flask, request, jsonify
import joblib
import numpy as np
from pathlib import Path
import re

app = Flask(__name__)

class BasicTextPreprocessor:
    """Text preprocessing with domain-specific handling."""
    
    def __init__(self):
        # Technical terms to preserve
        self.technical_terms = {
            'ml', 'ai', 'cv', 'nlp', 'gpu', 'cpu', 'api',
            'cnn', 'rnn', 'lstm', 'bert', 'neural'
        }
        
    def clean_text(self, text: str) -> str:
        """Clean text while preserving technical terms and numbers."""
        # Convert to lowercase
        text = text.lower()
        
        # Handle numbers and math operators
        text = re.sub(r'(\d+\.?\d*)', r' \1 ', text)  # Preserve numbers
        text = re.sub(r'[+\-*/=]', ' ', text)  # Remove math operators
        
        # Clean remaining text while preserving letters, numbers, and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text

# Load model artifacts at startup
def load_model():
    try:
        artifacts = joblib.load('model_artifacts.joblib')
        return artifacts
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model_artifacts = load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model_artifacts is not None:
        return jsonify({
            'status': 'healthy',
            'message': 'Model loaded successfully'
        })
    return jsonify({
        'status': 'unhealthy',
        'message': 'Model not loaded'
    }), 500

@app.route('/predict', methods=['POST'])
def predict_topics():
    """
    Predict topics for a given research paper.
    
    Expected JSON input:
    {
        "title": "string",
        "abstract": "string",
        "confidence_threshold": float (optional, default=0.3)
    }
    """
    try:
        # Get input data
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No input data provided'
            }), 400
        
        # Extract fields
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        confidence_threshold = float(data.get('confidence_threshold', 0.3))
        
        # Validate input
        if not title and not abstract:
            return jsonify({
                'error': 'Both title and abstract cannot be empty'
            }), 400
        
        # Combine and preprocess text
        text = f"{title} {abstract}".strip()
        preprocessor = BasicTextPreprocessor()
        cleaned_text = preprocessor.clean_text(text)
        
        # Vectorize
        X = model_artifacts['vectorizer'].transform([cleaned_text])
        
        # Get predictions and confidence scores
        decision_scores = model_artifacts['classifier'].decision_function(X)
        probas = 1 / (1 + np.exp(-decision_scores))
        
        # Make predictions
        y_pred = (probas[0] >= confidence_threshold).astype(int)
        
        # Ensure at least one prediction
        if y_pred.sum() == 0:
            best_label = np.argmax(probas[0])
            y_pred[best_label] = 1
        
        # Prepare results
        predictions = [
            {
                'topic': model_artifacts['topics'][i],
                'confidence': float(probas[0][i])
            }
            for i in range(len(model_artifacts['topics']))
            if y_pred[i] == 1
        ]
        
        # Sort predictions by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'metadata': {
                'input_length': len(text),
                'num_predictions': len(predictions),
                'avg_confidence': float(np.mean([p['confidence'] for p in predictions]))
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)