#!/usr/bin/env python3
"""
Multi-label Topic Classification API
Author: Your Name
Date: September 13, 2025

This Flask API provides topic classification for research papers.
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from pathlib import Path
import re
import os

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
        # Look for minimal model in the same directory as the script
        model_path = Path(__file__).parent / 'minimal_model.joblib'
        print(f"Attempting to load model from: {model_path}")
        
        if not model_path.exists():
            print(f"Model file not found at {model_path}")
            return None
            
        print("Loading model...")
        artifacts = joblib.load(str(model_path))
        print("Model loaded successfully!")
        return artifacts
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

print("Starting model loading process...")
model_artifacts = load_model()

# Initialize app with model status
if model_artifacts is None:
    print("WARNING: Model failed to load. API will return errors for predictions.")
else:
    print(f"Model loaded successfully with topics: {model_artifacts.get('topics', [])}")

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

@app.route('/')
def home():
    """Home page with API documentation."""
    return '''
    <html>
        <head>
            <title>Topic Classification API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                pre { background: #f4f4f4; padding: 15px; border-radius: 5px; }
                .endpoint { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Research Paper Topic Classification API</h1>
            <p>This API classifies research papers into multiple topics based on their title and abstract.</p>
            
            <div class="endpoint">
                <h2>Health Check</h2>
                <pre>GET /health</pre>
                <p>Check if the API is running and model is loaded.</p>
            </div>
            
            <div class="endpoint">
                <h2>Predict Topics</h2>
                <pre>POST /predict
Content-Type: application/json

{
    "title": "Your paper title",
    "abstract": "Your paper abstract",
    "confidence_threshold": 0.3
}</pre>
                <p>Returns predicted topics with confidence scores.</p>
            </div>
            
            <div class="endpoint">
                <h2>Available Topics</h2>
                <ul>
                    <li>Computer Science</li>
                    <li>Physics</li>
                    <li>Mathematics</li>
                    <li>Statistics</li>
                    <li>Quantitative Biology</li>
                    <li>Quantitative Finance</li>
                </ul>
            </div>
        </body>
    </html>
    '''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)