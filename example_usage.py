"""
Enhanced Topic Classification API

Features:
- Basic text preprocessing
- Confidence scores
- Multi-label predictions
- Detailed analysis
"""

import joblib
import numpy as np
from pathlib import Path
import re
from typing import Dict

def predict_topics(
    text: str,
    artifacts_path: str = 'model_artifacts.joblib',
    confidence_threshold: float = 0.3
) -> Dict:
    """
    Predict topics with detailed analysis.
    
    Args:
        text: Paper title and/or abstract
        artifacts_path: Path to model artifacts
        confidence_threshold: Minimum confidence for predictions
    
    Returns:
        Dict containing:
        - predictions: List of predicted topics with confidence
        - analysis: Detailed prediction analysis
        - confidence_scores: All topic confidence scores
    """
    # Load artifacts
    artifacts = joblib.load(artifacts_path)
    
    # Process text
    preprocessor = BasicTextPreprocessor()
    cleaned_text = preprocessor.clean_text(text)
    X = artifacts['vectorizer'].transform([cleaned_text])
    
    # Get predictions and confidence scores
    decision_scores = artifacts['classifier'].decision_function(X)
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
            'topic': artifacts['topics'][i],
            'confidence': float(probas[0][i])
        }
        for i in range(len(artifacts['topics']))
        if y_pred[i] == 1
    ]
    
    # Sort predictions by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'predictions': predictions,
        'analysis': {
            'num_predictions': len(predictions),
            'avg_confidence': float(np.mean([p['confidence'] for p in predictions]))
        },
        'confidence_scores': {
            topic: float(score)
            for topic, score in zip(artifacts['topics'], probas[0])
        }
    }

# Example usage
if __name__ == '__main__':
    sample_text = """
    This paper introduces a novel quantum computing approach
    that combines statistical methods with deep learning for
    optimization problems in financial markets.
    """
    
    results = predict_topics(sample_text)
    
    print("Predicted Topics:")
    for pred in results['predictions']:
        print(f"{pred['topic']}: {pred['confidence']:.4f}")
    
    print("\nAnalysis:")
    for key, value in results['analysis'].items():
        print(f"{key}: {value}")
    
    print("\nConfidence Scores:")
    for topic, score in results['confidence_scores'].items():
        print(f"{topic}: {score:.4f}")
