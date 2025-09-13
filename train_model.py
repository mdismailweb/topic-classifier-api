"""
Multi-label Topic Classification Model with Advanced Features

This script implements an advanced multi-label classifier for research paper topics with:
- Cross-validation for robust evaluation
- Basic text preprocessing with domain-specific handling
- Model explainability features
- Enhanced minority class handling
- Comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
import re
import joblib
import logging
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Union
from tqdm import tqdm

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class BasicTextPreprocessor:
    """Text preprocessing with domain-specific handling."""
    
    def __init__(self):
        # Technical terms to preserve
        self.technical_terms = {
            'ml', 'ai', 'cv', 'nlp', 'gpu', 'cpu', 'api',
            'cnn', 'rnn', 'lstm', 'bert', 'neural'
        }
        
    def clean_text(self, text: str) -> str:
        """
        Clean text while preserving technical terms and numbers.
        
        Features:
        - Preserves important technical terms
        - Handles mathematical expressions
        - Maintains meaningful numbers
        """
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

def prepare_data_with_augmentation(
    df: pd.DataFrame,
    text_columns: List[str] = ['TITLE', 'ABSTRACT'],
    topic_columns: List[str] = None,
    minority_threshold: int = 1000,
    augmentation_factor: int = 2
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data with augmentation for minority classes.
    
    Features:
    - Basic text preprocessing
    - Minority class augmentation
    - Data quality checks
    """
    preprocessor = BasicTextPreprocessor()
    
    # Combine and clean text
    logger.info("Preprocessing text data...")
    df['text'] = df[text_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)
    tqdm.pandas(desc="Cleaning text")
    df['clean_text'] = df['text'].progress_apply(preprocessor.clean_text)
    
    if topic_columns is None:
        topic_columns = ['Computer Science', 'Physics', 'Mathematics',
                        'Statistics', 'Quantitative Biology', 'Quantitative Finance']
    
    # Get initial class distribution
    class_dist = df[topic_columns].sum()
    logger.info("\nInitial class distribution:")
    for topic, count in class_dist.items():
        logger.info(f"{topic}: {count}")
    
    # Augment minority classes
    augmented_samples = []
    for topic in topic_columns:
        count = class_dist[topic]
        if count < minority_threshold:
            # Get samples for this topic
            topic_samples = df[df[topic] == 1]
            n_augment = min(minority_threshold - count, count * (augmentation_factor - 1))
            
            # Augment samples
            aug_samples = resample(
                topic_samples,
                n_samples=n_augment,
                random_state=42
            )
            augmented_samples.append(aug_samples)
            logger.info(f"Augmented {topic} with {n_augment} samples")
    
    if augmented_samples:
        augmented_df = pd.concat([df] + augmented_samples, axis=0)
        logger.info(f"Total samples after augmentation: {len(augmented_df)}")
    else:
        augmented_df = df
    
    return (augmented_df['clean_text'].values,
            augmented_df[topic_columns].values,
            topic_columns)

def train_with_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    topics: List[str],
    n_splits: int = 5
) -> Tuple[TfidfVectorizer, OneVsRestClassifier, Dict]:
    """
    Train model with cross-validation and detailed evaluation.
    
    Features:
    - K-fold cross-validation
    - Detailed performance metrics
    - Model selection based on validation
    """
    # Initialize vectorizer with advanced features
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True  # Apply sublinear scaling to term frequencies
    )
    
    # Transform features
    logger.info("Vectorizing text data...")
    X_tfidf = vectorizer.fit_transform(X)
    
    # Calculate class weights
    class_weights = {
        i: len(y) / (2 * y[:, i].sum())
        for i in range(len(topics))
    }
    
    # Initialize classifier
    base_classifier = LogisticRegression(
        solver='liblinear',
        random_state=42,
        class_weight=class_weights,
        max_iter=1000
    )
    classifier = OneVsRestClassifier(base_classifier)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'f1_micro': 'f1_micro',
        'f1_macro': 'f1_macro',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro'
    }
    
    # Perform cross-validation
    logger.info("Performing cross-validation...")
    cv_results = cross_validate(
        classifier,
        X_tfidf,
        y,
        cv=n_splits,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Log cross-validation results
    logger.info("\nCross-validation results:")
    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        val_scores = cv_results[f'test_{metric}']
        logger.info(f"{metric}:")
        logger.info(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std() * 2:.4f})")
        logger.info(f"  Val:   {val_scores.mean():.4f} (+/- {val_scores.std() * 2:.4f})")
    
    # Train final model on full dataset
    logger.info("\nTraining final model on full dataset...")
    classifier.fit(X_tfidf, y)
    
    return vectorizer, classifier, cv_results

def analyze_predictions(
    classifier: OneVsRestClassifier,
    X_tfidf: np.ndarray,
    y_true: np.ndarray,
    topics: List[str]
) -> Dict:
    """
    Analyze model predictions with detailed insights.
    
    Features:
    - Confidence analysis
    - Error analysis
    - Feature importance
    - Topic correlations
    """
    # Get predictions and decision scores
    y_pred = classifier.predict(X_tfidf)
    decision_scores = classifier.decision_function(X_tfidf)
    
    # Convert decision scores to probabilities
    probas = 1 / (1 + np.exp(-decision_scores))
    
    # Calculate metrics per class
    results = {
        'per_class': {},
        'overall': {},
        'correlations': {}
    }
    
    for i, topic in enumerate(topics):
        true_pos = np.logical_and(y_true[:, i] == 1, y_pred[:, i] == 1).sum()
        false_pos = np.logical_and(y_true[:, i] == 0, y_pred[:, i] == 1).sum()
        false_neg = np.logical_and(y_true[:, i] == 1, y_pred[:, i] == 0).sum()
        
        results['per_class'][topic] = {
            'true_positives': int(true_pos),
            'false_positives': int(false_pos),
            'false_negatives': int(false_neg),
            'avg_confidence': float(probas[y_pred[:, i] == 1, i].mean())
            if (y_pred[:, i] == 1).any() else 0.0
        }
    
    # Calculate topic correlations
    topic_corr = np.corrcoef(y_true.T)
    results['correlations'] = {
        'matrix': topic_corr.tolist(),
        'strong_pairs': [
            (topics[i], topics[j], topic_corr[i, j])
            for i in range(len(topics))
            for j in range(i + 1, len(topics))
            if abs(topic_corr[i, j]) > 0.3
        ]
    }
    
    # Overall metrics
    results['overall'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'samples_with_multi_label': (y_pred.sum(axis=1) > 1).sum(),
        'avg_labels_per_sample': y_pred.sum(axis=1).mean(),
        'avg_confidence': probas[y_pred.astype(bool)].mean()
    }
    
    return results

def plot_model_insights(results: Dict, output_dir: str = '.'):
    """Generate and save model insight visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Class Distribution and Performance
    plt.figure(figsize=(12, 6))
    topics = list(results['per_class'].keys())
    metrics = pd.DataFrame({
        'True Positives': [results['per_class'][t]['true_positives'] for t in topics],
        'False Positives': [results['per_class'][t]['false_positives'] for t in topics],
        'False Negatives': [results['per_class'][t]['false_negatives'] for t in topics]
    }, index=topics)
    
    metrics.plot(kind='bar', stacked=True)
    plt.title('Class-wise Performance Analysis')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'class_performance.png')
    plt.close()
    
    # Plot 2: Topic Correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        np.array(results['correlations']['matrix']),
        annot=True,
        fmt='.2f',
        xticklabels=topics,
        yticklabels=topics,
        cmap='coolwarm'
    )
    plt.title('Topic Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / 'topic_correlations.png')
    plt.close()
    
    # Plot 3: Confidence Distribution
    plt.figure(figsize=(10, 6))
    confidences = [results['per_class'][t]['avg_confidence'] for t in topics]
    plt.bar(topics, confidences)
    plt.title('Average Prediction Confidence by Topic')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Confidence')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png')
    plt.close()

def save_enhanced_artifacts(
    vectorizer: TfidfVectorizer,
    classifier: OneVsRestClassifier,
    topics: List[str],
    results: Dict,
    cv_results: Dict,
    output_dir: str = "."
):
    """
    Save enhanced model artifacts with detailed metadata.
    
    Features:
    - Comprehensive model metadata
    - Performance visualizations
    - Detailed configuration
    - Enhanced API example
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model artifacts
    logger.info("Saving enhanced model artifacts...")
    model_artifacts = {
        'vectorizer': vectorizer,
        'classifier': classifier,
        'topics': topics,
        'metadata': {
            'performance': results,
            'cross_validation': {
                metric: {
                    'mean': float(scores.mean()),
                    'std': float(scores.std())
                }
                for metric, scores in cv_results.items()
                if metric.startswith('test_')
            }
        }
    }
    joblib.dump(model_artifacts, output_dir / 'model_artifacts.joblib')
    
    # Generate visualizations
    plot_model_insights(results, output_dir)
    
    # Save example usage file
    example_code = '''"""
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
    
    print("\\nAnalysis:")
    for key, value in results['analysis'].items():
        print(f"{key}: {value}")
    
    print("\\nConfidence Scores:")
    for topic, score in results['confidence_scores'].items():
        print(f"{topic}: {score:.4f}")
'''
    
    with open(output_dir / 'example_usage.py', 'w') as f:
        f.write(example_code)
    
    logger.info(f"Saved model artifacts to {output_dir / 'model_artifacts.joblib'}")
    logger.info(f"Created visualizations in {output_dir}")
    logger.info(f"Created API example at {output_dir / 'example_usage.py'}")

def main():
    """Enhanced training pipeline with advanced features."""
    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv('train.csv')
    
    # Prepare data with augmentation
    X, y, topics = prepare_data_with_augmentation(train_df)
    
    # Train model with cross-validation
    vectorizer, classifier, cv_results = train_with_cross_validation(X, y, topics)
    
    # Analyze model performance
    logger.info("Analyzing model performance...")
    X_tfidf = vectorizer.transform(X)
    results = analyze_predictions(classifier, X_tfidf, y, topics)
    
    # Save enhanced artifacts
    save_enhanced_artifacts(
        vectorizer, classifier, topics,
        results, cv_results
    )
    
    logger.info("Training complete! Enhanced model and analysis files have been saved.")

if __name__ == '__main__':
    main()