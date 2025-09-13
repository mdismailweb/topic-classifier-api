"""
Create a smaller, optimized model for deployment
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib

def create_deployment_model():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    
    # Combine text
    print("Preprocessing text...")
    train_df['text'] = train_df['TITLE'] + ' ' + train_df['ABSTRACT'].fillna('')
    
    # Define topics
    topics = ['Computer Science', 'Physics', 'Mathematics', 
              'Statistics', 'Quantitative Biology', 'Quantitative Finance']
    
    # Create vectorizer with minimal features
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Reduced features
        ngram_range=(1, 1),  # Only unigrams
        max_df=0.95,
        min_df=2
    )
    
    X = vectorizer.fit_transform(train_df['text'])
    y = train_df[topics].values
    
    # Train a simpler model
    print("Training model...")
    model = OneVsRestClassifier(
        LogisticRegression(
            solver='liblinear',
            C=1.0,
            max_iter=100
        )
    )
    
    model.fit(X, y)
    
    # Save minimal artifacts
    print("Saving model...")
    artifacts = {
        'vectorizer': vectorizer,
        'classifier': model,
        'topics': topics
    }
    
    joblib.dump(artifacts, 'model_small.joblib', compress=3)
    print("Done! Created model_small.joblib")

if __name__ == "__main__":
    create_deployment_model()