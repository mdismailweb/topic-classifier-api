"""
Create an even smaller model for deployment
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib

def create_minimal_model():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    
    # Combine text
    print("Preprocessing text...")
    train_df['text'] = train_df['TITLE'] + ' ' + train_df['ABSTRACT'].fillna('')
    
    # Define topics
    topics = ['Computer Science', 'Physics', 'Mathematics', 
              'Statistics', 'Quantitative Biology', 'Quantitative Finance']
    
    # Create minimal vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=500,  # Very reduced features
        ngram_range=(1, 1),  # Only unigrams
        max_df=0.95,
        min_df=5,
        strip_accents='unicode',
        sublinear_tf=True
    )
    
    X = vectorizer.fit_transform(train_df['text'])
    y = train_df[topics].values
    
    # Train a very simple model
    print("Training model...")
    model = OneVsRestClassifier(
        LogisticRegression(
            solver='liblinear',
            C=1.0,
            max_iter=100,
            tol=0.01  # Looser tolerance
        )
    )
    
    model.fit(X, y)
    
    # Save minimal artifacts without compression
    print("Saving model...")
    artifacts = {
        'vectorizer': vectorizer,
        'classifier': model,
        'topics': topics
    }
    
    joblib.dump(artifacts, 'minimal_model.joblib', compress=False)
    print("Done! Created minimal_model.joblib")

if __name__ == "__main__":
    create_minimal_model()