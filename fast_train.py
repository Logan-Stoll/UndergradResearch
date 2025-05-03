import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from text_parser import TextParser

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

def train_linear_svm_model(sample_size=None, use_parser=True):
    """Train a linear SVM model for fake news detection with optimized settings"""
    start_time = time.time()
    print("Starting fast training with Linear SVM...")
    
    # Initialize text parser if requested
    text_parser = None
    if use_parser:
        print("Initializing TextParser...")
        text_parser = TextParser(
            remove_stopwords=True,
            lemmatize=True,
            stem=False,
            remove_urls=True,
            remove_numbers=True,
            remove_punctuation=True,
            remove_source_patterns=True
        )
    
    print("Loading datasets...")
    # Load datasets
    fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')
    
    # Add labels
    fake['label'] = 0  # Fake news
    true['label'] = 1  # True news
    
    # Merge datasets
    df = pd.concat([fake, true], ignore_index=True)
    
    # Use a sample for faster development iterations if specified
    if sample_size:
        print(f"Using sample of {sample_size} records for faster training...")
        df = df.sample(n=sample_size, random_state=42)
    else:
        print(f"Using full dataset with {len(df)} records...")
    
    print("Preprocessing text data...")
    # Apply preprocessing to the text
    if use_parser:
        df['processed_text'] = text_parser.batch_parse(df['text'])
    else:
        df['processed_text'] = df['text']
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Splitting data into train/test sets...")
    # Split into features and target
    X_text = df['processed_text']
    y = df['label']
    
    # Split into training and testing sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.25, random_state=42)
    
    # Create a pipeline with TF-IDF vectorizer and Linear SVM
    print("Creating TF-IDF and LinearSVC pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3, max_df=0.9)),
        ('classifier', LinearSVC(C=1.0, random_state=42, max_iter=1000))
    ])
    
    # Train the model
    print("Training Linear SVM model...")
    pipeline.fit(X_train_text, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test_text)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Linear SVM accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    print("Saving model...")
    model_dir = 'models'
    pipeline_filename = os.path.join(model_dir, 'linear_svm_pipeline.pkl')
    joblib.dump(pipeline, pipeline_filename)
    
    # Save a separate vectorizer for compatibility with the existing predict.py
    vectorizer = pipeline.named_steps['tfidf']
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    
    # Save the classifier
    classifier = pipeline.named_steps['classifier']
    joblib.dump(classifier, os.path.join(model_dir, 'linear_svm_model.pkl'))
    
    # Save a best_config file for the main application
    config = {
        'feature_method': 'tfidf',
        'best_model': 'linear_svm',
        'accuracy': accuracy,
        'parser_config': {
            'remove_stopwords': True,
            'lemmatize': True,
            'stem': False,
            'remove_urls': True,
            'remove_numbers': True,
            'remove_punctuation': True,
            'remove_source_patterns': True
        } if use_parser else None
    }
    
    joblib.dump(config, os.path.join(model_dir, 'best_config.pkl'))
    
    # Calculate total training time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return accuracy, training_time

if __name__ == "__main__":
    import argparse
    
    # Create argument parser for command line options
    parser = argparse.ArgumentParser(description='Train a fast Linear SVM model for fake news detection')
    parser.add_argument('--sample', type=int, default=None, help='Number of samples to use for training (default: use all)')
    parser.add_argument('--no-parser', action='store_true', help='Skip using TextParser for preprocessing')
    args = parser.parse_args()
    
    # Train the model
    accuracy, training_time = train_linear_svm_model(
        sample_size=args.sample,
        use_parser=not args.no_parser
    )
    
    print(f"\nSummary:")
    print(f"  Model: Linear SVM")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Sample size: {args.sample if args.sample else 'Full dataset'}")
    print(f"  Using TextParser: {not args.no_parser}")