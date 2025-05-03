import pandas as pd
import numpy as np
import joblib
import os
import re
import nltk
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from textstat import flesch_reading_ease
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from text_parser import TextParser

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True) 
nltk.download('vader_lexicon', quiet=True)

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Define a feature extractor for text properties
class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract relevant features from text data"""
    
    def __init__(self, use_sentiment=True, use_readability=True, use_structural=True):
        self.use_sentiment = use_sentiment
        self.use_readability = use_readability
        self.use_structural = use_structural
        self.sia = SentimentIntensityAnalyzer()
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """Extract features from a list of texts"""
        features = []
        
        for text in X:
            text_features = []
            
            if self.use_structural:
                # Basic text statistics
                word_count = len(text.split())
                sentence_count = max(1, len(nltk.sent_tokenize(text)))
                avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
                
                # Special characters
                exclamation_count = text.count('!')
                question_count = text.count('?')
                
                text_features.extend([
                    word_count, 
                    sentence_count, 
                    avg_word_length, 
                    exclamation_count,
                    question_count
                ])
            
            if self.use_readability:
                # Readability metrics
                try:
                    readability = flesch_reading_ease(text)
                except:
                    readability = 50  # Default value
                text_features.append(readability)
            
            if self.use_sentiment:
                # Sentiment features
                sentiment = self.sia.polarity_scores(text)
                text_features.extend([
                    sentiment['neg'],
                    sentiment['neu'],
                    sentiment['pos'],
                    sentiment['compound']
                ])
            
            features.append(text_features)
        
        return np.array(features)

def train_and_save_models(feature_method='combined', parser_config=None, sample_size=None):
    """
    Train models with specified feature extraction method
    
    Args:
        feature_method: 'tfidf', 'sentiment', or 'combined'
        parser_config: dictionary of configuration options for TextParser
    """
    print(f"Starting training with feature method: {feature_method}")
    
    # Default parser configuration if none provided
    if parser_config is None:
        parser_config = {
            'remove_stopwords': True,
            'lemmatize': True,
            'stem': False,
            'remove_urls': True,
            'remove_numbers': True,
            'remove_punctuation': True,
            'remove_source_patterns': True
        }
    
    # Initialize text parser
    text_parser = TextParser(**parser_config)
    
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
    
    print("Preprocessing text data with TextParser...")
    # Apply preprocessing to the text using the text_parser
    df['processed_text'] = text_parser.batch_parse(df['text'])
    
    # Keep original text for sentiment analysis
    df['original_text'] = df['text']
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Preparing data for training...")
    # Split into features and target
    X_processed = df['processed_text']
    X_original = df['original_text']  # For sentiment analysis
    y = df['label']
    
    # Split into training and testing sets
    X_train_processed, X_test_processed, X_train_original, X_test_original, y_train, y_test = train_test_split(
        X_processed, X_original, y, test_size=0.25, random_state=42)
    
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    
    # Initialize text feature extractor (for sentiment, readability, etc.)
    text_feature_extractor = TextFeatureExtractor()
    
    # Create feature extraction pipelines based on the chosen method
    if feature_method == 'tfidf':
        print("Using TF-IDF features only...")
        # Fit and transform using TF-IDF
        X_train = tfidf_vectorizer.fit_transform(X_train_processed)
        X_test = tfidf_vectorizer.transform(X_test_processed)
        
        # Save the vectorizer
        joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
        
    elif feature_method == 'sentiment':
        print("Using sentiment and text features only...")
        # Extract text features
        X_train = text_feature_extractor.fit_transform(X_train_original)
        X_test = text_feature_extractor.transform(X_test_original)
        
        # Save the feature extractor
        joblib.dump(text_feature_extractor, 'models/text_feature_extractor.pkl')
        
    else:  # combined
        print("Using combined TF-IDF and text features...")
        # Fit and transform using TF-IDF
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_processed)
        X_test_tfidf = tfidf_vectorizer.transform(X_test_processed)
        
        # Extract text features
        X_train_features = text_feature_extractor.fit_transform(X_train_original)
        X_test_features = text_feature_extractor.transform(X_test_original)
        
        # Combine features
        X_train_tfidf_dense = X_train_tfidf.toarray()
        X_test_tfidf_dense = X_test_tfidf.toarray()
        
        X_train = np.hstack((X_train_tfidf_dense, X_train_features))
        X_test = np.hstack((X_test_tfidf_dense, X_test_features))
        
        # Save the vectorizer and feature extractor
        joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
        joblib.dump(text_feature_extractor, 'models/text_feature_extractor.pkl')
    
    # Define model configurations - prioritize linear SVM
    models = {
        'linear_svm': LinearSVC(C=1.0, random_state=42, max_iter=1000),
        'naive_bayes': MultinomialNB() if feature_method != 'sentiment' else None,  # Skip NB for sentiment-only features
        #9 Optional: add more models if needed, but comment them out initially for faster training
        'decision_tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42),
    }
    
    # Remove None models
    models = {k: v for k, v in models.items() if v is not None}
    
    # Train each model and save
    results = {}
    for name, model in models.items():
        print(f"Training {name} model...")
        
        # For Naive Bayes with combined features, we use only TF-IDF portion
        if name == 'naive_bayes' and feature_method == 'combined':
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save the model with feature method in the filename
            model_filename = f'models/{name}_{feature_method}_model.pkl'
            joblib.dump(model, model_filename)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save the model with feature method in the filename
            model_filename = f'models/{name}_{feature_method}_model.pkl'
            joblib.dump(model, model_filename)
        
        print(f"{name} accuracy: {accuracy:.4f}")
        print(f"Classification report for {name}:")
        report = classification_report(y_test, y_pred)
        print(report)
        
        # Store results for comparison
        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    # Save results
    results_file = f'models/results_{feature_method}.pkl'
    joblib.dump(results, results_file)
    
    # Determine the best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest model for {feature_method} approach: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")
    
    # Create a config file that records the best approach
    config = {
        'feature_method': feature_method,
        'best_model': best_model[0],
        'accuracy': best_model[1]['accuracy'],
        'parser_config': parser_config
    }
    
    joblib.dump(config, 'models/best_config.pkl')
    
    print(f"All models trained and saved successfully for {feature_method} approach!")
    return results

if __name__ == "__main__":
    import argparse
    
    # Create argument parser for command line options
    parser = argparse.ArgumentParser(description='Train fake news detection models')
    parser.add_argument('--sample', type=int, default=None, help='Number of samples to use for training (default: use all)')
    parser.add_argument('--feature', type=str, choices=['tfidf', 'sentiment', 'combined', 'all'], default='all',
                      help='Feature extraction method to use (default: all)')
    args = parser.parse_args()
    
    # Define parser configurations
    parser_configs = [
        {
            'remove_stopwords': True,
            'lemmatize': True,
            'stem': False,
            'remove_urls': True,
            'remove_numbers': True,
            'remove_punctuation': True,
            'remove_source_patterns': True
        }
    ]
    
    # Determine which feature methods to train
    feature_methods = ['tfidf', 'sentiment', 'combined'] if args.feature == 'all' else [args.feature]
    
    # Train with selected feature methods
    for feature_method in feature_methods:
        for i, parser_config in enumerate(parser_configs):
            print(f"\n{'='*50}")
            print(f"Training with {feature_method} features and parser config {i+1}")
            print(f"{'='*50}\n")
            train_and_save_models(feature_method, parser_config, sample_size=args.sample)