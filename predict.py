import os
import joblib
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from text_parser import TextParser
from sklearn.svm import LinearSVC
from text_features import TextFeatureExtractor  # Import the TextFeatureExtractor class

# Download required NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class FakeNewsDetector:
    def __init__(self, models_dir="models", use_best_config=True):
        self.models_dir = models_dir
        self.models = {}
        self.vectorizer = None
        self.feature_extractor = None
        self.text_parser = None
        self.feature_method = 'combined'  # default
        
        # Load configuration or models directly
        if use_best_config:
            self.load_best_config()
        else:
            self.load_models()
        
    def load_best_config(self):
        """Load the best configuration based on previous training results"""
        try:
            config_path = os.path.join(self.models_dir, 'best_config.pkl')
            
            if os.path.exists(config_path):
                config = joblib.load(config_path)
                self.feature_method = config.get('feature_method', 'combined')
                parser_config = config.get('parser_config', {})
                
                print(f"Loading best configuration: {self.feature_method} feature method")
                
                # Initialize the text parser with saved configuration
                self.text_parser = TextParser(**parser_config)
                
                # Load appropriate models and feature extractors
                self.load_models(self.feature_method)
            else:
                print("No best configuration found, loading all available models.")
                self.load_models()
        except Exception as e:
            print(f"Error loading best configuration: {e}")
            print("Falling back to loading all models.")
            self.load_models()
    
    def load_models(self, feature_method=None):
        """Load models, vectorizer, and feature extractor based on feature method"""
        try:
            # Set feature method if provided
            if feature_method:
                self.feature_method = feature_method
            
            # Initialize text parser if not already initialized
            if self.text_parser is None:
                self.text_parser = TextParser()
            
            # Load TF-IDF vectorizer if using TF-IDF or combined features
            if self.feature_method in ['tfidf', 'combined']:
                vectorizer_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
                if os.path.exists(vectorizer_path):
                    self.vectorizer = joblib.load(vectorizer_path)
                else:
                    print(f"Warning: TF-IDF vectorizer not found at {vectorizer_path}")
            
            # Load feature extractor if using sentiment or combined features
            if self.feature_method in ['sentiment', 'combined']:
                extractor_path = os.path.join(self.models_dir, 'text_feature_extractor.pkl')
                if os.path.exists(extractor_path):
                    self.feature_extractor = joblib.load(extractor_path)
                else:
                    print(f"Warning: Text feature extractor not found at {extractor_path}")
            
            # Load all models with the appropriate feature method in the filename
            model_files = [f for f in os.listdir(self.models_dir) 
                          if f.endswith('_model.pkl') and (
                              self.feature_method in f or 
                              not any(method in f for method in ['tfidf', 'sentiment', 'combined'])
                          )]
            
            for model_file in model_files:
                # Extract model name without feature method and _model.pkl suffix
                if self.feature_method in model_file:
                    model_name = model_file.replace(f'_{self.feature_method}_model.pkl', '')
                else:
                    model_name = model_file.replace('_model.pkl', '')
                
                self.models[model_name] = joblib.load(os.path.join(self.models_dir, model_file))
                
            print(f"Loaded {len(self.models)} models successfully!")
            print(f"Available models: {', '.join(self.models.keys())}")
            
            # If no models were loaded, try loading any model
            if len(self.models) == 0:
                print("No models found for the specified feature method. Loading any available models.")
                all_model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.pkl')]
                
                for model_file in all_model_files:
                    model_name = model_file.replace('_model.pkl', '')
                    if '_' in model_name:  # Handle feature method in filename
                        model_name = model_name.split('_')[0]
                    
                    self.models[model_name] = joblib.load(os.path.join(self.models_dir, model_file))
                
                print(f"Loaded {len(self.models)} alternative models.")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def extract_features(self, text, processed_text=None):
        """Extract features based on the feature method"""
        if processed_text is None:
            processed_text = self.text_parser.parse(text)
        
        features = {}
        
        # Extract TF-IDF features if using TF-IDF or combined approach
        if self.feature_method in ['tfidf', 'combined'] and self.vectorizer:
            features['tfidf'] = self.vectorizer.transform([processed_text])
        
        # Extract text features if using sentiment or combined approach
        if self.feature_method in ['sentiment', 'combined'] and self.feature_extractor:
            features['text_features'] = self.feature_extractor.transform([text])
        
        return features
    
    def predict(self, article_text):
        """Predict whether the given article text is fake or real news"""
        if not article_text.strip():
            return {"error": "Empty text provided"}
            
        try:
            # Process the text using text_parser
            processed_text = self.text_parser.parse(article_text)
            
            # Extract features based on the feature method
            features = self.extract_features(article_text, processed_text)
            
            # Initialize results dictionary
            results = {}
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    # Prepare features for this model
                    if model_name == 'naive_bayes' and self.feature_method == 'combined':
                        # Naive Bayes only uses TF-IDF features
                        X = features['tfidf']
                    elif self.feature_method == 'tfidf':
                        X = features['tfidf']
                    elif self.feature_method == 'sentiment':
                        X = features['text_features']
                    else:  # combined
                        X_tfidf = features['tfidf'].toarray()
                        X_text = features['text_features']
                        X = np.hstack((X_tfidf, X_text))
                        
                    # Make prediction
                    prediction = model.predict(X)[0]
                    
                    # Handle LinearSVC which doesn't have predict_proba
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X)[0]
                        confidence = max(probabilities) * 100
                    elif isinstance(model, LinearSVC) or hasattr(model, 'decision_function'):
                        # For LinearSVC, use decision function to approximate confidence
                        decision_score = model.decision_function(X)[0]
                        # Convert to probability-like value between 0 and 1
                        if isinstance(decision_score, np.ndarray):
                            decision_score = abs(decision_score).max()
                        confidence = min(100, max(50, 50 + abs(decision_score) * 10))
                    else:
                        # Fallback for models without probability or decision function
                        confidence = 70.0
                    
                    results[model_name] = {
                        'prediction': 'Real' if prediction == 1 else 'Fake',
                        'confidence': round(confidence, 2)
                    }
                except Exception as e:
                    results[model_name] = {
                        'prediction': 'Error',
                        'confidence': 0,
                        'error': str(e)
                    }
            
            # Calculate ensemble prediction (weighted vote)
            if results and all('prediction' in results.get(model, {}) for model in results):
                # Weight by confidence
                real_sum = sum(results[model]['confidence'] for model in results 
                              if results[model].get('prediction') == 'Real')
                fake_sum = sum(results[model]['confidence'] for model in results 
                              if results[model].get('prediction') == 'Fake')
                
                ensemble_pred = 'Real' if real_sum > fake_sum else 'Fake'
                
                # Calculate overall confidence
                total_confidence = real_sum + fake_sum
                winning_confidence = max(real_sum, fake_sum)
                weighted_confidence = (winning_confidence / total_confidence) * 100 if total_confidence > 0 else 50
                
                results['ensemble'] = {
                    'prediction': ensemble_pred,
                    'confidence': round(weighted_confidence, 2),
                }