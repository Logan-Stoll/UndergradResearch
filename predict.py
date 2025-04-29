import os
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textstat import flesch_reading_ease
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class FakeNewsDetector:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self.vectorizer = None
        self.load_models()
        
    def load_models(self):
        """Load all saved models and vectorizer"""
        try:
            self.vectorizer = joblib.load(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'))
            
            # Load all available models
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.pkl')]
            for model_file in model_files:
                model_name = model_file.replace('_model.pkl', '')
                self.models[model_name] = joblib.load(os.path.join(self.models_dir, model_file))
                
            print(f"Loaded {len(self.models)} models successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Join tokens back into string
        processed_text = ' '.join(tokens)
        return processed_text
    
    def extract_text_features(self, text):
        """Extract additional features from text"""
        # Basic text statistics
        word_count = len(text.split())
        sentence_count = len(nltk.sent_tokenize(text))
        avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
        
        # Readability
        try:
            readability = flesch_reading_ease(text)
        except:
            readability = 50  # Default value
        
        # Sentiment analysis
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        
        # Count special characters
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        return [
            word_count, 
            sentence_count, 
            avg_word_length,
            readability,
            sentiment['compound'],
            exclamation_count,
            question_count
        ]
    
    def predict(self, article_text):
        """Predict whether the given article text is fake or real news"""
        if not article_text.strip():
            return {"error": "Empty text provided"}
            
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(article_text)
            
            # Vectorize the preprocessed text
            text_vector = self.vectorizer.transform([processed_text])
            
            # Extract additional features
            additional_features = np.array([self.extract_text_features(article_text)])
            
            # Combine features for models that use them
            text_vector_dense = text_vector.toarray()
            combined_features = np.hstack((text_vector_dense, additional_features))
            
            results = {}
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    # Naive Bayes only uses TF-IDF features
                    if model_name == 'naive_bayes':
                        prediction = model.predict(text_vector)[0]
                        probabilities = model.predict_proba(text_vector)[0]
                    else:
                        prediction = model.predict(combined_features)[0]
                        probabilities = model.predict_proba(combined_features)[0]
                        
                    confidence = max(probabilities) * 100
                    
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
            if all('prediction' in results[model] for model in results):
                # Weight by confidence
                real_sum = sum(results[model]['confidence'] for model in results 
                              if results[model]['prediction'] == 'Real')
                fake_sum = sum(results[model]['confidence'] for model in results 
                              if results[model]['prediction'] == 'Fake')
                
                ensemble_pred = 'Real' if real_sum > fake_sum else 'Fake'
                
                # Calculate overall confidence
                total_confidence = real_sum + fake_sum
                winning_confidence = max(real_sum, fake_sum)
                weighted_confidence = (winning_confidence / total_confidence) * 100 if total_confidence > 0 else 50
                
                results['ensemble'] = {
                    'prediction': ensemble_pred,
                    'confidence': round(weighted_confidence, 2)
                }
            
            return results
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}