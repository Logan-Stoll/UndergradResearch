import numpy as np
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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
                    from textstat import flesch_reading_ease
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