import pandas as pd
import numpy as np
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from textstat import flesch_reading_ease, syllable_count, lexicon_count
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

def preprocess_text(text):
    """Clean and preprocess text data"""
    if isinstance(text, str):
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
    else:
        return ""

def extract_text_features(text):
    """Extract additional features from text"""
    if not isinstance(text, str) or len(text) == 0:
        return [0, 0, 0, 0, 0, 0, 0]
    
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

def create_feature_matrix(texts):
    """Create a matrix of additional features for the texts"""
    feature_matrix = []
    for text in texts:
        features = extract_text_features(text)
        feature_matrix.append(features)
    return np.array(feature_matrix)

def train_and_save_models():
    print("Loading datasets...")
    # Load datasets
    fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')
    
    # Add labels
    fake['label'] = 0  # Fake news
    true['label'] = 1  # True news
    
    # Merge datasets
    df = pd.concat([fake, true], ignore_index=True)
    
    # For training efficiency, can use a smaller subset if needed
    # Uncomment the following line to use a sample
    # df = df.sample(n=10000, random_state=42)
    
    print("Preprocessing text data...")
    # Apply preprocessing to the text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Preparing data for training...")
    # Split into features and target
    X_text = df['processed_text']
    y = df['label']
    
    # Split into training and testing sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.25, random_state=42)
    
    # Extract additional features
    print("Extracting additional text features...")
    X_train_features = create_feature_matrix(X_train_text)
    X_test_features = create_feature_matrix(X_test_text)
    
    # Vectorize the text
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    # Combine TF-IDF features with additional features
    X_train_tfidf_dense = X_train_tfidf.toarray()
    X_test_tfidf_dense = X_test_tfidf.toarray()
    
    X_train = np.hstack((X_train_tfidf_dense, X_train_features))
    X_test = np.hstack((X_test_tfidf_dense, X_test_features))
    
    # Save the vectorizer
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    # Train and evaluate models
    models = {
        'decision_tree': DecisionTreeClassifier(max_depth=20, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', alpha=0.0001, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'svm': SVC(kernel='linear', probability=True, random_state=42),
        'naive_bayes': MultinomialNB()
    }
    
    # Train each model and save
    for name, model in models.items():
        print(f"Training {name} model...")
        
        # For Naive Bayes, we need positive features
        if name == 'naive_bayes':
            # Naive Bayes needs non-negative features, so use only TF-IDF
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            score = model.score(X_test_tfidf, y_test)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = model.score(X_test, y_test)
        
        print(f"{name} accuracy: {score:.4f}")
        print(f"Classification report for {name}:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        joblib.dump(model, f'models/{name}_model.pkl')
    
    print("All models trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_models()