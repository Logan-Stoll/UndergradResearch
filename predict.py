import pickle
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from parse import parse_text

def load_models_and_vectorizer():
    # Get all the models
    models = {}
    model_files = [f for f in os.listdir('models') if f.endswith('_model.pkl')]
    
    # Load the vectorizer
    try:
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        print("Error: TF-IDF vectorizer not found")
        return None, None
    
    # Load each model
    for model_file in model_files:
        model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
        try:
            with open(f'models/{model_file}', 'rb') as f:
                models[model_name] = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Model {model_file} not found")
    
    if not models:
        print("Error: No models found")
        return None, None
    
    return vectorizer, models

def predict_text(text, vectorizer, models):
    # Parse the input text using the new parsing function
    parsed_text = parse_text(text)
    
    # Vectorize the text
    text_vectorized = vectorizer.transform([parsed_text])
    
    # Get predictions from each model
    results = {}
    for model_name, model in models.items():
        prediction = model.predict(text_vectorized)[0]
        label = "REAL" if prediction == 1 else "FAKE"
        
     
        confidence = None
        
        # For models that support probability use predict_proba
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(text_vectorized)[0]
                confidence = probs[prediction] * 100  # This just makes it a percentage
            except:
                pass

        # For LinearSVC use decision_function
        elif hasattr(model, 'decision_function') and model.__class__.__name__ == 'LinearSVC':
            try:
               
                ''' LinearSVC does not just give back a probibility 
                I don't fully understand how this works but it essentially gets the distance from 
                the decision boundry for the text function (line 72) which is like a raw score.
                Then I use the sigmoid function (line 74) to make it a probability between 1 - 0.
                Learn more at: 
                https://scikit-learn.org/stable/modules/svm.html#scores-probabilities
                https://scikit-learn.org/stable/modules/calibration.html'''  
            
                decision_score = model.decision_function(text_vectorized)[0]
                import numpy as np
                raw_confidence = 1 / (1 + np.exp(-abs(decision_score)))
                
                # Make a percentage
                confidence = raw_confidence * 100
            except:
                pass
        
        # Store result
        results[model_name] = {
            'prediction': label,
            'confidence': f"{confidence:.2f}%" if confidence is not None else "N/A"
        }
    
    return results

def ensemble_prediction(results):
    votes = {'FAKE': 0, 'REAL': 0}
    for model_name, result in results.items():
        votes[result['prediction']] += 1
    
    # Determine the majority vote
    if votes['FAKE'] > votes['REAL']:
        return 'FAKE', votes['FAKE'] / sum(votes.values()) * 100
    elif votes['REAL'] > votes['FAKE']:
        return 'REAL', votes['REAL'] / sum(votes.values()) * 100
    else:
        return 'UNDECIDED', 50.0  # Tie

def classify_article(article_text):
    # Load models and vectorizer
    vectorizer, models = load_models_and_vectorizer()
    if not vectorizer or not models:
        return {
            'error': 'Models not found. Please train models first.'
        }
    
    results = predict_text(article_text, vectorizer, models)
    
    ensemble_label, ensemble_confidence = ensemble_prediction(results)
    
    word_count = len(article_text.split())
    
    # Format results for the Flask app
    return {
        'article_length': word_count,
        'model_predictions': results,
        'ensemble': {
            'prediction': ensemble_label,
            'confidence': f"{ensemble_confidence:.1f}%"
        }
    }

# Example 
if __name__ == "__main__":
    sample_text = "This is a sample article text for testing."
    results = classify_article(sample_text)
    print(results)