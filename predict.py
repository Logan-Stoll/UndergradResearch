import os
import joblib
import numpy as np

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
            self.models['decision_tree'] = joblib.load(os.path.join(self.models_dir, 'decision_tree_model.pkl'))
            self.models['random_forest'] = joblib.load(os.path.join(self.models_dir, 'random_forest_model.pkl'))
            self.models['neural_network'] = joblib.load(os.path.join(self.models_dir, 'neural_network_model.pkl'))
            print("All models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def predict(self, article_text):
        """Predict whether the given article text is fake or real news"""
        if not article_text.strip():
            return {"error": "Empty text provided"}
            
        # Vectorize the input text
        try:
            text_vector = self.vectorizer.transform([article_text])
        except Exception as e:
            return {"error": f"Error vectorizing text: {e}"}
        
        results = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                prediction = model.predict(text_vector)[0]
                probabilities = model.predict_proba(text_vector)[0]
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
        
        # Calculate ensemble prediction (majority vote)
        if all('prediction' in results[model] for model in results):
            predictions = [1 if results[model]['prediction'] == 'Real' else 0 for model in results]
            ensemble_pred = 'Real' if sum(predictions) > len(predictions)/2 else 'Fake'
            
            # Average confidence
            confidences = [results[model]['confidence'] for model in results]
            avg_confidence = sum(confidences) / len(confidences)
            
            results['ensemble'] = {
                'prediction': ensemble_pred,
                'confidence': round(avg_confidence, 2)
            }
        
        return results