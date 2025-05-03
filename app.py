from flask import Flask, request, render_template, jsonify, redirect, url_for
from predict import FakeNewsDetector
import os
import joblib

app = Flask(__name__)

# Initialize the detector with best config by default
detector = FakeNewsDetector(models_dir="models", use_best_config=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    feature_method = detector.feature_method  # Get current feature method
    
    if request.method == 'POST':
        article_text = request.form.get('article_text', '')
        if article_text:
            result = detector.predict(article_text)
    
    return render_template('index.html', result=result, feature_method=feature_method)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """API endpoint for predictions"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    result = detector.predict(data['text'])
    return jsonify(result)

@app.route('/switch_method/<method>', methods=['GET'])
def switch_method(method):
    """Switch between feature extraction methods"""
    if method in ['tfidf', 'sentiment', 'combined']:
        # Reload models with the specified feature method
        detector.load_models(feature_method=method)
        return redirect(url_for('index'))
    else:
        return jsonify({'error': 'Invalid feature method'}), 400

@app.route('/details')
def details():
    """Page showing model details and feature importance"""
    # Load results for different feature methods
    results = {}
    feature_methods = ['tfidf', 'sentiment', 'combined']
    
    for method in feature_methods:
        results_file = f'models/results_{method}.pkl'
        if os.path.exists(results_file):
            results[method] = joblib.load(results_file)
    
    # Get current best config
    best_config = None
    if os.path.exists('models/best_config.pkl'):
        best_config = joblib.load('models/best_config.pkl')
    
    return render_template('details.html', 
                          results=results, 
                          best_config=best_config, 
                          current_method=detector.feature_method)

if __name__ == '__main__':
    # Check if models exist, if not, suggest running training script
    if not os.path.exists('models/best_config.pkl'):
        print("Best configuration not found! Please run improved_train.py first.")
    
    # Print current feature method and available models
    print(f"Current feature method: {detector.feature_method}")
    print(f"Available models: {', '.join(detector.models.keys())}")
        
    app.run(debug=True)