from flask import Flask, request, render_template, jsonify
from predict import FakeNewsDetector
import os

app = Flask(__name__)
detector = FakeNewsDetector(models_dir="models")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        article_text = request.form.get('article_text', '')
        if article_text:
            result = detector.predict(article_text)
    return render_template('index.html', result=result)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """API endpoint for predictions"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    result = detector.predict(data['text'])
    return jsonify(result)

if __name__ == '__main__':
    # Check if models exist, if not, suggest running training script
    if not os.path.exists('models/tfidf_vectorizer.pkl'):
        print("Models not found! Please run train_models.py first.")
    
    app.run(debug=True)