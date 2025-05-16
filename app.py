from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from predict import classify_article
from article_scraper import ArticleScraper
import validators

app = Flask(__name__)
# Secret key for flash messages
app.secret_key = 'fake_news_detection_app_secret_key'

# Initialize article scraper
scraper = ArticleScraper()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    """Extract text from URL and allow user to edit before analysis"""
    article_url = request.form.get('article_url', '').strip()
    
    if not article_url:
        flash("No URL provided", "danger")
        return redirect(url_for('home'))
    
    # Try to scrape text from URL
    extracted_text, error = scraper.get_article_text(article_url)
    
    if error:
        flash(f"Error: {error}", "danger")
        return redirect(url_for('home'))
    
    # Show extracted text for review/editing
    return render_template('review.html', article_text=extracted_text, article_url=article_url)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if input is URL or text
    article_url = request.form.get('article_url', '').strip()
    article_text = request.form.get('article_text', '').strip()
    
    # Ensure we have text to analyze
    if not article_text:
        flash("No text provided for analysis", "danger")
        return redirect(url_for('home'))
    
    # Process the text through the fake news detection models
    results = classify_article(article_text)
    
    # If a URL was used add it to the results
    if article_url:
        results['source_url'] = article_url
    
    return render_template('result.html', results=results)

@app.route('/api/extract', methods=['POST'])
def api_extract():
    
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url']
    extracted_text, error = scraper.get_article_text(url)

    if error:
        return jsonify({'error': error}), 400
    
    # Turn the article text in to a json object for the HTML 
    return jsonify({'text': extracted_text})

if __name__ == '__main__':
    app.run(debug=True)