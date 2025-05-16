import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download  NLTK data 
# Only have to run this once than can be commented out
def download_nltk_resources():
    resources = ['punkt', 'wordnet', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)
 

# AI - Aassisted, I didn't want to bother with regex expressions...           
# Define a set of whitespace characters to handle consistently
WHITESPACE_CHARS = {' ', '\t', '\n', '\r', '\f', '\v'}

# AI - Aassisted, I didn't want to bother with regex expressions...

# Text cleaning function
def clean_text(text):
    """
    Clean text by removing special characters and normalizing all whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Normalize all whitespace (including indentations, newlines, tabs, etc.) to single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Main parsing function
def parse_text(text):

    # If you are pulling this from the repo you will want this uncommented
    # Comment this line out if already downloaded
    download_nltk_resources()
    
    # Clean the text and normalize all whitespace
    text = clean_text(text)
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = []
    for word in tokens:
        if word not in (stop_words):
            if word.strip():
                filtered_tokens.append(word)
 
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    
    for word in filtered_tokens:
        lemmatized_word = lemmatizer.lemmatize(word)
        lemmatized_tokens.append(lemmatized_word)
    
    # Join tokens back into a string with proper spacing
    processed_text = ' '.join(lemmatized_tokens)
    
    return processed_text
 
 # This runs on the training and testing data since it needs to parse through all articles
def parse_texts(texts):
    if hasattr(texts, 'apply'):
        # runs on the pandas dataframe
        return texts.apply(parse_text)
    else:
        # If its a list or other item that can be iterated through
        return [parse_text(text) for text in texts]

# Example usage
if __name__ == "__main__":
    sample_text = "This is an example text with some StopWords! It needs to be processed. https://example.com"
    processed = parse_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")