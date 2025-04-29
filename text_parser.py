import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

class TextParser:
    def __init__(self, remove_stopwords=True, lemmatize=True, stem=False, 
                 remove_urls=True, remove_numbers=True, remove_punctuation=True,
                 remove_source_patterns=True):
        
        # Download necessary NLTK resources
        self._download_nltk_resources()
        
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_source_patterns = remove_source_patterns
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Patterns that could bias the dataset
        # AI-Assisted
        self.source_patterns = [
            # Location (Source) patterns, common in Reuters, AP, etc.
            r'[A-Z]+,\s+[A-Za-z]+\s+\d+\s+\([A-Za-z]+\)\s+[-–]',  # LONDON, March 25 (Reuters) -
            r'[A-Z]+\s+\([A-Za-z]+\)\s+[-–]',  # WASHINGTON (Reuters) -
            r'[A-Z]+\s+\([A-Za-z]+/[A-Za-z]+\)\s+[-–]',  # WASHINGTON (Reuters/AP) -
            
            # Date patterns at beginning of articles
            r'^\s*\w+\s+\d{1,2},\s+\d{4}\s+[-–]',  # January 15, 2022 -
            
            # News agencies at beginning
            r'^\s*Reuters\s+[-–]',
            r'^\s*Associated Press\s+[-–]',
            r'^\s*AP\s+[-–]',
            r'^\s*CNN\s+[-–]',
            r'^\s*BBC\s+[-–]',
            r'^\s*Fox News\s+[-–]',
            
            # Bylines
            r'^\s*By\s+[A-Za-z\s\.]+\s+[-–]',  # By John Smith -
        ]
    
    def _download_nltk_resources(self):
        #Download required NLTK resources if they're not available

        resources = ['punkt', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                print(f"Checking for NLTK resource: {resource}")
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource)
            except Exception as e:
                print(f"Warning: Error checking/downloading {resource}: {e}")
                
        # Special case for punkt which is in tokenizers
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK resource: punkt")
            nltk.download('punkt')
        except Exception as e:
            print(f"Warning: Error checking/downloading punkt: {e}")
    
    def remove_agency_patterns(self, text):
        # Remove news agency identifiers and other patterns that might bias the dataset
        if not self.remove_source_patterns:
            return text
            
        # Apply each pattern
        for pattern in self.source_patterns:
            text = re.sub(pattern, '', text)
        
        # Remove any agency references with parentheses anywhere in text
        text = re.sub(r'\([A-Za-z]+\)', '', text)
        
        return text
    
    def parse(self, text):
        # Process the input text by applying various text cleaning operations
        
        if not isinstance(text, str):
            return ""
        
        # Remove news agency patterns first
        if self.remove_source_patterns:
            text = self.remove_agency_patterns(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize text
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # If tokenization fails, fall back to simple whitespace tokenization
            print("Warning: NLTK tokenization failed, falling back to basic tokenization")
            tokens = text.split()
        
        # Remove stopwords
        if self.remove_stopwords:
            try:
                tokens = [word for word in tokens if word not in self.stop_words]
            except:
                print("Warning: Stopword removal failed")
        
        # Lemmatize words
        if self.lemmatize:
            try:
                tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            except:
                print("Warning: Lemmatization failed")
        
        # Stem words
        if self.stem:
            try:
                tokens = [self.stemmer.stem(word) for word in tokens]
            except:
                print("Warning: Stemming failed")
        
        # Join tokens back into text
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def batch_parse(self, texts):
        # Process a batch of texts (e.g., DataFrame column)
        return [self.parse(text) for text in texts]


# Example usage
if __name__ == "__main__":
    parser = TextParser()
    
    # Test with news source patterns
    sample_texts = [
        "WASHINGTON (Reuters) - The president announced today...",
        "NEW YORK (AP) - Stock markets are rising...",
        "By John Smith - In today's news...",
        "January 15, 2022 - The Federal Reserve has decided...",
        "This is a normal text with (Reuters) in the middle."
    ]
    
    for text in sample_texts:
        processed = parser.parse(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print("-" * 50)