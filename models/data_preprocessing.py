"""Data preprocessing module for drug review dataset"""
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def decode_html(self, text):
        """Decode HTML-encoded characters and remove all HTML tags"""
        if pd.isna(text):
            return ""
        try:
            # Use BeautifulSoup to strip ALL HTML tags including </span>, <span>, etc.
            return BeautifulSoup(str(text), "html.parser").get_text()
        except:
            return str(text)
    
    def clean_review(self, text):
        """Clean and normalize review text"""
        if pd.isna(text) or text == "":
            return ""
        
        # HTML decoding
        text = self.decode_html(text)
        
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Lowercase
        text = text.lower()
        
        # Tokenization
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def load_and_preprocess(self, train_path: str, test_path: str, sample_frac: float = 0.3):
        """Load and preprocess train and test datasets"""
        print("Loading datasets...")
        
        # Load data with proper encoding
        train_df = pd.read_csv(train_path, sep='\t', on_bad_lines='skip')
        test_df = pd.read_csv(test_path, sep='\t', on_bad_lines='skip')
        
        # Sample for faster processing (adjust as needed)
        if sample_frac < 1.0:
            train_df = train_df.sample(frac=sample_frac, random_state=42)
            test_df = test_df.sample(frac=sample_frac, random_state=42)
        
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        # Clean HTML from condition and drugName columns
        print("Cleaning HTML tags from data...")
        for df in [train_df, test_df]:
            # Remove ALL HTML tags from condition names (removes </span>, <span>, etc.)
            df['condition'] = df['condition'].apply(self.decode_html)
            # Remove ALL HTML tags from drug names
            df['drugName'] = df['drugName'].apply(self.decode_html)
            # Strip extra whitespace
            df['condition'] = df['condition'].str.strip()
            df['drugName'] = df['drugName'].str.strip()
        
        # Drop rows with missing conditions
        train_df = train_df.dropna(subset=['condition'])
        test_df = test_df.dropna(subset=['condition'])
        
        # Drop rows with empty conditions after cleaning
        train_df = train_df[train_df['condition'] != '']
        test_df = test_df[test_df['condition'] != '']
        
        # Fill missing reviews
        train_df['review'] = train_df['review'].fillna("")
        test_df['review'] = test_df['review'].fillna("")
        
        # Extract date features
        print("Extracting date features...")
        for df in [train_df, test_df]:
            df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y', errors='coerce')
            df['day'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            df.drop('date', axis=1, inplace=True)
        
        # Clean reviews
        print("Cleaning reviews...")
        train_df['cleaned_review'] = train_df['review'].apply(self.clean_review)
        test_df['cleaned_review'] = test_df['review'].apply(self.clean_review)
        
        # Create effectiveness labels (rating >= 7 is effective)
        train_df['effective'] = (train_df['rating'] >= 7).astype(int)
        test_df['effective'] = (test_df['rating'] >= 7).astype(int)
        
        # Drop first column if unnamed
        if 'Unnamed: 0' in train_df.columns:
            train_df.drop('Unnamed: 0', axis=1, inplace=True)
        if 'Unnamed: 0' in test_df.columns:
            test_df.drop('Unnamed: 0', axis=1, inplace=True)
        
        print("✓ HTML tags removed from conditions and drug names")
        print("Preprocessing complete!")
        return train_df, test_df