"""Feature engineering module with TF-IDF and sentiment analysis"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
import pickle


class FeatureEngineer:
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.drug_encoder = LabelEncoder()
        self.condition_encoder = LabelEncoder()
        self.is_fitted = False
    
    def extract_sentiment(self, text):
        """Extract sentiment score using VADER"""
        if pd.isna(text) or text == "":
            return 0.0
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores['compound']
        except:
            return 0.0
    
    def fit_transform_tfidf(self, train_reviews, test_reviews):
        """Fit and transform TF-IDF features"""
        print("Extracting TF-IDF features...")
        
        # Fit on training data
        train_tfidf = self.tfidf_vectorizer.fit_transform(train_reviews)
        test_tfidf = self.tfidf_vectorizer.transform(test_reviews)
        
        # Convert to DataFrame
        feature_names = [f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
        train_tfidf_df = pd.DataFrame(
            train_tfidf.toarray(),
            columns=feature_names,
            index=train_reviews.index
        )
        test_tfidf_df = pd.DataFrame(
            test_tfidf.toarray(),
            columns=feature_names,
            index=test_reviews.index
        )
        
        return train_tfidf_df, test_tfidf_df
    
    def extract_all_features(self, train_df, test_df):
        """Extract all features from datasets"""
        print("Extracting sentiment features...")
        
        # Sentiment analysis on original reviews
        train_df['sentiment_score'] = train_df['review'].apply(self.extract_sentiment)
        test_df['sentiment_score'] = test_df['review'].apply(self.extract_sentiment)
        
        # TF-IDF features
        train_tfidf_df, test_tfidf_df = self.fit_transform_tfidf(
            train_df['cleaned_review'],
            test_df['cleaned_review']
        )
        
        # Encode categorical features
        print("Encoding categorical features...")
        all_drugs = pd.concat([train_df['drugName'], test_df['drugName']])
        all_conditions = pd.concat([train_df['condition'], test_df['condition']])
        
        self.drug_encoder.fit(all_drugs)
        self.condition_encoder.fit(all_conditions)
        
        train_df['drugName_encoded'] = self.drug_encoder.transform(train_df['drugName'])
        test_df['drugName_encoded'] = self.drug_encoder.transform(test_df['drugName'])
        
        train_df['condition_encoded'] = self.condition_encoder.transform(train_df['condition'])
        test_df['condition_encoded'] = self.condition_encoder.transform(test_df['condition'])
        
        # Combine features
        feature_cols = ['drugName_encoded', 'condition_encoded', 'usefulCount', 
                        'day', 'month', 'year', 'sentiment_score']
        
        # Handle missing values in numeric columns
        for col in ['usefulCount', 'day', 'month', 'year']:
            train_df[col] = train_df[col].fillna(0)
            test_df[col] = test_df[col].fillna(0)
        
        X_train = pd.concat([
            train_df[feature_cols].reset_index(drop=True),
            train_tfidf_df.reset_index(drop=True)
        ], axis=1)
        
        X_test = pd.concat([
            test_df[feature_cols].reset_index(drop=True),
            test_tfidf_df.reset_index(drop=True)
        ], axis=1)
        
        y_train = train_df['effective'].values
        y_test = test_df['effective'].values
        
        self.is_fitted = True
        print("Feature engineering complete!")
        
        return X_train, X_test, y_train, y_test, train_df, test_df
    
    def save(self, path: str = 'models/feature_engineer.pkl'):
        """Save feature engineer"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str = 'models/feature_engineer.pkl'):
        """Load feature engineer"""
        with open(path, 'rb') as f:
            return pickle.load(f)
        