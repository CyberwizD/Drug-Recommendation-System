"""Drug recommendation engine with ranking algorithm"""
import pandas as pd
import numpy as np
from typing import List, Dict
import pickle


class RecommendationEngine:
    def __init__(self, model, feature_engineer, train_df, test_df):
        self.model = model
        self.feature_engineer = feature_engineer
        # Combine train and test for full dataset
        self.full_df = pd.concat([train_df, test_df], ignore_index=True)
        self.condition_drugs = self._build_condition_index()
    
    def _build_condition_index(self):
        """Build an index of drugs per condition"""
        condition_drugs = {}
        for condition in self.full_df['condition'].unique():
            drugs = self.full_df[self.full_df['condition'] == condition]['drugName'].unique()
            condition_drugs[condition] = list(drugs)
        return condition_drugs
    
    def get_available_conditions(self):
        """Get list of all available conditions"""
        return sorted(self.full_df['condition'].unique())
    
    def recommend_drugs(self, condition: str, top_n: int = 5, alpha: float = 0.5, 
                        beta: float = 0.3, gamma: float = 0.2) -> List[Dict]:
        """
        Recommend drugs for a given condition
        
        Args:
            condition: Medical condition
            top_n: Number of top recommendations to return
            alpha: Weight for effectiveness probability
            beta: Weight for sentiment score
            gamma: Weight for useful count
        
        Returns:
            List of drug recommendations with scores
        """
        # Check if condition exists
        if condition not in self.condition_drugs:
            return []
        
        # Get all drugs for this condition
        condition_df = self.full_df[self.full_df['condition'] == condition].copy()
        
        if len(condition_df) == 0:
            return []
        
        # Aggregate reviews by drug
        drug_stats = []
        
        for drug_name in condition_df['drugName'].unique():
            drug_reviews = condition_df[condition_df['drugName'] == drug_name]
            
            # Calculate average sentiment
            avg_sentiment = drug_reviews['sentiment_score'].mean()
            
            # Calculate average rating
            avg_rating = drug_reviews['rating'].mean()
            
            # Calculate total useful count (with log dampening)
            total_useful = drug_reviews['usefulCount'].sum()
            log_useful = np.log1p(total_useful)  # log(1 + x) to handle zeros
            
            # Get a sample review for prediction
            sample_review = drug_reviews.iloc[0]
            
            # Prepare features for prediction
            try:
                # Create feature vector
                features = pd.DataFrame({
                    'drugName_encoded': [self.feature_engineer.drug_encoder.transform([drug_name])[0]],
                    'condition_encoded': [self.feature_engineer.condition_encoder.transform([condition])[0]],
                    'usefulCount': [total_useful],
                    'day': [sample_review['day']],
                    'month': [sample_review['month']],
                    'year': [sample_review['year']],
                    'sentiment_score': [avg_sentiment]
                })
                
                # Add TF-IDF features (use zeros as placeholder)
                tfidf_cols = [col for col in self.feature_engineer.tfidf_vectorizer.get_feature_names_out()]
                for i, col in enumerate(tfidf_cols):
                    features[f'tfidf_{i}'] = [0.0]
                
                # Predict effectiveness probability
                effectiveness_prob = self.model.predict_proba(features)[0][1]
            except Exception as e:
                print(f"Prediction error for {drug_name}: {e}")
                effectiveness_prob = 0.5  # Default to neutral
            
            # Normalize sentiment to 0-1 range (from -1 to 1)
            sentiment_norm = (avg_sentiment + 1) / 2
            
            # Normalize useful count
            max_useful = condition_df['usefulCount'].sum()
            useful_norm = log_useful / (np.log1p(max_useful) + 1e-10)
            
            # Calculate composite score
            score = (alpha * effectiveness_prob + 
                    beta * sentiment_norm + 
                    gamma * useful_norm)
            
            drug_stats.append({
                'drug_name': drug_name,
                'score': score,
                'effectiveness_prob': effectiveness_prob,
                'avg_sentiment': avg_sentiment,
                'avg_rating': avg_rating,
                'review_count': len(drug_reviews),
                'total_useful_count': int(total_useful)
            })
        
        # Sort by score and return top N
        drug_stats.sort(key=lambda x: x['score'], reverse=True)
        return drug_stats[:top_n]
    
    def get_drug_details(self, drug_name: str, condition: str) -> Dict:
        """Get detailed information about a drug"""
        drug_reviews = self.full_df[
            (self.full_df['drugName'] == drug_name) & 
            (self.full_df['condition'] == condition)
        ]
        
        if len(drug_reviews) == 0:
            return None
        
        # Sample reviews
        sample_reviews = drug_reviews.nlargest(3, 'usefulCount')[['review', 'rating', 'usefulCount']].to_dict('records')
        
        return {
            'drug_name': drug_name,
            'condition': condition,
            'avg_rating': float(drug_reviews['rating'].mean()),
            'total_reviews': len(drug_reviews),
            'avg_sentiment': float(drug_reviews['sentiment_score'].mean()),
            'sample_reviews': sample_reviews
        }
    
    def save(self, path: str = 'models/recommendation_engine.pkl'):
        """Save recommendation engine"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str = 'models/recommendation_engine.pkl'):
        """Load recommendation engine"""
        with open(path, 'rb') as f:
            return pickle.load(f)
        