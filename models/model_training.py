"""Model training and evaluation module"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        print("\n" + "="*50)
        print("Training Models")
        print("="*50)
        
        # Ensure all features are non-negative for Naive Bayes
        X_train_nb = X_train.copy()
        X_test_nb = X_test.copy()
        X_train_nb[X_train_nb < 0] = 0
        X_test_nb[X_test_nb < 0] = 0
        
        # Define models
        models_to_train = {
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
            'Perceptron': Perceptron(max_iter=1000, random_state=42),
            'LightGBM': LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbosity=-1
            )
        }
        
        best_accuracy = 0
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            try:
                # Use non-negative data for Naive Bayes
                if name == 'Naive Bayes':
                    model.fit(X_train_nb, y_train)
                    y_pred = model.predict(X_test_nb)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                
                # Store results
                self.models[name] = model
                self.results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm
                }
                
                print(f"{name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
                
                # Track best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.best_model = model
                    self.best_model_name = name
            
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
        
        print(f"\nBest Model: {self.best_model_name} with Accuracy: {best_accuracy:.4f}")
        return self.results
    
    def get_performance_comparison_chart(self):
        """Generate performance comparison chart as base64"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def get_confusion_matrix_chart(self, model_name=None):
        """Generate confusion matrix chart as base64"""
        if model_name is None:
            model_name = self.best_model_name
        
        cm = self.results[model_name]['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def save_best_model(self, path: str = 'models/best_model.pkl'):
        """Save the best performing model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'results': self.results
            }, f)
        print(f"Best model ({self.best_model_name}) saved to {path}")
    
    @staticmethod
    def load_best_model(path: str = 'models/best_model.pkl'):
        """Load the saved model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['model_name'], data['results']
    