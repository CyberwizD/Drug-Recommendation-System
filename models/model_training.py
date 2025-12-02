"""Model training and evaluation module"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import learning_curve
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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        print("\n" + "="*50)
        print("Training Models")
        print("="*50)
        
        # Store data for later use in visualizations
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
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
                    y_pred_proba = model.predict_proba(X_test_nb)[:, 1] if hasattr(model, 'predict_proba') else None
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
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
                    'confusion_matrix': cm,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
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
        
        colors = ['#667eea', '#764ba2', '#10b981', '#f59e0b']
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def get_roc_curves_chart(self):
        """Generate ROC curves for all models"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#667eea', '#764ba2', '#10b981', '#f59e0b']
        
        for i, (name, result) in enumerate(self.results.items()):
            if result['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
                       label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def get_precision_recall_chart(self):
        """Generate Precision-Recall curves for all models"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#667eea', '#764ba2', '#10b981', '#f59e0b']
        
        for i, (name, result) in enumerate(self.results.items()):
            if result['y_pred_proba'] is not None:
                precision, recall, _ = precision_recall_curve(self.y_test, result['y_pred_proba'])
                avg_precision = average_precision_score(self.y_test, result['y_pred_proba'])
                ax.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                       label=f'{name} (AP = {avg_precision:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def get_feature_importance_chart(self):
        """Generate feature importance chart for tree-based models"""
        # Get feature importance from Random Forest or LightGBM
        model_name = 'LightGBM' if 'LightGBM' in self.models else 'Random Forest'
        
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Get top 20 features
            indices = np.argsort(importances)[::-1][:20]
            top_importances = importances[indices]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_importances)))
            ax.barh(range(len(top_importances)), top_importances, color=colors)
            ax.set_yticks(range(len(top_importances)))
            ax.set_yticklabels([f'Feature {i}' for i in indices])
            ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Top 20 Feature Importances - {model_name}', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
        
        return None
    
    def get_metrics_radar_chart(self):
        """Generate radar chart comparing all models across metrics"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['#667eea', '#764ba2', '#10b981', '#f59e0b']
        
        for i, (name, result) in enumerate(self.results.items()):
            values = [result[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def get_class_distribution_chart(self):
        """Generate class distribution chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Training set distribution
        train_counts = pd.Series(self.y_train).value_counts().sort_index()
        colors_train = ['#ef4444', '#10b981']
        ax1.pie(train_counts, labels=['Ineffective', 'Effective'], autopct='%1.1f%%',
               colors=colors_train, startangle=90)
        ax1.set_title('Training Set Class Distribution', fontsize=12, fontweight='bold')
        
        # Test set distribution
        test_counts = pd.Series(self.y_test).value_counts().sort_index()
        ax2.pie(test_counts, labels=['Ineffective', 'Effective'], autopct='%1.1f%%',
               colors=colors_train, startangle=90)
        ax2.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def get_roc_curves_chart(self):
        """Generate ROC curves for all models"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#667eea', '#764ba2', '#10b981', '#f59e0b']
        
        for i, (name, result) in enumerate(self.results.items()):
            if result['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
                       label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def get_precision_recall_chart(self):
        """Generate Precision-Recall curves for all models"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#667eea', '#764ba2', '#10b981', '#f59e0b']
        
        for i, (name, result) in enumerate(self.results.items()):
            if result['y_pred_proba'] is not None:
                precision, recall, _ = precision_recall_curve(self.y_test, result['y_pred_proba'])
                avg_precision = average_precision_score(self.y_test, result['y_pred_proba'])
                ax.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                       label=f'{name} (AP = {avg_precision:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def get_feature_importance_chart(self):
        """Generate feature importance chart for tree-based models"""
        # Get feature importance from Random Forest or LightGBM
        model_name = 'LightGBM' if 'LightGBM' in self.models else 'Random Forest'
        
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Get top 20 features
            indices = np.argsort(importances)[::-1][:20]
            top_importances = importances[indices]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_importances)))
            ax.barh(range(len(top_importances)), top_importances, color=colors)
            ax.set_yticks(range(len(top_importances)))
            ax.set_yticklabels([f'Feature {i}' for i in indices])
            ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Top 20 Feature Importances - {model_name}', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
        
        return None
    
    def get_metrics_radar_chart(self):
        """Generate radar chart comparing all models across metrics"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['#667eea', '#764ba2', '#10b981', '#f59e0b']
        
        for i, (name, result) in enumerate(self.results.items()):
            values = [result[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def get_class_distribution_chart(self):
        """Generate class distribution chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Training set distribution
        train_counts = pd.Series(self.y_train).value_counts().sort_index()
        colors_train = ['#ef4444', '#10b981']
        ax1.pie(train_counts, labels=['Ineffective', 'Effective'], autopct='%1.1f%%',
               colors=colors_train, startangle=90)
        ax1.set_title('Training Set Class Distribution', fontsize=12, fontweight='bold')
        
        # Test set distribution
        test_counts = pd.Series(self.y_test).value_counts().sort_index()
        ax2.pie(test_counts, labels=['Ineffective', 'Effective'], autopct='%1.1f%%',
               colors=colors_train, startangle=90)
        ax2.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def save_best_model(self, path: str = 'models/best_model.pkl'):
        """Save the best performing model with all charts"""
        # Generate all charts before saving
        charts = {}
        
        try:
            charts['performance_chart'] = self.get_performance_comparison_chart()
            charts['confusion_matrix_chart'] = self.get_confusion_matrix_chart()
            charts['roc_curves_chart'] = self.get_roc_curves_chart()
            charts['precision_recall_chart'] = self.get_precision_recall_chart()
            charts['metrics_radar_chart'] = self.get_metrics_radar_chart()
            charts['feature_importance_chart'] = self.get_feature_importance_chart() or ""
            charts['class_distribution_chart'] = self.get_class_distribution_chart() or ""
        except Exception as e:
            print(f"Warning: Some charts could not be generated: {e}")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'results': self.results,
                'charts': charts  # Save all charts
            }, f)
        print(f"Best model ({self.best_model_name}) and charts saved to {path}")
    
    @staticmethod
    def load_best_model(path: str = 'models/best_model.pkl'):
        """Load the saved model and charts"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # Return charts if available, otherwise empty dict
        charts = data.get('charts', {})
        return data['model'], data['model_name'], data['results'], charts