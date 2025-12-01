"""State management for Drug Recommendation System"""
import reflex as rx
from typing import List, Dict, Optional
from pydantic import BaseModel
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from models.recommendation_engine import RecommendationEngine
from models.model_training import ModelTrainer


# Type classes for Reflex
class DrugRecommendation(BaseModel):
    """Type for drug recommendation data"""
    drug_name: str
    score: float
    effectiveness_prob: float
    avg_rating: float
    avg_sentiment: float
    review_count: int
    total_useful_count: int


class SearchHistoryItem(BaseModel):
    """Type for search history item"""
    condition: str
    timestamp: str
    recommendations: List[DrugRecommendation]


class ModelMetrics(BaseModel):
    """Type for model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class State(rx.State):
    """Main application state"""
    
    # User management
    username: str = ""
    user_id: Optional[int] = None
    is_logged_in: bool = False
    login_error: str = ""
    
    # Recommendation state
    selected_condition: str = ""
    available_conditions: List[str] = []
    recommendations: List[DrugRecommendation] = []
    is_loading: bool = False
    search_performed: bool = False
    
    # History
    search_history: List[SearchHistoryItem] = []
    
    # Analysis
    performance_chart: str = ""
    confusion_matrix_chart: str = ""
    model_results: Dict[str, ModelMetrics] = {}
    
    def on_load(self):
        """Initialize state when app loads"""
        if not self.is_logged_in:
            return
        
        # Load models if not already loaded
        if len(self.available_conditions) == 0:
            self.load_models()
    
    def load_models(self):
        """Load trained models and recommendation engine"""
        try:
            # Initialize database
            db = DatabaseManager()
            
            # Load recommendation engine
            recommendation_engine = RecommendationEngine.load('models/recommendation_engine.pkl')
            self.available_conditions = recommendation_engine.get_available_conditions()
            
            # Load model results for analysis
            _, _, model_results = ModelTrainer.load_best_model('models/best_model.pkl')
            # Convert dict to typed ModelMetrics objects
            self.model_results = {
                model_name: ModelMetrics(**metrics)
                for model_name, metrics in model_results.items()
            }
            
            # Generate charts
            trainer = ModelTrainer()
            trainer.results = model_results
            trainer.best_model_name = "LightGBM"
            self.performance_chart = trainer.get_performance_comparison_chart()
            self.confusion_matrix_chart = trainer.get_confusion_matrix_chart()
            
            print("✓ Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please run: python train_models.py")
    
    def handle_login(self):
        """Handle user login/creation"""
        if not self.username or len(self.username.strip()) == 0:
            self.login_error = "Please enter a username"
            return
        
        username = self.username.strip()
        
        # Initialize database
        db = DatabaseManager()
        
        # Try to get existing user
        user_id = db.get_user_id(username)
        
        if user_id is None:
            # Create new user
            user_id = db.create_user(username)
            if user_id is None:
                self.login_error = "Username already exists"
                return
        
        self.user_id = user_id
        self.is_logged_in = True
        self.login_error = ""
        
        # Load models after login
        self.load_models()
        self.load_history()
    
    def handle_logout(self):
        """Handle user logout"""
        self.username = ""
        self.user_id = None
        self.is_logged_in = False
        self.search_history = []
        self.recommendations = []
        self.search_performed = False
        self.selected_condition = ""
    
    def load_history(self):
        """Load user's search history"""
        if self.user_id:
            db = DatabaseManager()
            history_data = db.get_user_history(self.user_id)
            # Convert dict history to SearchHistoryItem objects
            self.search_history = [
                SearchHistoryItem(
                    condition=item['condition'],
                    timestamp=item['timestamp'],
                    recommendations=[
                        DrugRecommendation(**rec) for rec in item['recommendations']
                    ]
                )
                for item in history_data
            ]
    
    async def search_drugs(self):
        """Search for drug recommendations"""
        if not self.selected_condition:
            return
        
        async with self:
            self.is_loading = True
        
        try:
            # Load recommendation engine
            recommendation_engine = RecommendationEngine.load('models/recommendation_engine.pkl')
            
            # Get recommendations
            recommendations = recommendation_engine.recommend_drugs(
                self.selected_condition,
                top_n=5
            )
            
            async with self:
                # Convert dict recommendations to DrugRecommendation objects
                self.recommendations = [
                    DrugRecommendation(**rec) for rec in recommendations
                ]
                
                # Save to history
                if self.user_id and len(recommendations) > 0:
                    db = DatabaseManager()
                    db.add_search_history(
                        self.user_id,
                        self.selected_condition,
                        recommendations
                    )
                    self.load_history()
                
                self.search_performed = True
        except Exception as e:
            print(f"Error in search: {e}")
            async with self:
                self.recommendations = []
                self.search_performed = True
        finally:
            async with self:
                self.is_loading = False
    
    def load_from_history(self, condition: str):
        """Load a previous search from history"""
        self.selected_condition = condition
        return self.search_drugs()
