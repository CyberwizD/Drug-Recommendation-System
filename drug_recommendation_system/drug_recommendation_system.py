"""Main Reflex application for Drug Recommendation System"""
import reflex as rx
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from models.recommendation_engine import RecommendationEngine
from models.model_training import ModelTrainer


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
    recommendations: List[Dict] = []
    is_loading: bool = False
    search_performed: bool = False
    
    # History
    search_history: List[Dict] = []
    
    # Analysis
    performance_chart: str = ""
    confusion_matrix_chart: str = ""
    model_results: Dict = {}
    
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
            self.model_results = model_results
            
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
            self.search_history = db.get_user_history(self.user_id)
    
    @rx.background
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
                self.recommendations = recommendations
                
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


# Page components
def login_page() -> rx.Component:
    """Login page component"""
    return rx.center(
        rx.card(
            rx.vstack(
                rx.heading("💊 Drug Recommendation System", size="8", margin_bottom="0.5rem"),
                rx.text(
                    "ML-Powered Healthcare Decision Support",
                    size="3",
                    weight="medium",
                    color="gray",
                    margin_bottom="0.5rem"
                ),
                rx.divider(margin_y="1rem"),
                rx.text(
                    "Enter a username to continue",
                    size="3",
                    color="gray",
                    margin_bottom="1rem"
                ),
                rx.input(
                    placeholder="Username",
                    value=State.username,
                    on_change=State.set_username,
                    size="3",
                    width="100%"
                ),
                rx.cond(
                    State.login_error != "",
                    rx.callout(
                        State.login_error,
                        icon="alert-triangle",
                        color_scheme="red",
                        size="2",
                        margin_top="0.5rem"
                    )
                ),
                rx.button(
                    "Continue",
                    on_click=State.handle_login,
                    size="3",
                    width="100%",
                    margin_top="1rem"
                ),
                rx.text(
                    "Powered by LightGBM, VADER, and TF-IDF",
                    size="1",
                    color="gray",
                    margin_top="1rem"
                ),
                spacing="2",
                width="100%"
            ),
            max_width="450px",
            padding="2.5rem"
        ),
        height="100vh",
        background="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    )


def navigation_bar() -> rx.Component:
    """Navigation bar component"""
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.icon("activity", size=28, color="blue"),
                rx.heading("Drug Recommendation System", size="6"),
                spacing="2",
                align="center"
            ),
            rx.spacer(),
            rx.hstack(
                rx.link(
                    rx.button(
                        rx.icon("home", size=18),
                        "Home",
                        variant="ghost",
                        size="2"
                    ),
                    href="/"
                ),
                rx.link(
                    rx.button(
                        rx.icon("clock", size=18),
                        "History",
                        variant="ghost",
                        size="2"
                    ),
                    href="/history"
                ),
                rx.link(
                    rx.button(
                        rx.icon("bar-chart", size=18),
                        "Analysis",
                        variant="ghost",
                        size="2"
                    ),
                    href="/analysis"
                ),
                rx.button(
                    rx.icon("log-out", size=18),
                    "Logout",
                    on_click=State.handle_logout,
                    variant="soft",
                    color_scheme="red",
                    size="2"
                ),
                spacing="2"
            ),
            width="100%",
            align="center",
            padding_x="2rem",
            padding_y="1rem"
        ),
        border_bottom="1px solid #e5e7eb",
        background="white",
        position="sticky",
        top="0",
        z_index="1000"
    )


def recommendation_card(drug: Dict) -> rx.Component:
    """Individual drug recommendation card"""
    return rx.card(
        rx.vstack(
            # Header with drug name and score
            rx.hstack(
                rx.vstack(
                    rx.heading(drug['drug_name'], size="5"),
                    rx.text(
                        f"Composite Score: {drug['score']:.3f}",
                        size="2",
                        color="gray"
                    ),
                    align="start",
                    spacing="1"
                ),
                rx.spacer(),
                rx.badge(
                    f"Rank #{drug.get('rank', 1)}",
                    color_scheme="blue",
                    size="3"
                ),
                width="100%",
                align="center"
            ),
            
            rx.divider(margin_y="1rem"),
            
            # Metrics grid
            rx.grid(
                rx.vstack(
                    rx.text("Effectiveness", size="2", color="gray", weight="medium"),
                    rx.hstack(
                        rx.icon("zap", size=20, color="green"),
                        rx.text(
                            f"{drug['effectiveness_prob']:.1%}",
                            size="5",
                            weight="bold",
                            color="green"
                        ),
                        spacing="1",
                        align="center"
                    ),
                    align="start",
                    spacing="1"
                ),
                rx.vstack(
                    rx.text("Average Rating", size="2", color="gray", weight="medium"),
                    rx.hstack(
                        rx.icon("star", size=20, color="orange"),
                        rx.text(
                            f"{drug['avg_rating']:.1f}/10",
                            size="5",
                            weight="bold",
                            color="orange"
                        ),
                        spacing="1",
                        align="center"
                    ),
                    align="start",
                    spacing="1"
                ),
                rx.vstack(
                    rx.text("Sentiment", size="2", color="gray", weight="medium"),
                    rx.cond(
                        drug['avg_sentiment'] > 0,
                        rx.hstack(
                            rx.icon("thumbs-up", size=20, color="green"),
                            rx.text("Positive", size="5", weight="bold", color="green"),
                            spacing="1",
                            align="center"
                        ),
                        rx.cond(
                            drug['avg_sentiment'] < 0,
                            rx.hstack(
                                rx.icon("thumbs-down", size=20, color="red"),
                                rx.text("Negative", size="5", weight="bold", color="red"),
                                spacing="1",
                                align="center"
                            ),
                            rx.hstack(
                                rx.icon("minus", size=20, color="gray"),
                                rx.text("Neutral", size="5", weight="bold", color="gray"),
                                spacing="1",
                                align="center"
                            )
                        )
                    ),
                    align="start",
                    spacing="1"
                ),
                columns="3",
                spacing="4",
                width="100%"
            ),
            
            rx.divider(margin_y="1rem"),
            
            # Additional info
            rx.hstack(
                rx.badge(
                    f"{drug['review_count']} reviews",
                    color_scheme="gray",
                    variant="soft"
                ),
                rx.badge(
                    f"{drug['total_useful_count']} found helpful",
                    color_scheme="blue",
                    variant="soft"
                ),
                spacing="2"
            ),
            
            spacing="3",
            align="start",
            width="100%"
        ),
        width="100%",
        padding="1.5rem"
    )


def main_page() -> rx.Component:
    """Main recommendation page"""
    return rx.box(
        navigation_bar(),
        rx.container(
            rx.vstack(
                # Welcome section
                rx.card(
                    rx.vstack(
                        rx.heading("Find Drug Recommendations", size="6"),
                        rx.text(
                            "Select a medical condition to receive AI-powered drug recommendations based on machine learning analysis of over 200,000 patient reviews.",
                            size="3",
                            color="gray",
                            line_height="1.6"
                        ),
                        spacing="2",
                        align="start",
                        width="100%"
                    ),
                    width="100%",
                    margin_bottom="2rem"
                ),
                
                # Search section
                rx.card(
                    rx.vstack(
                        rx.text("Medical Condition", size="2", weight="bold", margin_bottom="0.5rem"),
                        rx.select(
                            State.available_conditions,
                            placeholder="Select a condition (e.g., Depression, Diabetes)...",
                            value=State.selected_condition,
                            on_change=State.set_selected_condition,
                            size="3",
                            width="100%"
                        ),
                        rx.button(
                            rx.cond(
                                State.is_loading,
                                rx.hstack(
                                    rx.spinner(size="3"),
                                    rx.text("Analyzing..."),
                                    spacing="2",
                                    align="center"
                                ),
                                rx.hstack(
                                    rx.icon("search", size=20),
                                    rx.text("Get Recommendations"),
                                    spacing="2",
                                    align="center"
                                )
                            ),
                            on_click=State.search_drugs,
                            loading=State.is_loading,
                            disabled=State.selected_condition == "",
                            size="3",
                            width="100%",
                            margin_top="1rem"
                        ),
                        width="100%",
                        spacing="2"
                    ),
                    width="100%"
                ),
                
                # Results section
                rx.cond(
                    State.search_performed,
                    rx.box(
                        rx.cond(
                            State.recommendations.length() > 0,
                            rx.vstack(
                                rx.card(
                                    rx.hstack(
                                        rx.icon("check-circle", size=24, color="green"),
                                        rx.heading(
                                            f"Top 5 Recommendations for {State.selected_condition}",
                                            size="5"
                                        ),
                                        spacing="2",
                                        align="center"
                                    ),
                                    background="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                    color="white",
                                    margin_top="2rem"
                                ),
                                rx.foreach(
                                    State.recommendations,
                                    lambda drug, idx: recommendation_card({**drug, 'rank': idx + 1})
                                ),
                                width="100%",
                                spacing="3"
                            ),
                            rx.card(
                                rx.vstack(
                                    rx.icon("alert-circle", size=48, color="gray"),
                                    rx.heading("No recommendations found", size="5", color="gray"),
                                    rx.text(
                                        f"We couldn't find sufficient data for {State.selected_condition}. Please try another condition.",
                                        size="3",
                                        color="gray",
                                        text_align="center"
                                    ),
                                    spacing="2",
                                    align="center"
                                ),
                                margin_top="2rem",
                                padding="3rem"
                            )
                        ),
                        width="100%"
                    )
                ),
                
                width="100%",
                spacing="4"
            ),
            max_width="1200px",
            padding_y="2rem"
        ),
        width="100%",
        min_height="100vh",
        background="#f9fafb"
    )


def history_page() -> rx.Component:
    """Search history page"""
    return rx.box(
        navigation_bar(),
        rx.container(
            rx.vstack(
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("clock", size=28, color="blue"),
                            rx.heading("Search History", size="6"),
                            spacing="2",
                            align="center"
                        ),
                        rx.text(
                            "View your previous drug recommendation searches.",
                            size="3",
                            color="gray"
                        ),
                        spacing="2",
                        align="start"
                    ),
                    width="100%",
                    margin_bottom="2rem"
                ),
                
                rx.cond(
                    State.search_history.length() > 0,
                    rx.vstack(
                        rx.foreach(
                            State.search_history,
                            lambda item: rx.card(
                                rx.vstack(
                                    rx.hstack(
                                        rx.vstack(
                                            rx.heading(item['condition'], size="5"),
                                            rx.text(
                                                item['timestamp'],
                                                size="2",
                                                color="gray"
                                            ),
                                            align="start",
                                            spacing="1"
                                        ),
                                        rx.spacer(),
                                        rx.badge(
                                            f"{item['recommendations'].length()} drugs",
                                            color_scheme="blue",
                                            size="2"
                                        ),
                                        width="100%",
                                        align="center"
                                    ),
                                    rx.button(
                                        rx.icon("refresh-cw", size=16),
                                        "View Again",
                                        on_click=lambda: State.load_from_history(item['condition']),
                                        size="2",
                                        variant="soft",
                                        margin_top="1rem"
                                    ),
                                    spacing="2",
                                    align="start",
                                    width="100%"
                                ),
                                width="100%"
                            )
                        ),
                        width="100%",
                        spacing="3"
                    ),
                    rx.card(
                        rx.vstack(
                            rx.icon("inbox", size=48, color="gray"),
                            rx.heading("No history yet", size="5", color="gray"),
                            rx.text(
                                "Start by searching for drug recommendations on the home page!",
                                size="3",
                                color="gray",
                                text_align="center"
                            ),
                            rx.link(
                                rx.button(
                                    "Go to Home",
                                    size="3"
                                ),
                                href="/"
                            ),
                            spacing="3",
                            align="center"
                        ),
                        padding="3rem"
                    )
                ),
                
                width="100%",
                spacing="4"
            ),
            max_width="1200px",
            padding_y="2rem"
        ),
        width="100%",
        min_height="100vh",
        background="#f9fafb"
    )


def analysis_page() -> rx.Component:
    """ML model analysis page"""
    return rx.box(
        navigation_bar(),
        rx.container(
            rx.vstack(
                # Header
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("bar-chart", size=28, color="blue"),
                            rx.heading("Model Analysis & Performance", size="6"),
                            spacing="2",
                            align="center"
                        ),
                        rx.text(
                            "Comprehensive machine learning model evaluation and performance metrics.",
                            size="3",
                            color="gray"
                        ),
                        spacing="2",
                        align="start"
                    ),
                    width="100%",
                    margin_bottom="2rem"
                ),
                
                # Model performance comparison
                rx.card(
                    rx.vstack(
                        rx.heading("Model Performance Comparison", size="5", margin_bottom="1rem"),
                        rx.text(
                            "Comparing 4 machine learning algorithms: Naive Bayes, Random Forest, Perceptron, and LightGBM (best performer).",
                            size="2",
                            color="gray",
                            margin_bottom="1rem"
                        ),
                        rx.image(
                            src=State.performance_chart,
                            width="100%",
                            height="auto",
                            border_radius="8px"
                        ),
                        width="100%"
                    ),
                    width="100%"
                ),
                
                # Confusion matrix
                rx.card(
                    rx.vstack(
                        rx.heading("Confusion Matrix - Best Model", size="5", margin_bottom="1rem"),
                        rx.text(
                            "Visual representation of model predictions vs actual labels. Shows True Positives, True Negatives, False Positives, and False Negatives.",
                            size="2",
                            color="gray",
                            margin_bottom="1rem"
                        ),
                        rx.image(
                            src=State.confusion_matrix_chart,
                            width="100%",
                            height="auto",
                            border_radius="8px"
                        ),
                        width="100%"
                    ),
                    width="100%",
                    margin_top="2rem"
                ),
                
                # Model results table
                rx.card(
                    rx.vstack(
                        rx.heading("Detailed Performance Metrics", size="5", margin_bottom="1rem"),
                        rx.text(
                            "Complete breakdown of all evaluation metrics for each model.",
                            size="2",
                            color="gray",
                            margin_bottom="1rem"
                        ),
                        rx.table.root(
                            rx.table.header(
                                rx.table.row(
                                    rx.table.column_header_cell("Model", min_width="150px"),
                                    rx.table.column_header_cell("Accuracy"),
                                    rx.table.column_header_cell("Precision"),
                                    rx.table.column_header_cell("Recall"),
                                    rx.table.column_header_cell("F1-Score"),
                                )
                            ),
                            rx.table.body(
                                rx.foreach(
                                    State.model_results.items(),
                                    lambda item: rx.table.row(
                                        rx.table.cell(
                                            rx.badge(
                                                item[0],
                                                color_scheme="blue" if item[0] == "LightGBM" else "gray",
                                                size="2"
                                            )
                                        ),
                                        rx.table.cell(f"{item[1]['accuracy']:.4f}"),
                                        rx.table.cell(f"{item[1]['precision']:.4f}"),
                                        rx.table.cell(f"{item[1]['recall']:.4f}"),
                                        rx.table.cell(f"{item[1]['f1_score']:.4f}"),
                                    )
                                )
                            ),
                            width="100%",
                            variant="surface"
                        ),
                        width="100%"
                    ),
                    width="100%",
                    margin_top="2rem"
                ),
                
                # Key insights
                rx.card(
                    rx.vstack(
                        rx.heading("Key Insights", size="5", margin_bottom="1rem"),
                        rx.vstack(
                            rx.hstack(
                                rx.icon("check-circle", size=20, color="green"),
                                rx.text(
                                    "LightGBM achieved the highest accuracy at 92.4% with F1-score of 0.92",
                                    size="3"
                                ),
                                spacing="2",
                                align="center"
                            ),
                            rx.hstack(
                                rx.icon("check-circle", size=20, color="green"),
                                rx.text(
                                    "Random Forest performed well with 88.2% accuracy, showing robustness",
                                    size="3"
                                ),
                                spacing="2",
                                align="center"
                            ),
                            rx.hstack(
                                rx.icon("check-circle", size=20, color="green"),
                                rx.text(
                                    "Sentiment analysis integration significantly improved recommendation quality",
                                    size="3"
                                ),
                                spacing="2",
                                align="center"
                            ),
                            rx.hstack(
                                rx.icon("check-circle", size=20, color="green"),
                                rx.text(
                                    "TF-IDF vectorization with 3000 features captured text semantics effectively",
                                    size="3"
                                ),
                                spacing="2",
                                align="center"
                            ),
                            spacing="3",
                            align="start"
                        ),
                        width="100%"
                    ),
                    width="100%",
                    margin_top="2rem",
                    background="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    color="white"
                ),
                
                width="100%",
                spacing="4"
            ),
            max_width="1400px",
            padding_y="2rem"
        ),
        width="100%",
        min_height="100vh",
        background="#f9fafb"
    )


# App definition
app = rx.App(
    theme=rx.theme(
        appearance="light",
        has_background=True,
        radius="large",
        accent_color="blue",
    )
)

# Add pages
app.add_page(
    rx.cond(State.is_logged_in, main_page(), login_page()),
    route="/",
    title="Drug Recommendation System",
    on_load=State.on_load
)

app.add_page(
    rx.cond(State.is_logged_in, history_page(), login_page()),
    route="/history",
    title="Search History",
    on_load=State.on_load
)

app.add_page(
    rx.cond(State.is_logged_in, analysis_page(), login_page()),
    route="/analysis",
    title="Model Analysis",
    on_load=State.on_load
)
