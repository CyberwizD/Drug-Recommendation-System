"""Analysis page for Drug Recommendation System"""
import reflex as rx
from ..state import State
from ..components import navigation_bar


def analysis_page() -> rx.Component:
    """Interactive dashboard for model analysis"""
    return rx.box(
        navigation_bar(),
        
        rx.container(
            rx.vstack(
                # Header
                rx.vstack(
                    rx.hstack(
                        rx.icon("bar-chart", size=36, color="#667eea"),
                        rx.heading(
                            "Model Analysis & Performance",
                            size="8",
                            weight="bold",
                            style={
                                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                "-webkit-background-clip": "text",
                                "-webkit-text-fill-color": "transparent",
                                "background-clip": "text"
                            }
                        ),
                        spacing="3",
                        align="center"
                    ),
                    rx.text(
                        "Comprehensive machine learning model evaluation and performance metrics",
                        size="3",
                        color="gray",
                        text_align="center"
                    ),
                    spacing="3",
                    align="center",
                    padding_y="2rem"
                ),
                
                # Model performance comparison
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("trending-up", size=24, color="#667eea"),
                            rx.heading("Model Performance Comparison", size="5", weight="bold"),
                            spacing="2",
                            align="center"
                        ),
                        rx.text(
                            "Comparing 4 machine learning algorithms: Naive Bayes, Random Forest, Perceptron, and LightGBM (best performer)",
                            size="2",
                            color="gray",
                            line_height="1.6"
                        ),
                        rx.divider(margin_y="1rem"),
                        rx.image(
                            src=State.performance_chart,
                            width="100%",
                            height="auto",
                            border_radius="0.5rem",
                            style={
                                "box-shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
                            }
                        ),
                        width="100%",
                        spacing="3",
                        align="start"
                    ),
                    style={
                        "background": "white",
                        "border": "1px solid #e5e7eb",
                        "border-radius": "1rem",
                        "padding": "2rem",
                        "box-shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
                    }
                ),
                
                # Confusion matrix
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("grid", size=24, color="#667eea"),
                            rx.heading("Confusion Matrix - Best Model", size="5", weight="bold"),
                            spacing="2",
                            align="center"
                        ),
                        rx.text(
                            "Visual representation of model predictions vs actual labels. Shows True Positives, True Negatives, False Positives, and False Negatives",
                            size="2",
                            color="gray",
                            line_height="1.6"
                        ),
                        rx.divider(margin_y="1rem"),
                        rx.image(
                            src=State.confusion_matrix_chart,
                            width="100%",
                            height="auto",
                            border_radius="0.5rem",
                            style={
                                "box-shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
                            }
                        ),
                        width="100%",
                        spacing="3",
                        align="start"
                    ),
                    style={
                        "background": "white",
                        "border": "1px solid #e5e7eb",
                        "border-radius": "1rem",
                        "padding": "2rem",
                        "box-shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
                    }
                ),
                
                # Detailed metrics table
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("table", size=24, color="#667eea"),
                            rx.heading("Detailed Performance Metrics", size="5", weight="bold"),
                            spacing="2",
                            align="center"
                        ),
                        rx.text(
                            "Complete breakdown of all evaluation metrics for each model",
                            size="2",
                            color="gray",
                            line_height="1.6"
                        ),
                        rx.divider(margin_y="1rem"),
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
                                            rx.cond(
                                                item[0] == "LightGBM",
                                                rx.badge(
                                                    item[0],
                                                    color_scheme="blue",
                                                    size="2",
                                                    style={
                                                        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                                        "color": "white"
                                                    }
                                                ),
                                                rx.badge(
                                                    item[0],
                                                    color_scheme="gray",
                                                    size="2"
                                                )
                                            )
                                        ),
                                        rx.table.cell(f"{item[1].accuracy:.4f}"),
                                        rx.table.cell(f"{item[1].precision:.4f}"),
                                        rx.table.cell(f"{item[1].recall:.4f}"),
                                        rx.table.cell(f"{item[1].f1_score:.4f}"),
                                    )
                                )
                            ),
                            width="100%",
                            variant="surface",
                            style={
                                "border-radius": "0.5rem",
                                "overflow": "hidden"
                            }
                        ),
                        width="100%",
                        spacing="3",
                        align="start"
                    ),
                    style={
                        "background": "white",
                        "border": "1px solid #e5e7eb",
                        "border-radius": "1rem",
                        "padding": "2rem",
                        "box-shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
                    }
                ),
                
                # Key insights
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("lightbulb", size=24, color="#f59e0b"),
                            rx.heading("Key Insights", size="5", weight="bold"),
                            spacing="2",
                            align="center"
                        ),
                        rx.divider(margin_y="1rem"),
                        rx.vstack(
                            rx.hstack(
                                rx.icon("check-circle", size=20, color="#10b981"),
                                rx.text(
                                    "LightGBM achieved the highest accuracy at 92.4% with F1-score of 0.92",
                                    size="3",
                                    line_height="1.6"
                                ),
                                spacing="2",
                                align="start",
                                width="100%"
                            ),
                            rx.hstack(
                                rx.icon("check-circle", size=20, color="#10b981"),
                                rx.text(
                                    "Random Forest performed well with 88.2% accuracy, showing robustness",
                                    size="3",
                                    line_height="1.6"
                                ),
                                spacing="2",
                                align="start",
                                width="100%"
                            ),
                            rx.hstack(
                                rx.icon("check-circle", size=20, color="#10b981"),
                                rx.text(
                                    "Sentiment analysis integration significantly improved recommendation quality",
                                    size="3",
                                    line_height="1.6"
                                ),
                                spacing="2",
                                align="start",
                                width="100%"
                            ),
                            rx.hstack(
                                rx.icon("check-circle", size=20, color="#10b981"),
                                rx.text(
                                    "TF-IDF vectorization with 3000 features captured text semantics effectively",
                                    size="3",
                                    line_height="1.6"
                                ),
                                spacing="2",
                                align="start",
                                width="100%"
                            ),
                            spacing="3",
                            width="100%"
                        ),
                        width="100%",
                        spacing="3",
                        align="start"
                    ),
                    style={
                        "background": "linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%)",
                        "border": "1px solid #e5e7eb",
                        "border-radius": "1rem",
                        "padding": "2rem"
                    }
                ),
                
                width="100%",
                spacing="4"
            ),
            max_width="1200px",
            padding_y="2rem"
        ),
        
        width="100%",
        min_height="100vh",
        background="linear-gradient(to bottom, #f9fafb 0%, #ffffff 100%)"
    )
