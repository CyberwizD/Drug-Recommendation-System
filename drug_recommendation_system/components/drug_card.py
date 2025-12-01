"""Drug card component for displaying recommendations"""
import reflex as rx
from ..state import DrugRecommendation


def drug_card(drug: DrugRecommendation, rank: int) -> rx.Component:
    """Beautiful drug recommendation card with animations"""
    return rx.card(
        rx.vstack(
            # Header with rank badge
            rx.hstack(
                rx.badge(
                    f"#{rank}",
                    size="3",
                    style={
                        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                        "color": "white",
                        "font-weight": "bold",
                        "padding": "0.5rem 1rem",
                        "border-radius": "9999px"
                    }
                ),
                rx.spacer(),
                rx.badge(
                    f"Score: {drug.score:.3f}",
                    size="2",
                    color_scheme="blue",
                    variant="soft"
                ),
                width="100%",
                align="center",
                margin_bottom="0.5rem"
            ),
            
            # Drug name
            rx.heading(
                drug.drug_name,
                size="6",
                weight="bold",
                style={
                    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    "-webkit-background-clip": "text",
                    "-webkit-text-fill-color": "transparent",
                    "background-clip": "text"
                }
            ),
            
            rx.divider(margin_y="1rem"),
            
            # Metrics grid
            rx.grid(
                # Effectiveness
                rx.vstack(
                    rx.hstack(
                        rx.icon("zap", size=24, color="#10b981"),
                        rx.vstack(
                            rx.text("Effectiveness", size="1", color="gray", weight="medium"),
                            rx.text(
                                f"{drug.effectiveness_prob * 100:.1f}%",
                                size="5",
                                weight="bold",
                                color="#10b981"
                            ),
                            spacing="0",
                            align="start"
                        ),
                        spacing="2",
                        align="center"
                    ),
                    align="start",
                    spacing="1"
                ),
                
                # Rating
                rx.vstack(
                    rx.hstack(
                        rx.icon("star", size=24, color="#f59e0b"),
                        rx.vstack(
                            rx.text("Avg Rating", size="1", color="gray", weight="medium"),
                            rx.text(
                                f"{drug.avg_rating:.1f}/10",
                                size="5",
                                weight="bold",
                                color="#f59e0b"
                            ),
                            spacing="0",
                            align="start"
                        ),
                        spacing="2",
                        align="center"
                    ),
                    align="start",
                    spacing="1"
                ),
                
                # Sentiment
                rx.vstack(
                    rx.cond(
                        drug.avg_sentiment.to(float) > 0,
                        rx.hstack(
                            rx.icon("thumbs-up", size=24, color="#10b981"),
                            rx.vstack(
                                rx.text("Sentiment", size="1", color="gray", weight="medium"),
                                rx.text("Positive", size="5", weight="bold", color="#10b981"),
                                spacing="0",
                                align="start"
                            ),
                            spacing="2",
                            align="center"
                        ),
                        rx.cond(
                            drug.avg_sentiment.to(float) < 0,
                            rx.hstack(
                                rx.icon("thumbs-down", size=24, color="#ef4444"),
                                rx.vstack(
                                    rx.text("Sentiment", size="1", color="gray", weight="medium"),
                                    rx.text("Negative", size="5", weight="bold", color="#ef4444"),
                                    spacing="0",
                                    align="start"
                                ),
                                spacing="2",
                                align="center"
                            ),
                            rx.hstack(
                                rx.icon("minus", size=24, color="#6b7280"),
                                rx.vstack(
                                    rx.text("Sentiment", size="1", color="gray", weight="medium"),
                                    rx.text("Neutral", size="5", weight="bold", color="#6b7280"),
                                    spacing="0",
                                    align="start"
                                ),
                                spacing="2",
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
                    rx.icon("message-circle", size=14),
                    f"{drug.review_count} reviews",
                    color_scheme="gray",
                    variant="soft",
                    size="2"
                ),
                rx.badge(
                    rx.icon("heart", size=14),
                    f"{drug.total_useful_count} helpful",
                    color_scheme="blue",
                    variant="soft",
                    size="2"
                ),
                spacing="2",
                wrap="wrap"
            ),
            
            spacing="3",
            align="start",
            width="100%"
        ),
        width="100%",
        style={
            "background": "white",
            "border": "1px solid #e5e7eb",
            "border-radius": "1rem",
            "padding": "1.5rem",
            "transition": "all 0.3s ease",
            "_hover": {
                "transform": "translateY(-4px)",
                "box-shadow": "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
                "border-color": "#667eea"
            }
        }
    )
