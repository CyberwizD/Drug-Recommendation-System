"""Home page for Drug Recommendation System"""
import reflex as rx
from ..state import State
from ..components import navigation_bar, drug_card, search_section, card_skeleton


def home_page() -> rx.Component:
    """Main recommendation page with enhanced UI"""
    return rx.box(
        navigation_bar(),
        
        rx.container(
            rx.vstack(
                # Hero section
                rx.vstack(
                    rx.heading(
                        f"Welcome back, {State.username}! 👋",
                        size="8",
                        weight="bold",
                        style={
                            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                            "-webkit-background-clip": "text",
                            "-webkit-text-fill-color": "transparent",
                            "background-clip": "text"
                        }
                    ),
                    rx.text(
                        "Discover the best medications for your health needs with AI-powered recommendations based on 200,000+ patient reviews",
                        size="4",
                        color="gray",
                        text_align="center",
                        line_height="1.6",
                        max_width="800px"
                    ),
                    spacing="3",
                    align="center",
                    padding_y="2rem"
                ),
                
                # Search section
                search_section(),
                
                # Results section
                rx.cond(
                    State.search_performed,
                    rx.box(
                        rx.cond(
                            State.recommendations.length() > 0,
                            rx.vstack(
                                # Results header
                                rx.card(
                                    rx.hstack(
                                        rx.icon("check-circle", size=28, color="white"),
                                        rx.heading(
                                            f"Top 5 Recommendations for {State.selected_condition}",
                                            size="6",
                                            color="white",
                                            weight="bold"
                                        ),
                                        spacing="3",
                                        align="center"
                                    ),
                                    style={
                                        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                        "border-radius": "1rem",
                                        "padding": "1.5rem",
                                        "box-shadow": "0 10px 15px -3px rgba(102, 126, 234, 0.3)"
                                    }
                                ),
                                
                                # Results grid
                                rx.box(
                                    rx.foreach(
                                        State.recommendations,
                                        lambda drug, idx: drug_card(drug, idx + 1)
                                    ),
                                    style={
                                        "display": "grid",
                                        "grid-template-columns": "repeat(auto-fill, minmax(350px, 1fr))",
                                        "gap": "1.5rem",
                                        "width": "100%"
                                    }
                                ),
                                
                                width="100%",
                                spacing="4"
                            ),
                            
                            # No results state
                            rx.card(
                                rx.vstack(
                                    rx.icon("alert-circle", size=64, color="#6b7280"),
                                    rx.heading("No Recommendations Found", size="6", color="gray"),
                                    rx.text(
                                        f"We couldn't find sufficient data for {State.selected_condition}. Please try another condition.",
                                        size="3",
                                        color="gray",
                                        text_align="center",
                                        max_width="500px"
                                    ),
                                    rx.button(
                                        rx.icon("refresh-cw", size=18),
                                        "Try Another Condition",
                                        on_click=lambda: State.set_search_performed(False),
                                        size="3",
                                        style={
                                            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                            "color": "white"
                                        }
                                    ),
                                    spacing="4",
                                    align="center",
                                    padding="3rem"
                                ),
                                style={
                                    "border": "2px dashed #e5e7eb",
                                    "border-radius": "1rem"
                                }
                            )
                        ),
                        width="100%",
                        margin_top="2rem"
                    )
                ),
                
                # Loading state
                rx.cond(
                    State.is_loading,
                    rx.vstack(
                        rx.heading("Analyzing...", size="5", color="gray"),
                        rx.box(
                            card_skeleton(),
                            card_skeleton(),
                            card_skeleton(),
                            style={
                                "display": "grid",
                                "grid-template-columns": "repeat(auto-fill, minmax(350px, 1fr))",
                                "gap": "1.5rem",
                                "width": "100%"
                            }
                        ),
                        width="100%",
                        spacing="3",
                        margin_top="2rem"
                    )
                ),
                
                width="100%",
                spacing="4"
            ),
            max_width="1400px",
            padding_y="2rem"
        ),
        
        width="100%",
        min_height="100vh",
        background="linear-gradient(to bottom, #f9fafb 0%, #ffffff 100%)"
    )
