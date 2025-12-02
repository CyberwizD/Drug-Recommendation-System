"""History page for Drug Recommendation System"""
import reflex as rx
from ..state import State
from ..components import navigation_bar


def history_page() -> rx.Component:
    """Timeline-style history page with modern design"""
    return rx.box(
        navigation_bar(),
        
        rx.container(
            rx.vstack(
                # Header
                rx.vstack(
                    rx.hstack(
                        rx.icon("clock", size=36, color="#667eea"),
                        rx.heading(
                            "Search History",
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
                        "View and revisit your previous drug recommendation searches",
                        size="3",
                        color="gray",
                        text_align="center"
                    ),
                    spacing="3",
                    align="center",
                    padding_y="2rem"
                ),
                
                # History items
                rx.cond(
                    State.search_history.length() > 0,
                    rx.vstack(
                        rx.foreach(
                            State.search_history,
                            lambda item: rx.card(
                                rx.vstack(
                                    # Header with condition and timestamp
                                    rx.hstack(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.icon("file-text", size=20, color="#667eea"),
                                                rx.heading(item.condition, size="5", weight="bold"),
                                                spacing="2",
                                                align="center"
                                            ),
                                            rx.hstack(
                                                rx.icon("calendar", size=14, color="gray"),
                                                rx.text(
                                                    item.timestamp,
                                                    size="2",
                                                    color="gray"
                                                ),
                                                spacing="1",
                                                align="center"
                                            ),
                                            align="start",
                                            spacing="2"
                                        ),
                                        rx.spacer(),
                                        rx.vstack(
                                            rx.badge(
                                                f"{item.recommendations.length()} recommendations",
                                                color_scheme="blue",
                                                size="2",
                                                variant="soft"
                                            ),
                                            rx.button(
                                                rx.icon("refresh-cw", size=16),
                                                "View Again",
                                                on_click=[
                                                    State.set_selected_condition(item.condition),
                                                    State.set_search_performed(False),
                                                    State.search_drugs
                                                ],
                                                size="2",
                                                variant="soft",
                                                style={
                                                    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                                    "color": "white",
                                                    "_hover": {
                                                        "transform": "translateY(-2px)",
                                                        "box-shadow": "0 4px 6px rgba(102, 126, 234, 0.3)"
                                                    }
                                                }
                                            ),
                                            spacing="2",
                                            align="end"
                                        ),
                                        width="100%",
                                        align="start"
                                    ),
                                    
                                    spacing="3",
                                    align="start",
                                    width="100%"
                                ),
                                width="100%",
                                style={
                                    "background": "white",
                                    "border": "1px solid #e5e7eb",
                                    "border-left": "4px solid #667eea",
                                    "border-radius": "0.75rem",
                                    "padding": "1.5rem",
                                    "transition": "all 0.3s ease",
                                    "_hover": {
                                        "transform": "translateX(4px)",
                                        "box-shadow": "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
                                        "border-left-color": "#764ba2"
                                    }
                                }
                            )
                        ),
                        width="100%",
                        spacing="3"
                    ),
                    
                    # Empty state
                    rx.card(
                        rx.vstack(
                            rx.icon("inbox", size=64, color="#6b7280"),
                            rx.heading("No History Yet", size="6", color="gray"),
                            rx.text(
                                "Start by searching for drug recommendations on the home page!",
                                size="3",
                                color="gray",
                                text_align="center",
                                max_width="400px"
                            ),
                            rx.link(
                                rx.button(
                                    rx.icon("home", size=18),
                                    "Go to Home",
                                    size="3",
                                    style={
                                        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                        "color": "white",
                                        "_hover": {
                                            "transform": "translateY(-2px)",
                                            "box-shadow": "0 10px 15px -3px rgba(102, 126, 234, 0.3)"
                                        }
                                    }
                                ),
                                href="/"
                            ),
                            spacing="4",
                            align="center",
                            padding="4rem"
                        ),
                        style={
                            "border": "2px dashed #e5e7eb",
                            "border-radius": "1rem"
                        }
                    )
                ),
                
                width="100%",
                spacing="4"
            ),
            max_width="1000px",
            padding_y="2rem"
        ),
        
        width="100%",
        min_height="100vh",
        background="linear-gradient(to bottom, #f9fafb 0%, #ffffff 100%)"
    )
