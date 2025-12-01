"""Search section component"""
import reflex as rx
from ..state import State


def search_section() -> rx.Component:
    """Enhanced search section with modern styling"""
    return rx.card(
        rx.vstack(
            # Header
            rx.hstack(
                rx.icon("search", size=24, color="#667eea"),
                rx.heading("Find Your Medication", size="5", weight="bold"),
                spacing="2",
                align="center"
            ),
            
            rx.text(
                "Select a medical condition to receive AI-powered drug recommendations",
                size="2",
                color="gray",
                line_height="1.6"
            ),
            
            rx.divider(margin_y="1rem"),
            
            # Search controls
            rx.vstack(
                rx.text("Medical Condition", size="2", weight="bold", margin_bottom="0.5rem"),
                rx.select(
                    State.available_conditions,
                    placeholder="Select a condition (e.g., Depression, Diabetes)...",
                    value=State.selected_condition,
                    on_change=State.set_selected_condition,
                    size="3",
                    width="100%",
                    style={
                        "border": "2px solid #e5e7eb",
                        "_focus": {
                            "border-color": "#667eea",
                            "box-shadow": "0 0 0 3px rgba(102, 126, 234, 0.1)"
                        }
                    }
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
                            rx.icon("sparkles", size=20),
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
                    style={
                        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                        "color": "white",
                        "font-weight": "600",
                        "padding": "1.5rem",
                        "border-radius": "0.75rem",
                        "transition": "all 0.3s ease",
                        "_hover": {
                            "transform": "translateY(-2px)",
                            "box-shadow": "0 10px 15px -3px rgba(102, 126, 234, 0.3)"
                        },
                        "_disabled": {
                            "opacity": "0.5",
                            "cursor": "not-allowed"
                        }
                    }
                ),
                width="100%",
                spacing="3"
            ),
            
            width="100%",
            spacing="3",
            align="start"
        ),
        width="100%",
        style={
            "background": "white",
            "border": "1px solid #e5e7eb",
            "border-radius": "1rem",
            "padding": "2rem",
            "box-shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
        }
    )
