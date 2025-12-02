"""Login page for Drug Recommendation System"""
import reflex as rx
from ..state import State


def login_page() -> rx.Component:
    """Stunning login page with glassmorphism and animations"""
    return rx.box(
        # Background with gradient
        rx.box(
            position="fixed",
            top="0",
            left="0",
            width="100%",
            height="100%",
            background="linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)",
            z_index="-1"
        ),
        
        # Animated background shapes
        rx.box(
            position="fixed",
            top="10%",
            left="10%",
            width="300px",
            height="300px",
            border_radius="50%",
            background="rgba(255, 255, 255, 0.1)",
            filter="blur(60px)",
            animation="float 6s ease-in-out infinite",
            z_index="-1"
        ),
        rx.box(
            position="fixed",
            bottom="10%",
            right="10%",
            width="400px",
            height="400px",
            border_radius="50%",
            background="rgba(255, 255, 255, 0.1)",
            filter="blur(80px)",
            animation="float 8s ease-in-out infinite reverse",
            z_index="-1"
        ),
        
        # Login card
        rx.center(
            rx.card(
                rx.vstack(
                    # Logo and branding
                    rx.vstack(
                        rx.icon(
                            "activity",
                            size=64,
                            style={
                                "color": "white",
                                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                "padding": "1rem",
                                "border-radius": "1rem",
                                "box-shadow": "0 10px 15px -3px rgba(102, 126, 234, 0.3)"
                            }
                        ),
                        rx.heading(
                            "MediRecommend",
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
                            "AI-Powered Healthcare Decision Support",
                            size="3",
                            weight="medium",
                            color="gray",
                            text_align="center"
                        ),
                        spacing="3",
                        align="center"
                    ),
                    
                    rx.divider(margin_y="1.5rem"),
                    
                    # Welcome message
                    rx.vstack(
                        rx.heading("Welcome", size="5", weight="bold"),
                        rx.text(
                            "Enter your username to access personalized drug recommendations",
                            size="2",
                            color="gray",
                            text_align="center",
                            line_height="1.6"
                        ),
                        spacing="2",
                        align="center"
                    ),
                    
                    # Input field with better visibility
                    rx.vstack(
                        rx.input(
                            placeholder="Enter your username",
                            value=State.username,
                            on_change=State.set_username,
                            size="3",
                            width="100%",
                            style={
                                "border": "2px solid #d1d5db",
                                "border-radius": "0.75rem",
                                "padding": "1rem",
                                "font-size": "1rem",
                                "color": "#111827",
                                "background": "white",
                                "_focus": {
                                    "border-color": "#667eea",
                                    "box-shadow": "0 0 0 3px rgba(102, 126, 234, 0.1)",
                                    "outline": "none"
                                },
                                "_hover": {
                                    "border-color": "#9ca3af"
                                }
                            }
                        ),
                        
                        # Error message
                        rx.cond(
                            State.login_error != "",
                            rx.callout(
                                State.login_error,
                                icon="alert-triangle",
                                color_scheme="red",
                                size="2"
                            )
                        ),
                        
                        # Login button
                        rx.button(
                            rx.hstack(
                                rx.icon("arrow-right", size=20),
                                rx.text("Continue", size="3", weight="bold"),
                                spacing="2",
                                align="center"
                            ),
                            on_click=State.handle_login,
                            size="3",
                            width="100%",
                            style={
                                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                "color": "white",
                                "padding": "1.5rem",
                                "border-radius": "0.75rem",
                                "font-weight": "600",
                                "transition": "all 0.3s ease",
                                "_hover": {
                                    "transform": "translateY(-2px)",
                                    "box-shadow": "0 10px 15px -3px rgba(102, 126, 234, 0.4)"
                                }
                            }
                        ),
                        
                        width="100%",
                        spacing="3"
                    ),
                    
                    # Features
                    rx.vstack(
                        rx.divider(margin_y="1rem"),
                        rx.text("Powered by", size="1", color="gray", text_align="center"),
                        rx.hstack(
                            rx.badge("LightGBM", color_scheme="purple", variant="soft"),
                            rx.badge("VADER", color_scheme="blue", variant="soft"),
                            rx.badge("TF-IDF", color_scheme="green", variant="soft"),
                            spacing="2",
                            wrap="wrap",
                            justify="center"
                        ),
                        spacing="2",
                        align="center",
                        width="100%"
                    ),
                    
                    spacing="4",
                    width="100%",
                    align="center"
                ),
                max_width="500px",
                style={
                    "background": "rgba(255, 255, 255, 0.95)",
                    "backdrop-filter": "blur(20px)",
                    "-webkit-backdrop-filter": "blur(20px)",
                    "border": "1px solid rgba(255, 255, 255, 0.3)",
                    "border-radius": "1.5rem",
                    "padding": "3rem",
                    "box-shadow": "0 25px 50px -12px rgba(0, 0, 0, 0.25)",
                    "animation": "fadeIn 0.6s ease-out"
                }
            ),
            height="100vh",
            width="100%"
        ),
        
        # CSS animations
        rx.html(
            """
            <style>
                @keyframes float {
                    0%, 100% { transform: translateY(0px); }
                    50% { transform: translateY(-20px); }
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            </style>
            """
        ),
        
        width="100%",
        min_height="100vh"
    )
