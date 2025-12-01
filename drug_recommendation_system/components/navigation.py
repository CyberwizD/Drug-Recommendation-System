"""Navigation component for Drug Recommendation System"""
import reflex as rx
from ..state import State


def navigation_bar() -> rx.Component:
    """Enhanced navigation bar with modern design"""
    return rx.box(
        rx.container(
            rx.hstack(
                # Logo and brand
                rx.hstack(
                    rx.icon(
                        "activity",
                        size=32,
                        color="white",
                        style={
                            "filter": "drop-shadow(0 2px 4px rgba(0,0,0,0.2))"
                        }
                    ),
                    rx.heading(
                        "MediRecommend",
                        size="6",
                        color="white",
                        weight="bold",
                        style={
                            "text-shadow": "0 2px 4px rgba(0,0,0,0.2)"
                        }
                    ),
                    spacing="3",
                    align="center"
                ),
                
                rx.spacer(),
                
                # Navigation links
                rx.hstack(
                    rx.link(
                        rx.button(
                            rx.icon("home", size=18),
                            "Home",
                            variant="ghost",
                            size="2",
                            color_scheme="gray",
                            style={
                                "color": "white",
                                "_hover": {
                                    "background": "rgba(255,255,255,0.1)",
                                    "transform": "translateY(-2px)",
                                    "transition": "all 0.3s ease"
                                }
                            }
                        ),
                        href="/"
                    ),
                    rx.link(
                        rx.button(
                            rx.icon("clock", size=18),
                            "History",
                            variant="ghost",
                            size="2",
                            style={
                                "color": "white",
                                "_hover": {
                                    "background": "rgba(255,255,255,0.1)",
                                    "transform": "translateY(-2px)",
                                    "transition": "all 0.3s ease"
                                }
                            }
                        ),
                        href="/history"
                    ),
                    rx.link(
                        rx.button(
                            rx.icon("bar-chart", size=18),
                            "Analysis",
                            variant="ghost",
                            size="2",
                            style={
                                "color": "white",
                                "_hover": {
                                    "background": "rgba(255,255,255,0.1)",
                                    "transform": "translateY(-2px)",
                                    "transition": "all 0.3s ease"
                                }
                            }
                        ),
                        href="/analysis"
                    ),
                    
                    # User section
                    rx.hstack(
                        rx.avatar(
                            fallback=rx.cond(
                                State.username != "",
                                State.username[0].upper(),
                                "U"
                            ),
                            size="2",
                            style={
                                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                "color": "white",
                                "border": "2px solid white"
                            }
                        ),
                        rx.text(
                            State.username,
                            size="2",
                            weight="medium",
                            color="white"
                        ),
                        rx.button(
                            rx.icon("log-out", size=16),
                            on_click=State.handle_logout,
                            variant="soft",
                            color_scheme="red",
                            size="2",
                            style={
                                "background": "rgba(239, 68, 68, 0.2)",
                                "color": "white",
                                "_hover": {
                                    "background": "rgba(239, 68, 68, 0.3)",
                                    "transform": "translateY(-2px)",
                                    "transition": "all 0.3s ease"
                                }
                            }
                        ),
                        spacing="5",
                        align="center",
                        padding_left="1rem",
                        border_left="1px solid rgba(255,255,255,0.2)"
                    ),
                    
                    spacing="5",
                    align="center"
                ),
                
                width="100%",
                align="center",
                padding_y="1rem"
            ),
            max_width="1400px"
        ),
        background="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        position="sticky",
        top="0",
        z_index="1000",
        box_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
        style={
            "backdrop_filter": "blur(10px)",
            "-webkit-backdrop-filter": "blur(10px)"
        }
    )
