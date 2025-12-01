"""Loading components for Drug Recommendation System"""
import reflex as rx


def loading_spinner() -> rx.Component:
    """Animated loading spinner"""
    return rx.center(
        rx.vstack(
            rx.spinner(
                size="3",
                color="purple"
            ),
            rx.text(
                "Loading...",
                size="3",
                color="gray",
                weight="medium"
            ),
            spacing="3",
            align="center"
        ),
        padding="3rem"
    )


def card_skeleton() -> rx.Component:
    """Skeleton loader for drug cards"""
    return rx.card(
        rx.vstack(
            rx.skeleton(height="2rem", width="100%"),
            rx.skeleton(height="1.5rem", width="80%"),
            rx.divider(margin_y="1rem"),
            rx.hstack(
                rx.skeleton(height="4rem", width="30%"),
                rx.skeleton(height="4rem", width="30%"),
                rx.skeleton(height="4rem", width="30%"),
                spacing="4",
                width="100%"
            ),
            rx.divider(margin_y="1rem"),
            rx.skeleton(height="2rem", width="60%"),
            spacing="3",
            width="100%"
        ),
        width="100%",
        padding="1.5rem"
    )
