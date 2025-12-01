"""Main Reflex application for Drug Recommendation System"""
import reflex as rx
from .state import State
from .pages import login_page, home_page, history_page, analysis_page


# Create the app
app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="purple",
        radius="large"
    ),
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
    ],
    style={
        "font_family": "Inter, sans-serif"
    }
)


# Add pages
app.add_page(
    rx.cond(State.is_logged_in, home_page(), login_page()),
    route="/",
    title="MediRecommend - Home",
    on_load=State.on_load
)

app.add_page(
    rx.cond(State.is_logged_in, history_page(), login_page()),
    route="/history",
    title="MediRecommend - History",
    on_load=State.on_load
)

app.add_page(
    rx.cond(State.is_logged_in, analysis_page(), login_page()),
    route="/analysis",
    title="MediRecommend - Analysis",
    on_load=State.on_load
)
