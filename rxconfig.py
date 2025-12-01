import reflex as rx

config = rx.Config(
    app_name="drug_recommendation_system",
    db_url="sqlite:///drug_recommendation.db",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)