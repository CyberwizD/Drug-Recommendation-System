import reflex as rx
import os

config = rx.Config(
    app_name="drug_recommendation_system",
    db_url="sqlite:///drug_recommendation.db",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)