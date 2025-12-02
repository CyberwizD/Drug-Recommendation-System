import reflex as rx
import os

config = rx.Config(
    app_name="drug_recommendation_system",
    db_url="sqlite:///drug_recommendation.db",
    # Production API URL for Render
    api_url=os.getenv("API_URL", "https://drug-recommendation-system-2zq1.onrender.com"),
    deploy_url="https://drug-recommendation-system-2zq1.onrender.com",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)