import requests
from sqlalchemy.orm import Session
from app.models import News
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")
URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"

def fetch_and_store_news(db: Session):
    response = requests.get(URL)
    articles = response.json().get("articles", [])

    for article in articles:
        news = News(
            title=article["title"],
            content=article.get("description"),
            source=article["source"]["name"],
            category="general"
        )
        db.add(news)
    db.commit()
