import os
import requests
from dotenv import load_dotenv
from agents import function_tool

load_dotenv()

@function_tool
def get_news(query: str) -> str:
    """Get latest news for a given city, country, or topic."""
    api_key = os.getenv("NEWS_API_KEY")
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&language=en&sortBy=publishedAt&pageSize=5"
    response = requests.get(url)
    data = response.json()

    if data.get("status") != "ok":
        return f"Error: {data.get('message', 'News fetch failed')}"

    articles = data.get("articles", [])
    if not articles:
        return f"{query} ke baare mein koi news nahi mili."

    result = f"{query} ki latest news:\n"
    for i, article in enumerate(articles[:5], start=1):
        result += f"{i}. {article['title']}\n"

    return result