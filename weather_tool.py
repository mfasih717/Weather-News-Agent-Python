import os
import requests
from dotenv import load_dotenv

load_dotenv()

def get_weather(city):
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code != 200:
        return f"Error: {data.get('message', 'City not found')}"
    
    temp = data["main"]["temp"]
    description = data["weather"][0]["description"]
    return f"{city} mein abhi temperature {temp}°C hai aur mausam {description} hai."