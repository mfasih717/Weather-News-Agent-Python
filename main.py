import os
import asyncio
import requests

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled
)
from pydantic import BaseModel
from typing import Optional, List, Literal


load_dotenv()

# Tracing disable kar rahe hain kyunke hum OpenAI ka apna key use nahi kar rahe
set_tracing_disabled(disabled=True)


# ---------- RESPONSE MODELS ----------

class AgentResponse(BaseModel):
    type: Literal["weather", "news", "both", "chat", "rejection"]
    message: str

    city: Optional[str] = None
    temperature: Optional[float] = None
    description: Optional[str] = None
    articles: Optional[List[str]] = None


# ---------- TOOLS ----------

last_tool_used = []

last_tool_data = {
    "weather": {},
    "news": {}
}


@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a given city."""

    global last_tool_used, last_tool_data

    try:
        api_key = os.getenv("WEATHER_API_KEY")

        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={api_key}&units=metric"
        )

        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return f"Error: {data.get('message', 'City not found')}"

        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]

        # Track weather tool
        if "weather" not in last_tool_used:
            last_tool_used.append("weather")

        last_tool_data["weather"] = {
            "city": city,
            "temperature": temp,
            "description": description
        }

        return (
            f"{city} currently has a temperature of "
            f"{temp}°C with {description}."
        )

    except requests.exceptions.RequestException:
        return (
            "Sorry, I couldn't reach the weather service right now. "
            "Please try again in a moment."
        )

    except Exception:
        return "Sorry, something went wrong while fetching the weather."


@function_tool
def get_news(query: str) -> str:
    """Get latest news for a given city, country, or topic."""

    global last_tool_used, last_tool_data

    try:
        api_key = os.getenv("NEWS_API_KEY")

        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={query}"
            f"&apiKey={api_key}"
            f"&language=en"
            f"&sortBy=publishedAt"
            f"&pageSize=5"
        )

        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("status") != "ok":
            return f"Error: {data.get('message', 'News fetch failed')}"

        articles = data.get("articles", [])

        if not articles:
            return f"No news found about {query}."

        headlines = [
            article["title"]
            for article in articles[:5]
        ]

        # Track news tool
        if "news" not in last_tool_used:
            last_tool_used.append("news")

        last_tool_data["news"] = {
            "articles": headlines
        }

        result = f"Latest news about {query}:\n"

        for i, headline in enumerate(headlines, start=1):
            result += f"{i}. {headline}\n"

        return result

    except requests.exceptions.RequestException:
        return (
            "Sorry, I couldn't reach the news service right now. "
            "Please try again in a moment."
        )

    except Exception:
        return "Sorry, something went wrong while fetching the news."


# ---------- AGENT SETUP ----------

# Groq ko OpenAI-compatible client ke through point kiya
groq_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


agent = Agent(
    name="Weather & News Assistant",

    instructions="""You are a friendly assistant whose ONLY job is to provide weather and news information.

STRICT DOMAIN RULE:

- If the user asks about weather or news, help them normally using your tools.

- If the user asks about both weather and news in the same message, answer both parts. Use the weather tool for the weather request and the news tool for the news request. You are allowed and expected to call both tools when both types of information are requested.

- For combined weather and news requests, make sure both requested pieces of information are retrieved before giving the final answer.

- If the user asks about anything else unrelated to weather or news (coding, math, general knowledge, opinions, etc.), politely decline and say: "Sorry, that's outside my domain — I can only help with weather and news."

- EXCEPTION: casual, friendly small talk is allowed — like greetings, the user telling you their name, or asking your name. Respond warmly and naturally to these, like a friendly companion would, but always try to gently steer back toward weather/news.

Always respond in English only. Never use Hindi, Urdu script, or any other language or script.

Never use Markdown formatting (no bold, no bullet points with *, no headers). Write plain, natural sentences only, since your response is shown in a plain-text chat bubble.

If the user mentions a city or location with a spelling mistake, silently correct it to the closest real place before calling a tool.

If you genuinely cannot identify what city, country, or topic the user means, do NOT call any tool. Instead reply exactly with:

"Sorry, kindly let me know which city you mean — I couldn't understand it."
""",

    tools=[
        get_weather,
        get_news
    ],

    model=OpenAIChatCompletionsModel(
        model="openai/gpt-oss-120b",
        openai_client=groq_client
    )

    # output_type=AgentResponse
)


# ---------- GUARDRAILS ----------

BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "ignore your instructions",
    "system prompt",
    "you are now",
    "act as",
    "forget your rules",
    "disregard the above",
]


def check_input_guardrail(user_message: str) -> str | None:
    """
    Return a safe rejection message if the input
    looks like a prompt injection attempt.
    """

    lowered = user_message.lower()

    for pattern in BLOCKED_PATTERNS:

        if pattern in lowered:
            return (
                "Sorry, I can't process that request. "
                "I can only help with weather and news."
            )

    return None


# ---------- RUN ----------

# Session memory - saari conversation yahan store hogi
conversation_history = []


async def run_agent(user_message: str) -> dict:

    global conversation_history
    global last_tool_used
    global last_tool_data

    # Guardrail check - Runner.run se pehle
    blocked_response = check_input_guardrail(user_message)

    if blocked_response:

        rejection = AgentResponse(
            type="rejection",
            message=blocked_response
        )

        return rejection.model_dump()

    # Har naye message se pehle tool-tracking reset karo
    last_tool_used = []

    last_tool_data = {
        "weather": {},
        "news": {}
    }

    # User ka naya message history mein add karo
    conversation_history.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    try:

        # Poori history agent ko bhejo
        result = await Runner.run(
            agent,
            conversation_history
        )

        final_text = result.final_output

    except Exception as e:

        print(f"Agent error: {e}")

        # User ka message history se hata do
        conversation_history.pop()

        error_response = AgentResponse(
            type="chat",
            message=(
                "Sorry, I'm having trouble connecting right now. "
                "Please try again in a moment."
            )
        )

        return error_response.model_dump()

    # History mein text save karo
    conversation_history.append(
        {
            "role": "assistant",
            "content": final_text
        }
    )

    # ---------- BOTH WEATHER + NEWS ----------

    if (
        "weather" in last_tool_used
        and "news" in last_tool_used
    ):

        weather_data = last_tool_data.get(
            "weather",
            {}
        )

        news_data = last_tool_data.get(
            "news",
            {}
        )

        agent_response = AgentResponse(
            type="both",
            message=final_text,

            city=weather_data.get("city"),
            temperature=weather_data.get("temperature"),
            description=weather_data.get("description"),

            articles=news_data.get("articles")
        )


    # ---------- WEATHER ONLY ----------

    elif "weather" in last_tool_used:

        weather_data = last_tool_data.get(
            "weather",
            {}
        )

        agent_response = AgentResponse(
            type="weather",
            message=final_text,

            city=weather_data.get("city"),
            temperature=weather_data.get("temperature"),
            description=weather_data.get("description")
        )


    # ---------- NEWS ONLY ----------

    elif "news" in last_tool_used:

        news_data = last_tool_data.get(
            "news",
            {}
        )

        agent_response = AgentResponse(
            type="news",
            message=final_text,

            articles=news_data.get("articles")
        )


    # ---------- NORMAL CHAT ----------

    else:

        agent_response = AgentResponse(
            type="chat",
            message=final_text
        )

    return agent_response.model_dump()


# ---------- CLEAR MEMORY ----------

def clear_memory():
    """Chat session khatam hone par memory clear karne ke liye."""

    global conversation_history

    conversation_history = []


# ---------- TERMINAL TESTING ----------

async def main():

    user_input = input(
        "What would you like to know? "
    )

    response = await run_agent(user_input)

    print(response)


if __name__ == "__main__":
    asyncio.run(main())