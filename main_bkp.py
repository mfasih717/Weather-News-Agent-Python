import os
import asyncio
import requests
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
from pydantic import BaseModel
from typing import Optional, List, Literal

load_dotenv()

# Tracing disable kar rahe hain kyunke hum OpenAI ka apna key use nahi kar rahe
set_tracing_disabled(disabled=True)

# ---------- RESPONSE MODELS ----------      ← YEH NAYA BLOCK YAHAN AAYEGA

class AgentResponse(BaseModel):
    type: Literal["weather", "news", "chat", "rejection"]
    message: str
    city: Optional[str] = None
    temperature: Optional[float] = None
    description: Optional[str] = None
    articles: Optional[List[str]] = None


# ---------- TOOLS ----------

last_tool_used = None
last_tool_data = {}



@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a given city."""
    global last_tool_used, last_tool_data

    try:
        api_key = os.getenv("WEATHER_API_KEY")
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return f"Error: {data.get('message', 'City not found')}"

        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]



        last_tool_used = "weather"
        last_tool_data = {"city": city, "temperature": temp, "description": description}

        return f"{city} currently has a temperature of {temp}°C with {description}."

    except requests.exceptions.RequestException:
        return "Sorry, I couldn't reach the weather service right now. Please try again in a moment."
    except Exception:
        return "Sorry, something went wrong while fetching the weather."


@function_tool
def get_news(query: str) -> str:
    """Get latest news for a given city, country, or topic."""
    global last_tool_used, last_tool_data

    try:
        api_key = os.getenv("NEWS_API_KEY")
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&language=en&sortBy=publishedAt&pageSize=5"
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("status") != "ok":
            return f"Error: {data.get('message', 'News fetch failed')}"

        articles = data.get("articles", [])
        if not articles:
            return f"No news found about {query}."

        headlines = [article["title"] for article in articles[:5]]

        last_tool_used = "news"
        last_tool_data = {"articles": headlines}

        result = f"Latest news about {query}:\n"
        for i, headline in enumerate(headlines, start=1):
            result += f"{i}. {headline}\n"

        return result

    except requests.exceptions.RequestException:
        return "Sorry, I couldn't reach the news service right now. Please try again in a moment."
    except Exception:
        return "Sorry, something went wrong while fetching the news."

# ---------- AGENT SETUP ----------
print("DEBUG - GROQ_API_KEY present:", bool(os.getenv("GROQ_API_KEY")))
print("DEBUG - GROQ_API_KEY length:", len(os.getenv("GROQ_API_KEY", "")))

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
- If the user asks about anything else unrelated to weather or news (coding, math, general knowledge, opinions, etc.), politely decline and say: "Sorry, that's outside my domain — I can only help with weather and news."
- EXCEPTION: casual, friendly small talk is allowed — like greetings, the user telling you their name, or asking your name. Respond warmly and naturally to these, like a friendly companion would, but always try to gently steer back toward weather/news.

Always respond in English only. Never use Hindi, Urdu script, or any other language or script.
Never use Markdown formatting (no **bold**, no bullet points with *, no headers). Write plain, natural sentences only, since your response is shown in a plain-text chat bubble.
If the user mentions a city or location with a spelling mistake, silently correct it to the closest real place before calling a tool.
If you genuinely cannot identify what city, country, or topic the user means, do NOT call any tool. Instead reply exactly with:
"Sorry, kindly let me know which city you mean — I couldn't understand it." """,
    tools=[get_weather, get_news],
    model=OpenAIChatCompletionsModel(
        model="openai/gpt-oss-120b",
        openai_client=groq_client
    ),
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
    """Return a safe rejection message if the input looks like a prompt injection attempt, else None."""
    lowered = user_message.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lowered:
            return "Sorry, I can't process that request. I can only help with weather and news."
    return None


# ---------- RUN ----------

# Session memory - saari conversation yahan store hogi
conversation_history = []

async def run_agent(user_message: str) -> dict:
    global conversation_history, last_tool_used, last_tool_data

    # Guardrail check - Runner.run se pehle
    blocked_response = check_input_guardrail(user_message)
    if blocked_response:
        rejection = AgentResponse(type="rejection", message=blocked_response)
        return rejection.model_dump()

    # Har naye message se pehle tool-tracking reset karo
    last_tool_used = None
    last_tool_data = {}

    # User ka naya message history mein add karo
    conversation_history.append({"role": "user", "content": user_message})

    try:
        # Poori history agent ko bhejo
        result = await Runner.run(agent, conversation_history)
        final_text = result.final_output

    except Exception as e:
        print(f"Agent error: {e}")  # Terminal mein error log hoga, debugging ke liye

        # User ka message history se hata do (kyunke response nahi mila, history clean rahe)
        conversation_history.pop()

        error_response = AgentResponse(
            type="chat",
            message="Sorry, I'm having trouble connecting right now. Please try again in a moment."
        )
        return error_response.model_dump()

    # History mein text save karo
    conversation_history.append({"role": "assistant", "content": final_text})

    # Ab hum khud decide karte hain response ka "type" - tool tracking ke hisab se
    if last_tool_used == "weather":
        agent_response = AgentResponse(
            type="weather",
            message=final_text,
            city=last_tool_data.get("city"),
            temperature=last_tool_data.get("temperature"),
            description=last_tool_data.get("description"),
        )
    elif last_tool_used == "news":
        agent_response = AgentResponse(
            type="news",
            message=final_text,
            articles=last_tool_data.get("articles"),
        )
    else:
        agent_response = AgentResponse(type="chat", message=final_text)

    return agent_response.model_dump()

def clear_memory():
    """Chat session khatam hone par memory clear karne ke liye"""
    global conversation_history
    conversation_history = []


# Terminal se testing ke liye (optional)
async def main():
    user_input = input("What would you like to know? ")
    response = await run_agent(user_input)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())