import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled

from weather_tool import get_weather
from news_tool import get_news

load_dotenv()

# Tracing disable kar rahe hain kyunke hum OpenAI ka apna key use nahi kar rahe
set_tracing_disabled(disabled=True)

# Groq ko OpenAI-compatible client ke through point kiya
groq_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

agent = Agent(
    name="Weather & News Assistant",
    instructions="""You are a helpful assistant that provides weather and news information.
Always respond in English only. Never use Hindi, Urdu script, or any other language or script.
If the user mentions a city or location with a spelling mistake, silently correct it to the closest real place before calling a tool.
If you genuinely cannot identify what city, country, or topic the user means, do NOT call any tool. Instead reply exactly with:
"Sorry, kindly let me know which city you mean — I couldn't understand it." """,
    tools=[get_weather, get_news],
    model=OpenAIChatCompletionsModel(
        model="openai/gpt-oss-120b",
        openai_client=groq_client
    )
)

async def main():
    user_input = input("What would you like to know? ")
    result = await Runner.run(agent, user_input)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())