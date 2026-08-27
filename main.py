import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from weather_tool import get_weather
from news_tool import get_news

load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Tools ki definition — AI ko batata hai kaunse tools available hain
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a given city. If the city name has a spelling mistake, correct it to the closest real city before calling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The corrected city name"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get latest news for a given city, country, or topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The city, country, or topic to search news about"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

system_prompt = """You are a helpful assistant that provides weather and news information.
Always respond in English only. Never use Hindi, Urdu script, or any other language or script.
If the user mentions a city or location with a spelling mistake, silently correct it to the closest real place before calling a tool.
If you genuinely cannot identify what city, country, or topic the user means, do NOT call any tool. Instead reply exactly with:
"Sorry, kindly let me know which country or city you mean — I couldn't understand it."
"""

def run_agent(user_input):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # Loop chalate raho jab tak AI tool calls karna band na kare
    while True:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message
        messages.append(message)

        # Agar AI ne koi tool call nahi kiya, to ye final answer hai
        if not message.tool_calls:
            return message.content

        # Warna, har tool call ko process karo
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if func_name == "get_weather":
                result = get_weather(args["city"])
            elif func_name == "get_news":
                result = get_news(args["query"])
            else:
                result = "Unknown tool"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

# def run_agent(user_input):
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_input}
#     ]

#     response = client.chat.completions.create(
#         # model="gpt-4o-mini",
#         # model="llama-3.3-70b-versatile",
#         model="openai/gpt-oss-120b",
#         messages=messages,
#         tools=tools
#     )

#     message = response.choices[0].message

#     if message.tool_calls:
#         messages.append(message)
#         for tool_call in message.tool_calls:
#             func_name = tool_call.function.name
#             args = json.loads(tool_call.function.arguments)

#             if func_name == "get_weather":
#                 result = get_weather(args["city"])
#             elif func_name == "get_news":
#                 result = get_news(args["query"])
#             else:
#                 result = "Unknown tool"

#             messages.append({
#                 "role": "tool",
#                 "tool_call_id": tool_call.id,
#                 "content": result
#             })


#         final_response = client.chat.completions.create(
#             # model="gpt-4o-mini",
#             # model="llama-3.3-70b-versatile",
#             model="openai/gpt-oss-120b",
#             messages=messages
#         )
#         return final_response.choices[0].message.content
#     else:
#         # AI ne bina tool call kiye seedha jawab diya (jaise sorry message)
#         return message.content


# Test loop
if __name__ == "__main__":
    user_input = input("What would you like to know? ")
    print(run_agent(user_input))