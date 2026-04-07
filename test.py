import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from langsmith import traceable

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Gets the current weather for a given city."""
    weather_data = {
        "San Francisco": "Foggy, 62°F",
        "New York": "Sunny, 75°F",
        "London": "Rainy, 55°F",
        "Tokyo": "Clear, 68°F",
    }
    return weather_data.get(city, "Weather data not available for this city.")


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

model_with_tools = model.bind_tools([get_weather])

SYSTEM_PROMPT = "You are a friendly travel assistant who helps with weather information."

TOOLS = {
    "get_weather": get_weather,
}


@traceable(
    name="weather_agent_run",
    tags=["gemini-2.0-flash", "test", "weather", "langsmith-check"],
)
async def run_agent(query: str) -> str:
    """
    Runs a simple ReAct-style loop:
    1. Send user message to Gemini
    2. If Gemini calls a tool -> execute it, send result back
    3. Repeat until Gemini gives a final text response
    """
    print(f"\n{'='*55}")
    print(f"  Query: {query}")
    print(f"{'='*55}")

    messages = [
        ("system", SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    while True:
        response: AIMessage = await model_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            final_answer = response.content
            print(f"\n[Final Answer]:\n{final_answer}")
            return final_answer

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            print(f"\n[Tool Call] -> {tool_name}({tool_args})")

            if tool_name in TOOLS:
                result = TOOLS[tool_name].invoke(tool_args)
            else:
                result = f"Error: tool '{tool_name}' not found."

            print(f"[Tool Result] -> {result}")

            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call_id,
                )
            )


async def main() -> None:
    await run_agent("What's the weather like in San Francisco and Tokyo?")

if __name__ == "__main__":
    asyncio.run(main())
