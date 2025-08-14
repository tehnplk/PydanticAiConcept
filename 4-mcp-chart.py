from unittest import result
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.mcp import MCPServerStdio , MCPServerSSE , MCPServerStreamableHTTP
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()


model = OpenAIModel(
    "openai/gpt-oss-20b",
    #"openai/gpt-oss-120b",
    #"google/gemini-2.5-flash-lite-preview-06-17",
    #"qwen/qwen3-30b-a3b",
    #"google/gemini-2.0-flash-lite-001",
    provider=OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY")),
)
#model = "google-gla:gemini-2.5-flash"


mcp_chart = MCPServerSSE(url='http://203.157.118.95/sse')
#mcp_chart = MCPServerStreamableHTTP(url='http://203.157.118.95/mcp')

agent = Agent(
    model=model,
    toolsets=[mcp_chart],
    system_prompt="You are a chart generating expert , you have mcp tool to generate chart.",
    output_type=str,
    
)


async def chat():
    async with agent:
        result = await agent.run(
            "แสดงกราฟข้อมูลนี้\n\n"
            "หมู่ที่ , ประชากร(คน)\n"
            "1 , 150\n"
            "2 , 200\n"
            "3 , 70\n"
            "4 , 50\n"
            "5 , 50"
        )
    print(result.output)

if __name__ == "__main__":
    asyncio.run(chat())
