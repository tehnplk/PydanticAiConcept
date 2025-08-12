from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.mcp import MCPServerSSE , MCPServerStdio
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

import logfire

logfire.configure()  
logfire.instrument_pydantic_ai()



mcp = MCPServerSSE(url="http://localhost:1122/sse")
#mcp = MCPServerStdio( "npx",[ "-y","@antv/mcp-server-chart"])

agent = Agent(
    model="google-gla:gemini-2.5-flash",
    toolsets=[mcp],
    system_prompt="You are a helpful assistant for build chart.",
    output_type=str,
)


async def chat():
    async with agent:
        result = await agent.run(
            "สร้างกราฟแท่งง่ายๆ โดยใช้ mcp"
        )
    print(result.output)


if __name__ == "__main__":
    asyncio.run(chat())
