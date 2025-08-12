from unittest import result
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.mcp import MCPServerStdio , MCPServerSSE
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()


class ChartResult(BaseModel):
    result: str = Field(description="markdown tag")


model = OpenAIModel(
    "openai/gpt-oss-20b",
    provider=OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY")),
)
#model = "google-gla:gemini-2.5-flash-lite-preview-06-17"

#mcp_chart = MCPServerStdio("npx", ["-y", "@antv/mcp-server-chart"])
mcp_chart = MCPServerSSE(url="http://localhost:1221/sse")

agent = Agent(
    model=model,
    toolsets=[mcp_chart],
    system_prompt="You are a helpful assistant.",
    output_type=str,
    retries=1,
)


async def chat():
    async with agent:
        result = await agent.run(
            "แสดงแผนภูมิแท่งจากข้อมูลนี้\n\n"
            "หมู่ที่ , ประชากร(คน)\n"
            "1 , 100\n"
            "2 , 200\n"
            "3 , 70\n"
            "4 , 50\n"
            "5 , 50"
        )
    print(result.output)


if __name__ == "__main__":
    asyncio.run(chat())
