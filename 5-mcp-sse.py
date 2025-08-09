from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.mcp import MCPServerSSE
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

mcp_server = MCPServerSSE(url="http://localhost:8000/sse")


class Result(BaseModel):
    query: str = Field(description="SQL query")
    result: str = Field(description="result of the SQL query display in tabular format")


sys_prompt = open("sys_prompt.txt", "r", encoding="utf-8").read()
agent = Agent(
    model="google-gla:gemini-2.5-flash",
    toolsets=[mcp_server],
    system_prompt=sys_prompt,
    output_type=Result,
)


async def chat():
    async with agent:
        result = await agent.run("แสดงจำนวนผู้ป่วย 10 ลำดับโรคที่พบมากที่สุด ปี 2567")
    print(result.output.query)
    print(result.output.result)


if __name__ == "__main__":
    asyncio.run(chat())
