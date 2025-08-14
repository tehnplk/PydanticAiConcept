from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.mcp import MCPServerSSE, MCPServerStreamableHTTP
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

mcp_chart = MCPServerStreamableHTTP(url=os.getenv("MCP_CHART_URL"))
mcp_db = MCPServerSSE(url=os.getenv("MCP_SANDBOX_HOS_URL"))


class Result(BaseModel):
    query: str = Field(description="SQL query")
    result: str = Field(description="result of the SQL query display in tabular format")


model = OpenAIModel(
    "openai/gpt-oss-20b",
    # "openai/gpt-oss-120b",
    # "google/gemini-2.5-flash-lite-preview-06-17",
    # "qwen/qwen3-30b-a3b",
    # "google/gemini-2.0-flash-lite-001",
    provider=OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY")),
)
# model = "google-gla:gemini-2.5-flash"
sys_prompt = open("sys_prompt.txt", "r", encoding="utf-8").read()
agent = Agent(
    model=model,
    toolsets=[mcp_chart, mcp_db],
    system_prompt=sys_prompt,
    output_type=str,
)


async def chat():
    async with agent:
        result = await agent.run("นับจำนวนประชากรแยกรายหมู่บ้าน นับจาก house และนำผลลัพธ์ไปสร้างเป็นกราฟแท่ง")
    print(result.output)
    #print(result.output.result)


if __name__ == "__main__":
    asyncio.run(chat())
