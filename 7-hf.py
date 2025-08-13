from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers.huggingface import HuggingFaceProvider
from pydantic_ai.mcp import MCPServerSSE
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio

load_dotenv()

import logfire
import os

logfire.configure()
logfire.instrument_pydantic_ai()

model = HuggingFaceModel(
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    provider=HuggingFaceProvider(api_key=os.getenv("HF_TOKEN")),
)

mcp_server = MCPServerSSE(url="http://localhost:8081/sse")

agent = Agent(
    model=model,
    toolsets=[mcp_server],
    system_prompt="You are a database and chart generating expert , you have mcp tool to perform the task.",
    output_type=str,
    retries=5,
)


async def chat():
    async with agent:
        result = await agent.run("ขอดูรายละเอียดตาราง person")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(chat())
