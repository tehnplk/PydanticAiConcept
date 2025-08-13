from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.mcp import MCPServerStdio
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

import logfire

logfire.configure()  
logfire.instrument_pydantic_ai()

mcp_mysql = MCPServerStdio(
    "uvx",
    ["--from", "mysql-mcp-server", "mysql_mcp_server"],
    {
        "MYSQL_HOST": os.getenv("MYSQL_HOST"),
        "MYSQL_PORT": os.getenv("MYSQL_PORT", "3310"),
        "MYSQL_USER": os.getenv("MYSQL_USER", "root"),
        "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD", "112233"),
        "MYSQL_DATABASE": os.getenv("MYSQL_DATABASE", "hos2"),
    },
)




class Result(BaseModel):
    query: str = Field(description="SQL query")
    result: str = Field(description="result of the SQL query display in tabular format")


agent = Agent(
    model="google-gla:gemini-2.5-flash",
    toolsets=[mcp_mysql],
    system_prompt="You are a helpful assistant.",
    output_type=Result,
)


async def chat():
    async with agent:
        result = await agent.run(
            "ขอรายชื่อประชากร 10 คน จากตาราง person เอาคอลัมน์ที่สำคัญๆ 5 คอลัมน์ ดึงข้อมูลโดยใช้ mcp tool"
        )
    print(result.output.query)
    print(result.output.result)


if __name__ == "__main__":
    asyncio.run(chat())
