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



mcp_chart  = MCPServerStdio(
    "npx",
      ["-y", "@antv/mcp-server-chart"],
      {
        "DISABLED_TOOLS": "generate_treemap_chart,generate_district_map,generate_organization_chart,generate_fishbone_diagram,generate_mind_map"
      }
      )


class Result(BaseModel):
    chart: str = Field(description="chart")
    explanition: str = Field(description="conclusion")


agent = Agent(
    model="google-gla:gemini-2.5-flash",
    toolsets=[ mcp_chart],
    system_prompt="You are a helpful assistant for build chart.",
    output_type=Result,
)


async def chat():
    async with agent:
        result = await agent.run(
            "สร้างกราฟแท่งง่ายๆ โดยใช้ mcp tool"
        )
    print(result.output.chart)
    print(result.output.explanition)


if __name__ == "__main__":
    asyncio.run(chat())
