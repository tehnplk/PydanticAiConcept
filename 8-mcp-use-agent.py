import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient


async def main():
    # Load environment variables
    load_dotenv()

    # Create configuration dictionary
    mcp_config = {
        "mcpServers": {
            "airbnb": {"command": "npx", "args": ["-y", "@openbnb/mcp-server-airbnb"]},
            "stat": {"url": "http://localhost:8080/sse"},
        }
    }

    # Create MCPClient from configuration dictionary
    mcp_client = MCPClient.from_dict(mcp_config)

    # Create LLM
    gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    gpt = ChatOpenAI(model="gpt-4o")

    # Create agent with the client
    agent = MCPAgent(llm=gemini, client=mcp_client, max_steps=30)

    # Run the query

    result = await agent.run(
        """
        หาค่าเฉลี่ย  3,5,6,9,1
          
          """
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    asyncio.run(main())
