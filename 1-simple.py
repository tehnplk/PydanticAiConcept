from pydantic_ai import Agent
import asyncio
from dotenv import load_dotenv
from pydantic_ai.mcp import MCPServerSSE
from pydantic import BaseModel, Field

load_dotenv()

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

mcp_server = MCPServerSSE(url="http://localhost:8080/sse")

agent = Agent(
    model="google-gla:gemini-2.5-flash",
    system_prompt="คุณเป็นผู้ช่วยที่เชี่ยวชาญการวิเคราะห์ข้อมูล โดยใช้เครื่องมือ MCP",  # ควรอธิบายให้ model รู้จักตารางและการ join
    output_type=str,
    toolsets=[mcp_server],
)


async def start_chat():
    message_history = []
    while True:
        user_prompt = input("User: ")
        if user_prompt == "exit":
            break
        async with agent:
            result = await agent.run(user_prompt, message_history=message_history)
        message_history = result.all_messages()
        print("AI: ")
        print(result.output)


if __name__ == "__main__":
    asyncio.run(start_chat())
