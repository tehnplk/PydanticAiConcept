from pydantic_ai import Agent
import asyncio
from dotenv import load_dotenv
from pydantic_ai.mcp import MCPServerStdio
from pydantic import BaseModel, Field

load_dotenv()


agent = Agent(
    model="google-gla:gemini-2.5-flash",
    system_prompt="คุณเป็นผู้ช่วยเขียน SQL สำหรับฐานข้อมูล MySQL โดยใช้เครื่องมือ MCP",  # ควรอธิบายให้ model รู้จักตารางและการ join
    output_type=str,
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
