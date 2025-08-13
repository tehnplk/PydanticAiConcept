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
    model="google-gla:gemini-2.0-flash",
    system_prompt="คุณเป็นผู้ช่วยที่เชี่ยวชาญการวิเคราะห์ข้อมูล โดยใช้เครื่องมือ MCP", 
    output_type=str,
    toolsets=[mcp_server],
)


async def start_chat():
    async with agent:
        result = await agent.run("""
        
        
| พื้นที่ใช้สอย (ตร.ม.) — X1     | จำนวนห้องนอน — X2  | ราคาบ้าน (บาท) — Y  |
| -------------------------- | ----------------- | ------------------ |
| 50                         | 1                 | 1,100,000          |
| 60                         | 2                 | 1,250,000          |
| 70                         | 2                 | 1,400,000          |
| 80                         | 3                 | 1,600,000          |
| 90                         | 3                 | 1,750,000          |

        
    นำข้อมูลตัวอย่างไปเทรน
    แล้วทำนายราคาบ้านที่มีพื้นที่ใช้สอย 110 ตร.ม. และ 4 ห้องนอน
        
        """)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(start_chat())
