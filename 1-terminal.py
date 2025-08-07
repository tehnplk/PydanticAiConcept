from pydantic_ai import Agent
import asyncio
from dotenv import load_dotenv
from pydantic_ai.mcp import MCPServerStdio
from pydantic import BaseModel, Field

load_dotenv()

class Result(BaseModel):
    query: str = Field(description="SQL query id use alias name need cover by ``")
    result: str = Field(description="SQL result in CSV format")
    explanation: str = Field(
        description="Result explanation , if error occur show error message"
    )


mcp_mysql = MCPServerStdio(
    "uvx",
    ["--from", "mysql-mcp-server", "mysql_mcp_server"],
    {
        "MYSQL_HOST": "localhost",
        "MYSQL_PORT": "3310",
        "MYSQL_USER": "root",
        "MYSQL_PASSWORD": "112233",
        "MYSQL_DATABASE": "hos2",
    },
)

agent = Agent(
    #model = "openai:gpt-4o",
    model="google-gla:gemini-2.5-flash", 
    system_prompt="คุณเป็นผู้ช่วยเขียน SQL สำหรับฐานข้อมูล MySQL โดยใช้เครื่องมือ MCP", #ควรอธิบายให้ model รู้จักตารางและการ join
    toolsets=[mcp_mysql],
    output_type=Result,
)



async def start_chat():
    message_history = []
    while True:
        user_prompt = input("User: ")
        if user_prompt == "exit":
            break
        async with agent:
            result = await agent.run(user_prompt,message_history=message_history)
        message_history = result.all_messages()   
        print("AI: ")     
        print('คำอธิบายจาก AI :',result.output.explanation)  
        print('SQL ที่ AI ใช้ :',result.output.query)        
        print('ผลลัพธ์ข้อมูลที่ได้ :',result.output.result) 


if __name__ == "__main__":    
    asyncio.run(start_chat())
