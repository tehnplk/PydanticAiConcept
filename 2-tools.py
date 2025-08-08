import random
from dotenv import load_dotenv

load_dotenv()

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
import pymysql


class Result(BaseModel):
    query: str = Field(..., description="SQL query")
    explanation: str = Field(..., description="Explanation of the SQL query")


def execute_sql_command(sql_command: str):
    """Execute a SQL command."""
    connection = pymysql.connect(
        host="localhost",
        user="root",
        password="112233",
        port=3310,
        database="hos2",
    )
    cursor = connection.cursor(cursor=pymysql.cursors.DictCursor)
    cursor.execute(sql_command)
    return cursor.fetchall()


agent = Agent(
    "google-gla:gemini-2.5-flash",
    deps_type=str,
    system_prompt=(
        "You're a SQL Expert, you should create a SQL command from user's question"
    ),
    output_type=Result,
)


@agent.tool_plain
def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))


@agent.tool
def create_sql_command(ctx: RunContext[str]) -> Result:
    """Create a SQL command."""
    return ctx.deps




async def chat():
    while True:
        user_prompt = input("User: ")
        if user_prompt == "exit":
            break
        async with agent:
            result = await agent.run(
                user_prompt,
                deps="description table structure for build correct sql command",
            )
        print(result.output)
        sql = result.output.query
        rows = execute_sql_command(sql)
        print("result", rows)


if __name__ == "__main__":
    import asyncio

    asyncio.run(chat())
