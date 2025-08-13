from mcp.server.fastmcp import FastMCP, Context
import pandas as pd
import numpy as np
from io import StringIO
from typing import Dict, Any, List, Union

mcp = FastMCP("Data Analysis MCP Server", host="0.0.0.0", port=8080)


@mcp.tool()
def mutiple_linear_regression(data: str,context: Context) -> Dict[str, Any]:
    """
    -หาค่าสถิติต่างๆจากข้อมูลเพื่อใช้วิเคราะห์
    -หาค่าสัมประสิทธิ์ของสมการถดถอย
    -ใช้ multiple linear regression เพื่อวิเคราะห์ข้อมูล

    Args:
        data: ข้อมูลที่ต้องการวิเคราะห์

    Returns:
        Result of the multiple linear regression
    """

    return {"success": True, "result": context}


if __name__ == "__main__":
    mcp.run(transport="sse")
