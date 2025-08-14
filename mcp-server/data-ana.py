from mcp.server.fastmcp import FastMCP, Context
import pandas as pd
import numpy as np
from io import StringIO
from typing import Dict, Any, List, Union

mcp = FastMCP("Data Analysis MCP Server", host="0.0.0.0", port=8080)


@mcp.tool()
def mean(data: str) -> Dict[str, Any]:
    """
    -หาค่าเฉลี่ย
    

    Args:
        data: ข้อมูลที่ต้องการวิเคราะห์

    Returns:
        Result of the mean
    """
    data = io.StringIO(data)
    return {"success": True, "result": np.mean(data)}


if __name__ == "__main__":
    mcp.run(transport="sse")
