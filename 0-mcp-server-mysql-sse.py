import pymysql
import os
import json
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("MySQL Explorer", host="0.0.0.0", port=8080)


@mcp.resource("schema://main")
def get_schema() -> str:
    """Provide the database schema as a resource"""
    conn = pymysql.connect(
        host=os.getenv('MYSQL_HOST', 'localhost'),
        user=os.getenv('MYSQL_USER', 'root'),
        password=os.getenv('MYSQL_PASSWORD', ''),
        database=os.getenv('MYSQL_DATABASE', 'his'),
        charset='utf8mb4'
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            schema_info = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                
                schema_info.append(f"Table: {table_name}")
                for col in columns:
                    # MySQL DESCRIBE returns: Field, Type, Null, Key, Default, Extra
                    col_info = " | ".join(str(item) if item is not None else "NULL" for item in col)
                    schema_info.append(f"  {col_info}")
                schema_info.append("")
            
            return "\n".join(schema_info)
    finally:
        conn.close()


@mcp.tool()
def query_data(sql: str) -> str:
    """Execute SQL queries safely"""
    conn = pymysql.connect(
        host=os.getenv('MYSQL_HOST', 'localhost'),
        port=os.getenv('MYSQL_PORT', 3310),
        user=os.getenv('MYSQL_USER', 'root'),
        password=os.getenv('MYSQL_PASSWORD', '112233'),
        database=os.getenv('MYSQL_DATABASE', 'hos2'),
        charset='utf8mb4'
    )
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            
            # Format results as JSON (DictCursor already returns dictionaries)
            if result:
                return json.dumps(result, ensure_ascii=False, indent=2, default=str)
            else:
                return json.dumps({"message": "No results found"}, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        conn.close()

if __name__ == "__main__":
    mcp.run(transport='sse')
