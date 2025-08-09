import pymysql
import os
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("MySQL Explorer")


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
                    schema_info.append(f"  {col[0]} {col[1]} {col[2]} {col[3]} {col[4]} {col[5]}")
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
        with conn.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            
            # Format results in a table-like format
            if result:
                # Get column names from cursor description
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                
                if columns:
                    # Create header
                    header = " | ".join(columns)
                    separator = "-" * len(header)
                    
                    # Format rows
                    rows = []
                    for row in result:
                        formatted_row = " | ".join(str(val) if val is not None else "NULL" for val in row)
                        rows.append(formatted_row)
                    
                    return f"{header}\n{separator}\n" + "\n".join(rows)
                else:
                    return "\n".join(str(row) for row in result)
            else:
                return "No results found"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        conn.close()

if __name__ == "__main__":
    mcp.run(transport='sse')
