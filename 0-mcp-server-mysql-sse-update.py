#!/usr/bin/env python3
"""
MySQL MCP Server using FastMCP
Provides tools for querying MySQL database and resources for table descriptions
"""

import os
import json
from typing import List, Dict, Any, Optional
import pymysql
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP
mcp = FastMCP("MySQL Database Server", host="0.0.0.0", port=8081)

# Database connection configuration
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3310)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "112233"),
    "database": os.getenv("MYSQL_DATABASE", "hos2"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}


def get_db_connection():
    """Create and return a database connection"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        raise Exception(f"Failed to connect to database: {str(e)}")


def get_all_tables() -> List[str]:
    """Get list of all tables in the database"""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = [
                row[f"Tables_in_{DB_CONFIG['database']}"] for row in cursor.fetchall()
            ]
        return tables
    finally:
        connection.close()


def describe_table(table_name: str) -> Dict[str, Any]:
    """Get table structure description"""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # Get table structure
            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()

            # Get table info
            cursor.execute(f"SHOW TABLE STATUS LIKE '{table_name}'")
            table_info = cursor.fetchone()

            # Get indexes
            cursor.execute(f"SHOW INDEX FROM {table_name}")
            indexes = cursor.fetchall()

            return {
                "table_name": table_name,
                "columns": columns,
                "table_info": table_info,
                "indexes": indexes,
            }
    finally:
        connection.close()


@mcp.tool()
def query_data(query: str, limit: Optional[int] = 100) -> Dict[str, Any]:
    """
    Execute a SQL query on the MySQL database

    Args:
        query: SQL query to execute (SELECT statements only for safety)
        limit: Maximum number of rows to return (default: 100)

    Returns:
        Dictionary containing query results and metadata
    """
    # Safety check - only allow SELECT queries
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return {
            "error": "Only SELECT queries are allowed for security reasons",
            "query": query,
        }

    # Add LIMIT if not present and limit is specified
    if limit and "LIMIT" not in query_upper:
        query += f" LIMIT {limit}"

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()

            return {
                "query": query,
                "row_count": len(results),
                "data": results,
                "success": True,
            }
    except Exception as e:
        return {"error": str(e), "query": query, "success": False}
    finally:
        connection.close()


@mcp.resource("table://")
def get_table_resources() -> List[str]:
    """List all available table resources"""
    try:
        tables = get_all_tables()
        return [f"table://{table}" for table in tables]
    except Exception as e:
        return [f"error://failed-to-list-tables: {str(e)}"]


@mcp.resource("table://{table_name}")
def get_table_description(table_name: str) -> str:
    """
    Get detailed description of a specific table

    Args:
        table_name: Name of the table to describe

    Returns:
        JSON string containing table structure and metadata
    """
    try:
        description = describe_table(table_name)

        # Format the description nicely
        output = {
            "table_name": table_name,
            "database": DB_CONFIG["database"],
            "columns": [],
            "indexes": [],
            "table_info": description.get("table_info", {}),
        }

        # Format column information
        for col in description.get("columns", []):
            output["columns"].append(
                {
                    "name": col["Field"],
                    "type": col["Type"],
                    "null": col["Null"],
                    "key": col["Key"],
                    "default": col["Default"],
                    "extra": col["Extra"],
                }
            )

        # Format index information
        for idx in description.get("indexes", []):
            output["indexes"].append(
                {
                    "name": idx["Key_name"],
                    "column": idx["Column_name"],
                    "unique": not bool(idx["Non_unique"]),
                    "type": idx["Index_type"],
                }
            )

        return json.dumps(output, indent=2, ensure_ascii=False, default=str)

    except Exception as e:
        error_info = {
            "error": f"Failed to describe table '{table_name}': {str(e)}",
            "table_name": table_name,
        }
        return json.dumps(error_info, indent=2, ensure_ascii=False)


@mcp.resource("database://info")
def get_database_info() -> str:
    """Get general database information"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # Get database version
                cursor.execute("SELECT VERSION() as version")
                version_info = cursor.fetchone()

                # Get table count
                tables = get_all_tables()

                # Get database size
                cursor.execute(
                    f"""
                    SELECT 
                        ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS 'size_mb'
                    FROM information_schema.tables 
                    WHERE table_schema = '{DB_CONFIG['database']}'
                """
                )
                size_info = cursor.fetchone()

                info = {
                    "database_name": DB_CONFIG["database"],
                    "host": DB_CONFIG["host"],
                    "port": DB_CONFIG["port"],
                    "version": (
                        version_info.get("version", "Unknown")
                        if version_info
                        else "Unknown"
                    ),
                    "table_count": len(tables),
                    "tables": tables,
                    "size_mb": size_info.get("size_mb", 0) if size_info else 0,
                    "connection_status": "Connected",
                }

                return json.dumps(info, indent=2, ensure_ascii=False, default=str)
        finally:
            connection.close()

    except Exception as e:
        error_info = {
            "error": f"Failed to get database info: {str(e)}",
            "connection_status": "Failed",
        }
        return json.dumps(error_info, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Test connection on startup
    try:
        connection = get_db_connection()
        connection.close()
        print(f"✅ Successfully connected to MySQL database: {DB_CONFIG['database']}")
        print(f"📊 Available tables: {', '.join(get_all_tables())}")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print("Please check your database configuration in environment variables:")
        print("- MYSQL_HOST (default: localhost)")
        print("- MYSQL_PORT (default: 3310)")
        print("- MYSQL_USER (default: root)")
        print("- MYSQL_PASSWORD (default: 112233)")
        print("- MYSQL_DATABASE (default: hos2)")

    # Run the MCP server
    mcp.run(transport='sse')
