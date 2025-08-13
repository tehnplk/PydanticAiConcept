import sqlite3

def read_data_from_db(db_name='his.db'):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Example: List all tables in the database
        cursor.execute("SELECT * FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in {db_name}: {tables}")

        # Example: Read data from a specific table (replace 'your_table_name' with an actual table name)
        # if tables:
        #     table_name = tables[0][0] # Get the first table name
        #     print(f"\nReading data from table: {table_name}")
        #     cursor.execute(f"SELECT * FROM {table_name}")
        #     rows = cursor.fetchall()
        #     for row in rows:
        #         print(row)

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    read_data_from_db()
