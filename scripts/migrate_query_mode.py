import sqlite3
import os

DB_PATH = "adaptive_rag.db"

def migrate_mode():
    if not os.path.exists(DB_PATH):
        print("Database not found, nothing to migrate.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("PRAGMA table_info(queries)")
        query_cols = [info[1] for info in cursor.fetchall()]
        if "mode" not in query_cols:
            print("Adding mode column to queries table with default 'standard'")
            cursor.execute("ALTER TABLE queries ADD COLUMN mode VARCHAR DEFAULT 'standard' NOT NULL")
            conn.commit()
            print("Successfully added mode column to queries table.")
        else:
            print("queries table already has mode column.")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_mode()
