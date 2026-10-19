import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "adaptive_rag.db")

def migrate():
    print(f"Checking database schema at: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print("Database file does not exist yet; will be created on startup.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Create conversations table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id VARCHAR PRIMARY KEY,
        title VARCHAR NOT NULL,
        workspace_id VARCHAR NOT NULL,
        user_id INTEGER NOT NULL,
        kb_id VARCHAR,
        created_at DATETIME,
        updated_at DATETIME,
        FOREIGN KEY(workspace_id) REFERENCES workspaces(id),
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(kb_id) REFERENCES knowledge_bases(id)
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_conversations_id ON conversations (id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_conversations_workspace_id ON conversations (workspace_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_conversations_user_id ON conversations (user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_conversations_kb_id ON conversations (kb_id)")
    print("Table 'conversations' verified/created.")

    # 2. Create query_feedback table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS query_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_id INTEGER NOT NULL,
        conversation_id VARCHAR,
        user_id INTEGER NOT NULL,
        workspace_id VARCHAR NOT NULL,
        rating VARCHAR NOT NULL,
        reason VARCHAR,
        comment TEXT,
        created_at DATETIME,
        updated_at DATETIME,
        FOREIGN KEY(query_id) REFERENCES queries(id) ON DELETE CASCADE,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(workspace_id) REFERENCES workspaces(id),
        CONSTRAINT uq_query_feedback_user UNIQUE (query_id, user_id)
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_query_feedback_id ON query_feedback (id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_query_feedback_query_id ON query_feedback (query_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_query_feedback_user_id ON query_feedback (user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_query_feedback_workspace_id ON query_feedback (workspace_id)")
    print("Table 'query_feedback' verified/created.")

    # 3. Check and add columns to 'queries'
    cursor.execute("PRAGMA table_info(queries)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    columns_to_add = [
        ("conversation_id", "VARCHAR"),
        ("user_id", "INTEGER"),
        ("answer_status", "VARCHAR DEFAULT 'SUCCESS'"),
        ("adaptive_outcome", "VARCHAR"),
        ("citations_data", "TEXT")
    ]

    for col_name, col_type in columns_to_add:
        if col_name not in existing_columns:
            print(f"Adding column '{col_name}' ({col_type}) to 'queries' table...")
            cursor.execute(f"ALTER TABLE queries ADD COLUMN {col_name} {col_type}")
        else:
            print(f"Column '{col_name}' already exists in 'queries'.")

    # Create index on queries.conversation_id if not exists
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_queries_conversation_id ON queries (conversation_id)")

    # 4. Realign configurations index (fix legacy prototype unique index on version_id)
    try:
        cursor.execute("DROP INDEX IF EXISTS ix_configurations_version_id")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_configurations_version_id ON configurations (version_id)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_configurations_workspace_version ON configurations (workspace_id, version_id)")
        print("Configurations indexes verified/realigned.")
    except Exception as e:
        print(f"Configurations index notice: {e}")

    conn.commit()
    conn.close()
    print("Migration completed successfully.")

if __name__ == "__main__":
    migrate()
