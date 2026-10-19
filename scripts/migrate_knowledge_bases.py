import sqlite3
import shutil
import os
import uuid

DB_PATH = "adaptive_rag.db"
BACKUP_PATH = "adaptive_rag.db.bak_p1_1"

def migrate():
    print(f"Checking database at {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print("Database not found, nothing to migrate.")
        return

    print(f"Backing up database to {BACKUP_PATH}")
    shutil.copy(DB_PATH, BACKUP_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # 1. Create knowledge_bases table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                description TEXT,
                workspace_id VARCHAR NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(workspace_id) REFERENCES workspaces(id)
            )
        """)

        # 2. Check and add columns to documents table
        cursor.execute("PRAGMA table_info(documents)")
        doc_cols = [info[1] for info in cursor.fetchall()]
        if "kb_id" not in doc_cols:
            print("Adding kb_id column to documents table")
            cursor.execute("ALTER TABLE documents ADD COLUMN kb_id VARCHAR")
        if "error_message" not in doc_cols:
            print("Adding error_message column to documents table")
            cursor.execute("ALTER TABLE documents ADD COLUMN error_message TEXT")

        # 3. Check and add columns to chunks table
        cursor.execute("PRAGMA table_info(chunks)")
        chunk_cols = [info[1] for info in cursor.fetchall()]
        if "kb_id" not in chunk_cols:
            print("Adding kb_id column to chunks table")
            cursor.execute("ALTER TABLE chunks ADD COLUMN kb_id VARCHAR")

        # 3.5 Check and add columns to queries table
        cursor.execute("PRAGMA table_info(queries)")
        query_cols = [info[1] for info in cursor.fetchall()]
        if "kb_id" not in query_cols:
            print("Adding kb_id column to queries table")
            cursor.execute("ALTER TABLE queries ADD COLUMN kb_id VARCHAR")

        # 4. Ensure default knowledge base exists for each workspace
        cursor.execute("SELECT id, name FROM workspaces")
        workspaces = cursor.fetchall()
        if not workspaces:
            # If no workspaces, insert 'legacy' workspace
            cursor.execute("INSERT OR IGNORE INTO workspaces (id, name) VALUES ('legacy', 'Legacy Workspace')")
            workspaces = [('legacy', 'Legacy Workspace')]

        for ws_id, ws_name in workspaces:
            cursor.execute("SELECT id FROM knowledge_bases WHERE workspace_id = ? ORDER BY created_at ASC LIMIT 1", (ws_id,))
            kb_row = cursor.fetchone()
            if not kb_row:
                kb_id = f"default_{ws_id}"
                print(f"Creating default knowledge base for workspace {ws_id}: {kb_id}")
                cursor.execute(
                    "INSERT INTO knowledge_bases (id, name, description, workspace_id) VALUES (?, ?, ?, ?)",
                    (kb_id, "Default Knowledge Base", "Default workspace knowledge base", ws_id)
                )
            else:
                kb_id = kb_row[0]

            # Assign orphaned documents in this workspace to default KB
            cursor.execute("UPDATE documents SET kb_id = ? WHERE workspace_id = ? AND (kb_id IS NULL OR kb_id = '')", (kb_id, ws_id))
            # Assign chunks in this workspace to default KB
            cursor.execute("UPDATE chunks SET kb_id = ? WHERE workspace_id = ? AND (kb_id IS NULL OR kb_id = '')", (kb_id, ws_id))

        conn.commit()
        print("Knowledge bases migration completed successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Migration error: {e}")
        if os.path.exists(BACKUP_PATH):
            print("Restoring from backup...")
            shutil.copy(BACKUP_PATH, DB_PATH)
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
