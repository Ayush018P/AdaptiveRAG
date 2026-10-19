import sqlite3
import shutil
import os

DB_PATH = "adaptive_rag.db"
BACKUP_PATH = "adaptive_rag.db.bak"

def migrate():
    print(f"Backing up database to {BACKUP_PATH}")
    if os.path.exists(DB_PATH):
        shutil.copy(DB_PATH, BACKUP_PATH)
    else:
        print("Database not found, nothing to migrate.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Create workspaces table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspaces (
                id VARCHAR PRIMARY KEY,
                name VARCHAR,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert legacy workspace if not exists
        cursor.execute("SELECT id FROM workspaces WHERE id = 'legacy'")
        if not cursor.fetchone():
            cursor.execute("INSERT INTO workspaces (id, name) VALUES ('legacy', 'Legacy Workspace')")
            
        tables_to_migrate = [
            "documents",
            "chunks",
            "queries",
            "configurations",
            "experiments"
        ]
        
        for table in tables_to_migrate:
            print(f"Migrating table: {table}")
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [info[1] for info in cursor.fetchall()]
            
            if "workspace_id" not in columns:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN workspace_id VARCHAR NOT NULL DEFAULT 'legacy'")
                
            cursor.execute(f"UPDATE {table} SET workspace_id = 'legacy' WHERE workspace_id IS NULL")
            
        conn.commit()
        print("Migration complete.")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        if os.path.exists(BACKUP_PATH):
            print("Restoring from backup...")
            shutil.copy(BACKUP_PATH, DB_PATH)
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
