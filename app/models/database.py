import sqlite3
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

def configure_sqlite_connection(dbapi_connection, connection_record):
    """
    Configure SQLite connection with:
    1. PRAGMA foreign_keys = ON (enforces relational integrity)
    2. PRAGMA busy_timeout = 30000 (30-second bounded timeout for write contention)
    3. PRAGMA journal_mode = WAL & PRAGMA synchronous = NORMAL (for persistent disk databases only)
    """
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.execute("PRAGMA busy_timeout = 30000;")

            # Check whether this connection is attached to a persistent file or is in-memory
            cursor.execute("PRAGMA database_list;")
            databases = cursor.fetchall()
            is_persistent = False
            for db_info in databases:
                # db_info tuple: (seq, name, file_path)
                if db_info[1] == "main" and db_info[2]:
                    is_persistent = True
                    break

            if is_persistent:
                cursor.execute("PRAGMA journal_mode = WAL;")
                cursor.execute("PRAGMA synchronous = NORMAL;")
        finally:
            cursor.close()

def attach_sqlite_pragmas(target_engine):
    """Attach the configure_sqlite_connection event listener to an engine."""
    event.listen(target_engine, "connect", configure_sqlite_connection)
    return target_engine

connect_args = {"check_same_thread": False, "timeout": 30.0}
engine = create_engine(settings.database_url, connect_args=connect_args)
attach_sqlite_pragmas(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
