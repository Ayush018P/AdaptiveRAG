import os
import sqlite3
import tempfile
import threading
import time
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from app.models.database import Base, attach_sqlite_pragmas, get_db
from app.models.workspace import Workspace
from app.models.user import User
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.conversation import Conversation
from app.models.query import QueryLog
from app.models.feedback import QueryFeedback
from app.rag.ingestion import delete_document, delete_knowledge_base_cascade


def create_temp_persistent_engine():
    """Helper to create an isolated persistent SQLite database engine with attached pragmas."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_persistent.db")
    eng = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False, "timeout": 30.0})
    attach_sqlite_pragmas(eng)
    Base.metadata.create_all(bind=eng)
    return eng, db_path, temp_dir


def test_persistent_database_pragmas_configured():
    """Verify that persistent SQLite engines enable foreign keys, WAL mode, and 30s busy timeout."""
    eng, db_path, temp_dir = create_temp_persistent_engine()
    try:
        with eng.connect() as conn:
            fk = conn.execute(text("PRAGMA foreign_keys;")).scalar()
            jm = conn.execute(text("PRAGMA journal_mode;")).scalar()
            bt = conn.execute(text("PRAGMA busy_timeout;")).scalar()

            assert fk == 1, "Foreign keys must be enabled"
            assert str(jm).lower() == "wal", "Persistent database must use WAL mode"
            assert bt == 30000, "Busy timeout must be set to 30000 ms (30s)"
    finally:
        eng.dispose()


def test_in_memory_database_compatibility():
    """Verify that in-memory SQLite engines enable foreign keys and busy timeout without crashing on WAL."""
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    attach_sqlite_pragmas(eng)
    with eng.connect() as conn:
        fk = conn.execute(text("PRAGMA foreign_keys;")).scalar()
        jm = conn.execute(text("PRAGMA journal_mode;")).scalar()
        bt = conn.execute(text("PRAGMA busy_timeout;")).scalar()

        assert fk == 1, "Foreign keys must be enabled in in-memory engines"
        assert str(jm).lower() == "memory", "In-memory database should retain memory journal mode safely"
        assert bt == 30000, "Busy timeout must be set to 30000 ms"
    eng.dispose()


def test_foreign_key_enforcement_prevents_orphan_records():
    """Verify that foreign key constraints actively prevent inserting records with invalid parent references."""
    eng, _, temp_dir = create_temp_persistent_engine()
    Session = sessionmaker(bind=eng)
    db = Session()
    try:
        # 1. Attempt to insert user referencing nonexistent workspace
        bad_user = User(
            username="orphan_user",
            hashed_password="hashed_pw",
            workspace_id="nonexistent_workspace_id_12345"
        )
        db.add(bad_user)
        with pytest.raises(IntegrityError) as exc_info:
            db.commit()
        db.rollback()
        assert "FOREIGN KEY constraint failed" in str(exc_info.value)

        # 2. Insert valid workspace and user
        ws = Workspace(name="Test Workspace")
        db.add(ws)
        db.commit()
        db.refresh(ws)

        valid_user = User(username="valid_user", hashed_password="pw", workspace_id=ws.id)
        db.add(valid_user)
        db.commit()
        assert valid_user.id is not None

        # 3. Attempt to insert chunk referencing nonexistent document
        bad_chunk = Chunk(
            document_id=999999,
            text="Orphan chunk",
            chunk_index=0,
            workspace_id=ws.id
        )
        db.add(bad_chunk)
        with pytest.raises(IntegrityError) as exc_info_chunk:
            db.commit()
        db.rollback()
        assert "FOREIGN KEY constraint failed" in str(exc_info_chunk.value)

    finally:
        db.close()
        eng.dispose()


def test_failed_write_rolls_back_cleanly():
    """Verify that aborted operations roll back cleanly and leave no lingering state or locks."""
    eng, _, temp_dir = create_temp_persistent_engine()
    Session = sessionmaker(bind=eng)
    db = Session()

    try:
        # Insert a valid workspace
        ws = Workspace(name="Rollback Test WS")
        db.add(ws)
        db.flush()

        # Simulate a transaction failure by attempting an illegal insert
        orphan = User(username="fail_user", hashed_password="pw", workspace_id="bogus_ws")
        db.add(orphan)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()

        # Verify that after rollback, the uncommitted workspace was rolled back cleanly
        count = db.query(Workspace).filter(Workspace.name == "Rollback Test WS").count()
        assert count == 0, "Uncommitted data must not be persisted after rollback"

        # Verify session is healthy and can perform subsequent successful operations
        clean_ws = Workspace(name="Clean WS")
        db.add(clean_ws)
        db.commit()
        assert clean_ws.id is not None
    finally:
        db.close()
        eng.dispose()


def test_referential_integrity_on_document_and_kb_deletion():
    """Verify that deleting documents and knowledge bases cleanly handles foreign keys and cascades."""
    eng, _, temp_dir = create_temp_persistent_engine()
    Session = sessionmaker(bind=eng)
    db = Session()

    try:
        # Setup workspace and user
        ws = Workspace(name="Cascade WS")
        db.add(ws)
        db.commit()

        user = User(username="cascade_user", hashed_password="pw", workspace_id=ws.id)
        db.add(user)
        db.commit()

        # Setup KB
        kb = KnowledgeBase(name="KB Cascade Test", workspace_id=ws.id)
        db.add(kb)
        db.commit()

        # Setup Document and Chunks
        doc = Document(filename="doc1.txt", doc_type=".txt", workspace_id=ws.id, kb_id=kb.id, status="READY")
        db.add(doc)
        db.commit()

        chunk1 = Chunk(document_id=doc.id, text="Chunk 1 text", chunk_index=0, workspace_id=ws.id, kb_id=kb.id)
        chunk2 = Chunk(document_id=doc.id, text="Chunk 2 text", chunk_index=1, workspace_id=ws.id, kb_id=kb.id)
        db.add_all([chunk1, chunk2])
        db.commit()

        # Setup Conversation, QueryLog and Feedback referencing KB
        conv = Conversation(title="KB Conv", workspace_id=ws.id, user_id=user.id, kb_id=kb.id)
        db.add(conv)
        db.commit()

        query_log = QueryLog(
            query="Hello",
            answer="World",
            workspace_id=ws.id,
            user_id=user.id,
            conversation_id=conv.id,
            kb_id=kb.id,
            mode="standard"
        )
        db.add(query_log)
        db.commit()

        feedback = QueryFeedback(
            query_id=query_log.id,
            conversation_id=conv.id,
            user_id=user.id,
            workspace_id=ws.id,
            rating="helpful"
        )
        db.add(feedback)
        db.commit()

        # 1. Test document deletion
        success_doc = delete_document(ws.id, doc.id, db)
        assert success_doc is True
        assert db.query(Document).filter(Document.id == doc.id).first() is None
        assert db.query(Chunk).filter(Chunk.document_id == doc.id).count() == 0

        # 2. Test knowledge base cascade deletion
        # delete_knowledge_base_cascade should nullify kb_id on conversation and query, and delete KB without error
        success_kb = delete_knowledge_base_cascade(ws.id, kb.id, db)
        assert success_kb is True
        assert db.query(KnowledgeBase).filter(KnowledgeBase.id == kb.id).first() is None

        # Verify conversation and queries survived with kb_id dissociated
        db.refresh(conv)
        db.refresh(query_log)
        assert conv.kb_id is None
        assert query_log.kb_id is None
        assert db.query(QueryFeedback).filter(QueryFeedback.id == feedback.id).first() is not None

        # 3. Test conversation deletion cascades to query and feedback
        db.delete(conv)
        db.commit()
        assert db.query(Conversation).filter(Conversation.id == conv.id).first() is None
        assert db.query(QueryLog).filter(QueryLog.id == query_log.id).first() is None
        assert db.query(QueryFeedback).filter(QueryFeedback.id == feedback.id).first() is None

    finally:
        db.close()
        eng.dispose()


def test_wal_mode_concurrent_readers_and_writers():
    """Verify that WAL mode permits concurrent reads while a write transaction is open."""
    eng, db_path, temp_dir = create_temp_persistent_engine()
    Session = sessionmaker(bind=eng)

    # Insert initial workspace
    db_setup = Session()
    ws = Workspace(name="Concurrency WS")
    db_setup.add(ws)
    db_setup.commit()
    db_setup.close()

    read_results = []
    read_exception = None

    def read_worker():
        nonlocal read_exception
        try:
            db_reader = Session()
            workspaces = db_reader.query(Workspace).all()
            read_results.append(len(workspaces))
            db_reader.close()
        except Exception as e:
            read_exception = e

    # Open a write transaction in Session 1
    db_writer = Session()
    try:
        new_ws = Workspace(name="Uncommitted WS")
        db_writer.add(new_ws)
        db_writer.flush()  # Flushes SQL to DB without committing

        # Concurrently read from another session (WAL mode should not block this read)
        t = threading.Thread(target=read_worker)
        t.start()
        t.join(timeout=5.0)

        assert read_exception is None, f"Reader was blocked or failed: {read_exception}"
        assert len(read_results) == 1
        # Reader in WAL mode sees the committed snapshot (1 workspace), not the uncommitted one
        assert read_results[0] == 1

        db_writer.commit()
    finally:
        db_writer.close()
        eng.dispose()


def test_busy_timeout_handles_competing_writers():
    """Verify that busy timeout allows a competing writer to wait and succeed once the prior writer commits."""
    eng, db_path, temp_dir = create_temp_persistent_engine()
    Session = sessionmaker(bind=eng)

    writer2_success = False
    writer2_error = None

    def writer2_task():
        nonlocal writer2_success, writer2_error
        try:
            # Short sleep to ensure writer 1 has started its transaction
            time.sleep(0.05)
            db2 = Session()
            ws2 = Workspace(name="Writer 2 WS")
            db2.add(ws2)
            db2.commit()
            db2.close()
            writer2_success = True
        except Exception as e:
            writer2_error = e

    db1 = Session()
    try:
        ws1 = Workspace(name="Writer 1 WS")
        db1.add(ws1)
        db1.flush()

        t2 = threading.Thread(target=writer2_task)
        t2.start()

        # Hold the lock briefly, then commit
        time.sleep(0.1)
        db1.commit()

        t2.join(timeout=5.0)

        assert writer2_error is None, f"Writer 2 failed: {writer2_error}"
        assert writer2_success is True, "Writer 2 should have succeeded after waiting for busy timeout"
    finally:
        db1.close()
        eng.dispose()


def test_existing_database_integrity():
    """Verify that existing database adaptive_rag.db passes SQLite integrity and foreign key checks."""
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "adaptive_rag.db")
    if not os.path.exists(db_path):
        pytest.skip("adaptive_rag.db not present in workspace")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        # Check SQLite integrity
        cur.execute("PRAGMA integrity_check;")
        integrity = cur.fetchall()
        assert integrity == [("ok",)], f"Integrity check failed: {integrity}"

        # Check SQLite foreign keys
        cur.execute("PRAGMA foreign_key_check;")
        fk_violations = cur.fetchall()
        assert fk_violations == [], f"Existing database has foreign key violations: {fk_violations}"
    finally:
        conn.close()
