# AdaptiveRAG — Production Acceptance Checklist

> Version: 1.0 | Status: Pre-launch audit
> Purpose: Gate criteria before any real-user pilot. Each item must be checked off with evidence, not assumed.

---

## How to Use This Checklist

- Each item has a **Priority**: `BLOCKER` (must pass before pilot), `REQUIRED` (must pass before GA), `HARDENING` (post-GA).
- Each item has a **Current Status** based on source code inspection.
- Completion requires: code evidence + passing test + reviewer sign-off.

---

## Section 1 — Security

### 1.1 CSRF Protection
**Priority:** BLOCKER
**Current Status:** ❌ ABSENT — confirmed by reading all route files; no CSRF middleware, no token generation, no token validation in any form handler.

**Why it's a blocker:** Any website can submit a POST form to `/documents/upload`, `/documents/{id}/delete`, `/query`, or `/conversations/{id}/delete` on behalf of a logged-in user. This allows cross-site request forgery against every state-changing operation.

**Implementation:**
- Add CSRF middleware (e.g., `starlette-csrf` or custom `double-submit cookie` pattern).
- Inject `csrf_token` into every Jinja2 template that has a `<form method="POST">`.
- Validate token in all POST handlers before processing.

**Relevant files:** `app/main.py` (middleware), all templates in `app/templates/`, all `routes_*.py` POST handlers.

**Acceptance test:**
```python
# A POST request to /documents/upload without a valid CSRF token
# must return 403, not 200.
def test_csrf_rejects_missing_token(client):
    response = client.post("/documents/upload", data={"file": ...})
    assert response.status_code == 403
```

**Completion criteria:** All POST forms include a CSRF token input. Server rejects any POST without a valid matching token. Automated test passes.

---

### 1.2 Password Policy
**Priority:** BLOCKER
**Current Status:** ⚠️ WEAK — `routes_auth.py:37` enforces minimum 6 characters only.

**Implementation:** Raise minimum to 8 characters. Add check for username minimum 3 characters.

**Files:** `app/api/routes_auth.py:37`

**Acceptance test:**
```python
def test_password_too_short_rejected(client):
    response = client.post("/register", data={"username": "testuser", "password": "abc123"})
    assert "too short" in response.text.lower()

def test_password_8_chars_accepted(client):
    response = client.post("/register", data={"username": "validuser", "password": "abcd1234"})
    assert response.status_code == 302  # redirect on success
```

---

### 1.3 Session Security
**Priority:** REQUIRED
**Current Status:** ✅ IMPLEMENTED — `main.py` reads `session_secret_key` from env; raises at startup if absent in production; `https_only` set by `is_cookie_secure`.

**Acceptance test:** `tests/test_secrets_and_session_security.py` — confirmed 10 tests.

---

### 1.4 Authorization on Every Resource
**Priority:** BLOCKER
**Current Status:** ✅ Mostly implemented — `validate_workspace` and `get_current_user` used in all route handlers. Chunk inspection enforces `chunk.workspace_id == user.workspace_id`.

**Gap:** Verify all resource endpoints explicitly:

| Endpoint | Ownership check | Status |
|---|---|---|
| `GET /chunks/{chunk_id}` | `chunk.workspace_id == user.workspace_id` | ✅ |
| `POST /documents/{doc_id}/delete` | `Document.workspace_id == ws_id` filter | ✅ |
| `POST /conversations/{id}/feedback/{qid}` | Conversation workspace check | ✅ |
| `GET /conversations/{id}` | Workspace filter on query | ✅ |
| `POST /knowledge-bases/{kb_id}/delete` | `validate_workspace` + KB workspace filter | ✅ |

**Acceptance test:** `tests/test_workspace_isolation.py`, `tests/test_conversations_and_feedback.py:test_cross_workspace_conversation_isolation`

---

### 1.5 No Secrets in Source / Git History
**Priority:** BLOCKER
**Current Status:** ✅ `.gitignore` excludes `.env`; `.env.example` has safe placeholders; `config.py` validates key at startup.

**Verification:**
```bash
git log --all --full-history -- .env   # must return no commits
git grep -i "groq_api_key\s*=" --and --not "= None"  # must return only .env.example
```

---

### 1.6 Rate Limiting (Login Brute Force)
**Priority:** REQUIRED
**Current Status:** ❌ ABSENT — `POST /login` has no rate limit.

**Implementation:** Add `slowapi` or a simple in-process counter middleware on `POST /login`.

**Acceptance criterion:** After 10 failed login attempts within 60 seconds from the same IP, return 429.

---

## Section 2 — Data Integrity and Persistence

### 2.1 SQLite Connection Reliability
**Priority:** REQUIRED
**Current Status:** ✅ IMPLEMENTED (P2.3) — `PRAGMA foreign_keys = ON`, `PRAGMA busy_timeout = 30000`, WAL mode on persistent DB.

**Evidence:** `app/models/database.py:6-33`
**Tests:** `tests/test_database_reliability.py` — 8 tests confirmed.

---

### 2.2 FAISS Index Persistence
**Priority:** REQUIRED
**Current Status:** ⚠️ IN-MEMORY ONLY — `vector_store.py` stores indices in a Python dict. No `faiss.write_index()` call exists anywhere in the codebase.

**What this means:**
- On server restart, `reindex_all(db)` in `main.py:lifespan` rebuilds indices from chunk text stored in SQLite. This works but scales linearly with chunk count.
- A crash mid-ingestion (chunks partially written to DB, no FAISS entry) is handled by the startup rebuild.
- **Risk:** Large knowledge bases will cause slow startup (minutes for >100k chunks).

**Mitigation for pilot:** Acceptable if knowledge base is small (< 5,000 chunks / < 50 documents). Document this limitation explicitly.

**Completion criteria for REQUIRED:** Add `faiss.write_index` after each ingestion and load at startup. A restart with 10,000 chunks should complete in < 5 seconds.

---

### 2.3 Database Backup
**Priority:** REQUIRED
**Current Status:** ❌ ABSENT — no automated backup mechanism, no `scripts/backup.sh`.

**Implementation:** Create `scripts/backup_db.sh` that runs `sqlite3 adaptive_rag.db ".backup backup/adaptive_rag_$(date +%Y%m%d).db"`.

**Acceptance criterion:** Backup script runs without error; restored database passes `PRAGMA integrity_check`.

---

### 2.4 Migration Safety
**Priority:** REQUIRED
**Current Status:** ⚠️ MANUAL — migration scripts exist in `scripts/` but are not automated. `Base.metadata.create_all()` in `main.py:24` adds new tables but does not alter existing ones.

**Risk:** Adding a new column to an existing table without an Alembic migration will silently fail for existing databases — the column will not be added, and the application will crash with `OperationalError`.

**Recommendation:** Add Alembic for schema migrations before any production deployment. Alternatively, document clearly that schema changes require manual migration scripts.

---

### 2.5 Foreign Key Cascade Correctness
**Priority:** REQUIRED
**Current Status:** ✅ IMPLEMENTED — `delete_knowledge_base_cascade` explicitly nulls `kb_id` on `Conversation` and `QueryLog` records before deleting KB.

**Evidence:** `app/rag/ingestion.py:delete_knowledge_base_cascade`
**Test:** `tests/test_database_reliability.py:test_referential_integrity_on_document_and_kb_deletion`

---

## Section 3 — Functional Correctness

### 3.1 Configuration Propagation
**Priority:** REQUIRED
**Current Status:** ✅ IMPLEMENTED (Phase 1 Correctness) — `pipeline.py` accepts explicit `config` dict; `routes_query.py` passes `current_config`; `experiment_engine.py` uses explicit config dict, not `activate_config`.

**Evidence:** `app/rag/pipeline.py:run(config=...)`, `routes_query.py:140-141`
**Tests:** `tests/test_config_propagation.py`

---

### 3.2 Workspace Data Isolation Under Concurrent Load
**Priority:** REQUIRED
**Current Status:** ✅ DB-level isolation works. FAISS uses separate in-memory dict keys per `workspace_id:kb_id`.

**Gap to verify:** Two concurrent uploads to different workspaces do not corrupt each other's FAISS indices (since `faiss_store.indices` is a shared dict with no locking).

**Acceptance test:**
```python
# Two threads simultaneously uploading to different workspaces
# Neither should retrieve chunks from the other's workspace.
def test_concurrent_uploads_isolate_indices():
    ...
```

---

### 3.3 Empty KB and Provider Failure Handling
**Priority:** REQUIRED
**Current Status:** ✅ IMPLEMENTED — early return for EMPTY_KNOWLEDGE_BASE in `pipeline.py`, PROVIDER_FAILURE status returned and displayed.

---

### 3.4 Conversation-Citation Graceful Degradation
**Priority:** REQUIRED
**Current Status:** ✅ IMPLEMENTED — conversation detail view handles missing chunks.

**Acceptance test:** Delete a document, navigate to a conversation citing it, confirm no 500 error and a "source unavailable" indicator.

---

## Section 4 — Operational Readiness

### 4.1 Health Check Endpoint
**Priority:** REQUIRED
**Current Status:** ✅ IMPLEMENTED — `GET /health` returns `{"status": "ok"}`.

**Gap:** Health check does not verify DB connectivity or index state.

**Improved acceptance criterion:**
```python
@app.get("/health")
async def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok", "db": "connected", "indices": len(faiss_store.indices)}
```

---

### 4.2 Structured Logging
**Priority:** REQUIRED
**Current Status:** ⚠️ PARTIAL — `logging_config.py` exists; `setup_logging()` called in `main.py`. Format and log level are configurable.

**Gap:** No request-level correlation IDs. Log format is not JSON (not machine-parseable).

**Acceptance criterion for pilot:** All errors logged with traceback. Log level controllable via env var `LOG_LEVEL`.

---

### 4.3 Startup Failure Modes
**Priority:** REQUIRED
**Current Status:** ✅ Application raises `RuntimeError` if `SESSION_SECRET_KEY` is missing (`main.py:51-52`). LLM provider failures surface at query time, not startup.

**Gap:** If the DB file is corrupted or missing, `create_all` will fail with a SQLAlchemy error that is not wrapped in a human-readable message.

---

### 4.4 Process Management / Auto-restart
**Priority:** HARDENING
**Current Status:** ❌ ABSENT — `run.py` starts uvicorn directly; no supervisor, no systemd unit, no Docker.

**Recommendation:** Provide a `systemd` unit file or `supervisord` config for the pilot environment.

---

### 4.5 Error Pages
**Priority:** REQUIRED
**Current Status:** ⚠️ PARTIAL — 401 redirects to login. 403, 404, 500 return FastAPI default JSON responses, not user-facing HTML pages.

**Acceptance criterion:** 403, 404, and 500 responses render HTML pages consistent with the application UI.

---

## Section 5 — LLM Cost and Reliability Controls

### 5.1 LLM Provider Timeout
**Priority:** REQUIRED
**Current Status:** ⚠️ PARTIAL — `tenacity` is in requirements.txt. Verify it is actually applied to LLM calls in `generator.py`.

**Verification needed:** Read `app/rag/generator.py` and confirm retry/timeout is applied.

**Acceptance criterion:** An LLM request that hangs returns a `PROVIDER_FAILURE` response within 30 seconds, not an indefinite hang.

---

### 5.2 LLM Cost Visibility
**Priority:** HARDENING
**Current Status:** ⚠️ PARTIAL — `llm_calls` and `token_usage` are stored in `QueryLog`. Not visible on any dashboard.

**Recommendation:** Add a simple LLM call counter to the dashboard for pilot monitoring.

---

### 5.3 Prompt Injection Awareness
**Priority:** HARDENING
**Current Status:** ⚠️ No input sanitization before query is placed in prompt. Adversarial users could attempt prompt injection via document content.

**Recommendation:** Document known risk; implement context escaping in generator prompt template.

---

## Section 6 — Test Coverage

### 6.1 Current Test Status
| File | Tests | Last Known Result |
|---|---|---|
| `test_auth.py` | 4 | Pass |
| `test_conversations_and_feedback.py` | 14 | Pass |
| `test_database_reliability.py` | 8 | Pass |
| `test_query_evidence.py` | 10 | Pass |
| `test_query_modes.py` | 12 | Pass |
| `test_config_propagation.py` | 16+ | Added Phase 1 Correctness |
| `test_workspace_isolation.py` | 3 | Pass |
| `test_secrets_and_session_security.py` | 10 | Pass |
| `test_ingestion_security.py` | 9 | Pass |
| `test_knowledge_base.py` | 11 | Pass |
| **TOTAL** | **109+** | Must verify fresh run |

### 6.2 Missing Tests (Required Before Pilot)
| Missing Test | Priority |
|---|---|
| CSRF token validation (when implemented) | BLOCKER |
| Concurrent upload isolation | REQUIRED |
| Password minimum 8 chars enforcement | REQUIRED |
| Health check DB connectivity | REQUIRED |
| LLM timeout returns PROVIDER_FAILURE | REQUIRED |

---

## Sign-Off Checklist

| Item | Status | Evidence | Reviewer |
|---|---|---|---|
| 1.1 CSRF on all POST forms | ❌ | | |
| 1.2 Password ≥ 8 chars | ❌ | | |
| 1.4 Authorization on all endpoints | ✅ | Source audit | |
| 1.5 No secrets in git | ✅ | `.gitignore` | |
| 2.1 SQLite WAL + FK enforcement | ✅ | `database.py`, 8 tests | |
| 2.2 Index rebuild on restart verified | ✅ | `main.py:lifespan` | |
| 2.3 Backup script | ❌ | | |
| 3.1 Config propagation | ✅ | `pipeline.py`, tests | |
| 4.1 Health check | ⚠️ | Needs DB check | |
| 4.2 Structured logging | ⚠️ | Partial | |
| 6.1 All tests pass fresh | ❌ | Run `pytest` | |

**Gate:** All BLOCKER items must be ✅ before any real-user pilot begins.

---

*Created: October 2026*
