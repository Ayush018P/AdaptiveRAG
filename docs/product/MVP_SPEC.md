# AdaptiveRAG — MVP Product Specification

> Version: 1.0 | Status: Draft | Audience: Engineering + Product

---

## 1. Target Customer

**Primary customer:** Knowledge workers and small internal teams (5–50 people) who maintain documentation — policy documents, technical runbooks, onboarding guides, API references — and need to answer questions grounded in that documentation without searching manually.

**Concrete persona:**

| Attribute | Detail |
|---|---|
| Role | Support engineer, technical writer, internal tools team |
| Pain | Searching PDFs and wikis manually to answer recurring questions |
| Environment | Internal corporate tools; security-conscious |
| Technical comfort | Non-developer; can use a web form |
| Success metric | Gets an accurate, cited answer in under 10 seconds instead of searching for 5 minutes |

**What they are NOT:** Research engineers running benchmark experiments, developers integrating via API, or teams requiring real-time document ingestion.

---

## 2. Problem Statement

Internal teams waste time re-reading the same documentation to answer questions that are already answered somewhere in their knowledge base. Existing tools (search, wikis) return documents — not answers. LLM chatbots hallucinate without grounding. **AdaptiveRAG's core value:** ground every answer in the user's own uploaded documents and cite the exact source, so the answer is both fast and verifiable.

---

## 3. MVP Scope — What We Are Building

The MVP is a **multi-user, workspace-isolated, document-grounded Q&A tool** with verifiable citations. Adaptive configuration is a background capability; it must not be exposed as a configuration surface to MVP users.

### In Scope

| # | Feature | Justification |
|---|---|---|
| 1 | User registration and login | Without identity, no isolation |
| 2 | One workspace per user, strict data isolation | Core security requirement |
| 3 | Upload PDF, TXT, MD documents (≤10 MB each) | Primary content surface |
| 4 | Organize documents into named Knowledge Bases | Multi-context support |
| 5 | Standard query mode: ask a question, receive a cited answer | Core product value |
| 6 | Citations with source filename, page number, and excerpt | Verifiability — the differentiator |
| 7 | Conversation history: multi-turn threads that persist across sessions | User retention and context |
| 8 | Helpful/Not helpful feedback per answer | Improvement signal |
| 9 | Delete a document and confirm answer citations become unavailable | Data integrity |
| 10 | Logout and session expiry | Basic security |

### Explicitly Out of Scope for MVP

- Adaptive mode exposure to end users (runs transparently; trace not shown to end users)
- Experiment builder UI
- FAISS/BM25 index export or import
- Multi-user admin panel
- API key management UI
- Rate limiting per user (deferred to hardening phase)
- Email verification or password reset
- Docker or cloud deployment configuration (operational concern)
- Real-time streaming answers

---

## 4. Existing Architecture — Verified from Source

> Claims verified by reading actual files, not prior completion reports.

| Component | Status | Verified Location |
|---|---|---|
| FastAPI + Starlette | IMPLEMENTED | `app/main.py` |
| bcrypt auth + session | IMPLEMENTED | `app/core/auth.py`, `routes_auth.py` |
| Workspace isolation via FK + query filter | IMPLEMENTED | `auth.py:validate_workspace` |
| Document upload: PDF, TXT, MD | IMPLEMENTED | `routes_documents.py:upload_document` |
| 10 MB file size check | IMPLEMENTED | `routes_documents.py:184` |
| Chunking + FAISS + BM25 | IMPLEMENTED | `app/rag/ingestion.py`, `vector_store.py` |
| Standard query + citations | IMPLEMENTED | `routes_query.py`, `pipeline.py` |
| Config propagation to pipeline | IMPLEMENTED (P1-Correctness) | `pipeline.py:config param` |
| Conversation history | IMPLEMENTED | `routes_conversations.py`, `models/conversation.py` |
| Feedback model | IMPLEMENTED | `models/feedback.py` |
| SQLite WAL + FK enforcement | IMPLEMENTED (P2.3) | `models/database.py` |
| Secret key from env | IMPLEMENTED (P2.2) | `app/core/config.py` |
| **FAISS indices in-memory only** | **GAP** | `vector_store.py` — no `faiss.write_index` |
| **CSRF protection** | **ABSENT** | No middleware or token in any form |
| **Rate limiting** | **ABSENT** | No middleware |
| Password minimum = 6 chars | **WEAK** | `routes_auth.py:37` |
| No `docs/` directory | **ABSENT** | Created now |

---

## 5. Functional Requirements with Acceptance Criteria

### F-1: Registration and Login

**Acceptance Criteria:**
- Username < 3 characters rejected with clear message.
- Password < 8 characters rejected. *(Currently only enforces ≥6 — requires fix.)*
- Duplicate username returns error without revealing whether account exists.
- After logout, session cookie cannot access protected pages.
- Session cookies carry `HttpOnly` and `SameSite=Lax`.

**Files:** `routes_auth.py`, `app/core/auth.py`
**Test:** `tests/test_auth.py` — add password length regression

---

### F-2: Document Upload and Management

**Acceptance Criteria:**
- Only `.pdf`, `.txt`, `.md` accepted; others rejected before write.
- File > 10 MB rejected after streaming, before write.
- Uploading duplicate filename in same KB atomically replaces old document.
- Deleting a document triggers index rebuild; subsequent queries do not return deleted chunks.
- FAILED documents show retry option.

**Files:** `routes_documents.py`, `app/rag/ingestion.py`
**Test:** `tests/test_ingestion_security.py`, `tests/test_knowledge_base.py`

---

### F-3: Standard Query and Citations

**Acceptance Criteria:**
- Each citation has: `filename`, `excerpt` (≤240 chars), `page_number` (PDF only).
- No chunk from another workspace appears in any result set.
- Empty KB returns "no documents" message — not an LLM answer.
- LLM failure surfaces an error; no partial answer shown as complete.
- `retrieval_config` in query log records the config actually used (not global defaults).

**Files:** `routes_query.py`, `pipeline.py`, `hybrid_retriever.py`
**Test:** `tests/test_query_evidence.py`, `tests/test_config_propagation.py`

---

### F-4: Conversation History

**Acceptance Criteria:**
- Multi-turn conversation persists across full browser restart.
- User A cannot read, rename, delete, or submit feedback against User B's conversation (403).
- Deleted conversation removes associated query logs.

**Files:** `routes_conversations.py`, `models/conversation.py`
**Test:** `tests/test_conversations_and_feedback.py`

---

### F-5: Answer Feedback

**Acceptance Criteria:**
- Submitting feedback twice for the same answer updates existing record (upsert).
- Cross-user feedback attempt returns 403.
- Reason validated against allowed set: `{incorrect_answer, insufficient_evidence, irrelevant_response, citation_problem, other}`.

**Files:** `routes_conversations.py:350–420`, `models/feedback.py`

---

## 6. Non-Functional Requirements

| Requirement | MVP Target | Note |
|---|---|---|
| Response time | < 10s Standard mode, < 50 docs | Synchronous embedding |
| Concurrent users | 5–10 | SQLite WAL (P2.3 complete) |
| Uptime | Manual restart acceptable | No automated restart yet |
| Data durability | SQLite WAL + nightly backup script | FAISS rebuilds from DB on restart |
| Exposure | Internal LAN / localhost only | CSRF required before any public URL |

---

## 7. Definition of MVP Complete

1. New user registers → uploads a PDF → asks a question → receives a cited answer → conversation is visible in a new browser session.
2. Two separate accounts cannot read each other's documents, answers, or conversations.
3. Deleting a document does not crash existing conversation views; citations are marked unavailable.
4. All test files pass: `python -m pytest` with zero failures.
5. Application starts from clean checkout using only `.env` — no manual DB edits required.
6. CSRF tokens are present on all state-changing forms.

---

*Created: October 2026*
