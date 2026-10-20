# AdaptiveRAG — End-to-End User Journey

> Version: 1.0 | Audience: Engineering, QA, Design

This document maps the complete user journey from first visit through deletion of documents and account logout. Each step identifies the current implementation status, the happy path, error paths, and the acceptance criterion that defines "done."

---

## Journey Map Overview

```
Register → Login → Create KB → Upload Docs → Ask Question → Read Citations
    → Continue Conversation → Give Feedback → Delete Doc → Logout
```

---

## Step 0: First Visit (Unauthenticated)

**Entry point:** `GET /`

**What happens:** The user is redirected to `/login` (via `401` handler in `main.py:78`).

**Acceptance criterion:** Any unauthenticated request to any protected route redirects to `/login` with HTTP 302, not 401 JSON.

**Verified:** ✅ `app/main.py:77-79`

---

## Step 1: Registration

**Route:** `POST /register`
**File:** `app/api/routes_auth.py:31-57`

### Happy Path
1. User navigates to `/register`.
2. Enters a unique username and a password ≥ 8 characters.
3. System creates: `Workspace` record → `User` record with bcrypt hash → session.
4. Redirected to `/` (dashboard).

### Error Paths

| Condition | Current Behaviour | Acceptance Criterion |
|---|---|---|
| Username already exists | Shows error in template | ✅ Passes |
| Password < 6 chars | Rejected | ⚠️ Must enforce ≥8 chars |
| Username < 3 chars | **Not validated** | ❌ Must add check |
| Empty username | FastAPI 422 before handler | ✅ Acceptable |

### Security Notes
- Session is set immediately after registration — user is logged in. ✅
- Password stored as bcrypt hash. ✅
- No email verification — acceptable for internal MVP.

---

## Step 2: Login

**Route:** `POST /login`
**File:** `app/api/routes_auth.py:18-25`

### Happy Path
1. User enters credentials.
2. bcrypt verification passes.
3. `user_id` written to encrypted session cookie.
4. Redirected to `/`.

### Error Paths

| Condition | Current Behaviour | Criterion |
|---|---|---|
| Wrong password | Generic error message (no username leak) | ✅ |
| Unknown username | Generic error message | ✅ |
| Session expired | Redirect to `/login` | ✅ (`get_current_user`) |

---

## Step 3: Create a Knowledge Base

**Route:** `POST /knowledge-bases`
**File:** `app/api/routes_documents.py:86-110`

### Happy Path
1. User clicks "New Knowledge Base", enters a name.
2. System creates a `KnowledgeBase` record scoped to `workspace_id`.
3. Returns to documents page with the new KB selected.

### Error Paths

| Condition | Behaviour | Criterion |
|---|---|---|
| Duplicate KB name (same workspace) | Shown error | ✅ Case-insensitive check |
| Name > 100 chars | Shown error | ✅ |
| Cross-workspace KB creation | 403 via `validate_workspace` | ✅ |

---

## Step 4: Upload a Document

**Route:** `POST /documents/upload`
**File:** `app/api/routes_documents.py:159-220`

### Happy Path
1. User selects a PDF, TXT, or MD file ≤ 10 MB.
2. System streams file, validates format and size.
3. File written to `data/documents/{workspace_id}/{uuid}{ext}`.
4. `Document` record created (status: PENDING → PROCESSING → READY).
5. Text extracted → chunked → embedded → added to FAISS + BM25 in-memory index.
6. Page shows the document with status "READY".

### Error Paths

| Condition | Behaviour | Criterion |
|---|---|---|
| Unsupported format | Error before write | ✅ |
| File > 10 MB | Error after streaming limit | ✅ |
| Empty file | Error | ✅ |
| Malformed PDF header | Error | ✅ (`content.startswith(b"%PDF-")`) |
| Non-UTF-8 TXT | Error | ✅ |
| Ingestion failure (parse error) | Document set to FAILED | ✅ |
| **Server restart** | **FAISS index lost** | ❌ BLOCKER — index rebuilt from DB on startup via `reindex_all` ✅ (mitigated) |

### Index Rebuild on Restart
`main.py:lifespan` calls `reindex_all(db)` on startup, which reads all READY chunks from the DB and rebuilds FAISS/BM25 indices in memory. **This is the only index recovery mechanism.** No FAISS binary is persisted to disk. This means:
- Restart time scales with total chunk count.
- Index state is consistent with the DB (reliable).
- Binary index persistence is a post-MVP optimization.

---

## Step 5: Ask a Question (Standard Mode)

**Route:** `POST /query`
**File:** `app/api/routes_query.py:90-433`

### Happy Path
1. User types a question, selects a KB, clicks Submit.
2. System fetches workspace's active `Configuration` from DB.
3. Pipeline runs: embed query → FAISS search → BM25 search → hybrid fusion → top-K → generate.
4. Answer displayed with numbered citations (filename, excerpt, page).
5. `QueryLog` record saved with `retrieval_config` = actual config used.

### Error Paths

| Condition | Behaviour | Criterion |
|---|---|---|
| Empty KB | Clear message, no LLM call | ✅ |
| No relevant chunks found | "No relevant information" message | ✅ |
| LLM provider down | PROVIDER_FAILURE surfaced clearly | ✅ |
| Cross-workspace chunk in results | Filtered out by FK + workspace check | ✅ |
| Invalid mode parameter | 400 response | ✅ |

### What the User Sees
```
Question: What is the authentication mechanism?

Answer: The system uses JWT tokens issued after bcrypt password verification...

Sources:
  [1] auth_guide.txt — "...tokens are signed with RS256..."
  [2] deployment.txt, Page 4 — "...token expiry is configurable..."
```

---

## Step 6: Verify a Citation

**Route:** `GET /chunks/{chunk_id}`
**File:** `app/api/routes_query.py:32-50`

### Happy Path
1. User clicks a citation to inspect the full chunk text.
2. System verifies `chunk.workspace_id == user.workspace_id`.
3. Returns full chunk text, document name, page number.

**Acceptance criterion:** A user cannot retrieve chunks from another workspace via this endpoint. Test: send a valid chunk_id from workspace B while authenticated as workspace A — expect 404.

---

## Step 7: Continue in a Conversation

**Route:** `POST /conversations`, `POST /query` (with `conversation_id`)
**File:** `routes_conversations.py`, `routes_query.py:101-112`

### Happy Path
1. User creates a named conversation.
2. Each query submitted with `conversation_id` links the query log to the thread.
3. Conversation detail view shows all turns in order.

**Persistence criterion:** After restarting the browser and re-authenticating, all conversation turns are visible — they are stored in SQLite, not browser state.

---

## Step 8: Submit Answer Feedback

**Route:** `POST /conversations/{id}/feedback/{query_id}`
**File:** `routes_conversations.py:350-420`

### Happy Path
1. User clicks "Helpful" or "Not Helpful".
2. `QueryFeedback` record created or updated (upsert on `query_id`).
3. Optional reason selected from allowed set.

### Error Paths

| Condition | Behaviour | Criterion |
|---|---|---|
| Cross-user feedback | 403 | ✅ Ownership check present |
| Invalid rating | Rejected | ✅ `ALLOWED_RATINGS` check |
| Invalid reason | Rejected | ✅ `ALLOWED_REASONS` check |
| Duplicate feedback | Update existing | ✅ Upsert |

---

## Step 9: Delete a Document

**Route:** `POST /documents/{doc_id}/delete`
**File:** `routes_documents.py:348-356`, `app/rag/ingestion.py`

### Happy Path
1. User clicks Delete on a document.
2. System verifies ownership (workspace match).
3. Chunks deleted from DB → index rebuilt for that KB → physical file removed.
4. Existing conversation views: citations referencing deleted chunks show "source unavailable."

**Acceptance criterion:** Deleting a document and then visiting an existing conversation that cited it must not raise a 500 error. Citation should show a graceful "unavailable" state.

**Verified:** `routes_conversations.py` includes deleted-source handling in conversation detail view.

---

## Step 10: Logout

**Route:** `GET /logout` or `POST /logout`
**File:** `routes_auth.py:59-63`

### Happy Path
1. User clicks Logout.
2. Session cleared.
3. Redirected to `/login`.
4. Back-button or direct URL access to `/` redirects to `/login`.

**Acceptance criterion:** After logout, any previously valid session cookie does not grant access — Starlette session middleware cryptographically signs cookies; clearing the session invalidates it.

---

## Known Journey Gaps (Launch Blockers)

| Gap | Impact | Where |
|---|---|---|
| No CSRF on any form | Any website can submit forms on behalf of a logged-in user | All POST routes |
| Password minimum = 6 chars | Weak accounts | `routes_auth.py:37` |
| No username length minimum | Trivial usernames | `routes_auth.py` |
| No rate limiting on login | Brute force possible | `POST /login` |
| Upload file stored before validation complete | Partial disk writes on error | `routes_documents.py:208-213` (mitigated — exception handling deletes file) |
| No `Retry-After` on LLM failures | No guidance to user | `pipeline.py` |

---

*Created: October 2026*
