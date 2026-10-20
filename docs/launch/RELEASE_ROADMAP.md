# AdaptiveRAG — Release Roadmap

> Version: 1.0 | Status: Draft
> Horizon: From current state to controlled pilot with real users

---

## Current Baseline (Verified October 2026)

| Capability | Status |
|---|---|
| FastAPI + SQLite, session auth, workspace isolation | ✅ Working |
| Upload, chunk, embed, retrieve, generate, cite | ✅ Working |
| Config propagation (active workspace config → pipeline) | ✅ Fixed (Phase 1 Correctness) |
| Experiment isolation (no global state mutation) | ✅ Fixed (Phase 1 Correctness) |
| SQLite WAL + FK enforcement | ✅ Implemented (P2.3) |
| Secrets in env, not source | ✅ Implemented (P2.2) |
| 109 tests passing | ✅ Confirmed |
| CSRF protection | ❌ Absent |
| Rate limiting | ❌ Absent |
| FAISS persistence to disk | ❌ In-memory only |
| Database backup automation | ❌ Absent |
| Alembic / schema migration | ❌ Absent |
| Docker / deployment config | ❌ Absent |
| Error pages (403, 404, 500) | ⚠️ JSON only |
| Password policy ≥ 8 chars | ⚠️ Only enforces ≥6 |

---

## Release Phases

---

### Phase R1 — Security Baseline (Launch Blocker)
**Target:** Complete before any real-user access
**Estimated effort:** 3–5 days

| Task | Priority | Files | Acceptance Criterion |
|---|---|---|---|
| **R1.1** Add CSRF middleware and tokens to all POST forms | BLOCKER | `main.py`, all templates, all POST routes | POST without token → 403 |
| **R1.2** Enforce password minimum 8 chars | BLOCKER | `routes_auth.py:37` | Password < 8 chars → error, test passes |
| **R1.3** Enforce username minimum 3 chars | BLOCKER | `routes_auth.py` | Username < 3 → error, test passes |
| **R1.4** Add login rate limiting (10 req / 60s / IP) | REQUIRED | `main.py` middleware or `routes_auth.py` | 11th attempt → 429 within window |

**Completion gate:** All BLOCKER items in `PRODUCTION_ACCEPTANCE_CHECKLIST.md` Section 1 checked off with test evidence.

---

### Phase R2 — Data Reliability
**Target:** Complete before pilot; can run in parallel with R1
**Estimated effort:** 2–3 days

| Task | Priority | Files | Acceptance Criterion |
|---|---|---|---|
| **R2.1** Database backup script | REQUIRED | `scripts/backup_db.sh` | Script runs; restored DB passes `PRAGMA integrity_check` |
| **R2.2** Improve health check (DB + index state) | REQUIRED | `app/main.py:/health` | `/health` returns DB connected + index count |
| **R2.3** Structured error pages (403, 404, 500) | REQUIRED | `app/main.py`, templates | Non-JSON HTML response for browser clients |
| **R2.4** Document FAISS startup rebuild limitation | REQUIRED | `docs/`, `README.md` | Known limitation documented; rebuild tested with 5,000 chunks |

**Completion gate:** All REQUIRED items in Checklist Section 2 and 4 checked off.

---

### Phase R3 — Controlled Pilot
**Target:** 3–5 internal users, limited KB size, monitored
**Estimated effort:** 1–2 days setup + monitoring

| Task | Priority | Notes |
|---|---|---|
| **R3.1** Run `python -m pytest` fresh, confirm 0 failures | REQUIRED | Record exact count |
| **R3.2** Deploy to internal host (LAN only, not public) | REQUIRED | Document host, port, backup schedule |
| **R3.3** Verify startup from clean checkout | REQUIRED | Fresh Python venv, `.env` from `.env.example` |
| **R3.4** Pilot user onboarding guide | REQUIRED | How to register, upload, query, give feedback |
| **R3.5** Set KB size limit guidance | REQUIRED | Recommend < 50 documents / user for pilot |
| **R3.6** Monitor logs for errors during pilot | REQUIRED | Check `LOG_LEVEL=INFO` output daily |
| **R3.7** Collect feedback after 5 sessions | REQUIRED | Rate answer quality, report citation accuracy |

**Pilot exit criteria:**
- Zero data isolation failures (no cross-workspace data visible)
- < 5% of queries result in a 500 error
- Average response time < 10 seconds
- At least 3 users complete the full journey (register → upload → query → feedback)

---

### Phase R4 — Post-Pilot Hardening
**Target:** After successful pilot; before broader rollout
**Estimated effort:** 1–2 weeks

| Task | Priority | Rationale |
|---|---|---|
| **R4.1** FAISS index persistence to disk | REQUIRED | Eliminate slow startup at scale |
| **R4.2** Alembic schema migration | REQUIRED | Safe schema evolution without manual scripts |
| **R4.3** Async document ingestion (background task) | REQUIRED | Uploads > 1 MB block the event loop |
| **R4.4** LLM timeout and retry (verify tenacity applied) | REQUIRED | Prevent indefinite hangs |
| **R4.5** Per-user storage quota | REQUIRED | Prevent abuse in multi-tenant environment |
| **R4.6** Process supervisor (systemd / supervisord) | REQUIRED | Auto-restart on crash |
| **R4.7** Log aggregation (file rotation or external sink) | REQUIRED | Operational visibility |
| **R4.8** Concurrent upload isolation test | REQUIRED | Shared FAISS dict write safety |
| **R4.9** Prompt injection documentation and mitigations | HARDENING | Risk management |
| **R4.10** Cost monitoring dashboard for LLM calls | HARDENING | Token/cost visibility |

---

### Phase R5 — Scale and Differentiation (Post-GA)
**Target:** After stable pilot; product growth phase
**Estimated effort:** 4–8 weeks

| Task | Rationale |
|---|---|
| Adaptive mode as optional user setting | Expose differentiated capability to power users |
| Multi-document reasoning (cross-KB queries) | Broaden Q&A capability |
| PostgreSQL migration | Eliminate SQLite write contention at scale |
| REST API for programmatic access | Integration with external tools |
| Email verification + password reset | Enterprise readiness |
| SSO / SAML integration | Enterprise customer requirement |
| Answer quality analytics dashboard | Show value; drive product decisions |
| Admin panel (user management, usage stats) | Operational self-service |

---

## Implementation Order — First Tasks

The following is the strict recommended sequence. Each task is independently verifiable before the next begins.

### Task 1 (Smallest, First): Password Policy Fix
**Why first:** Zero risk, zero dependencies, closes a BLOCKER in 30 minutes.

```python
# routes_auth.py — change line 37
if len(password) < 8:  # was < 6
    ...
# Add username check:
if len(username.strip()) < 3:
    return templates.TemplateResponse("register.html",
        {"request": request, "error": "Username must be at least 3 characters."})
```

**Verification:**
```bash
python -m pytest tests/test_auth.py -v
# Add: test_password_7_chars_rejected, test_username_2_chars_rejected
```

**Done when:** Tests pass, BLOCKER 1.2 and 1.3 checked in the acceptance checklist.

---

### Task 2: CSRF Protection
**Why second:** Highest security impact; required before any exposure beyond localhost.

**Approach:** Use `itsdangerous` (already a transitive dep via Starlette) to generate signed tokens stored in session and validated in each POST handler. Or install `starlette-csrf`.

**Files to change:**
1. `app/main.py` — add CSRF middleware
2. `app/templates/base.html` — inject `{{ csrf_token }}` into all forms
3. All POST route handlers — validate token before processing

**Verification:** `python -m pytest tests/test_csrf.py -v` (new test file)

---

### Task 3: Database Backup Script
**Why third:** Zero code risk; pure operational safety net.

```bash
# scripts/backup_db.sh
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p backup
sqlite3 adaptive_rag.db ".backup backup/adaptive_rag_${DATE}.db"
echo "Backup complete: backup/adaptive_rag_${DATE}.db"
```

**Verification:** Script runs; `sqlite3 backup/adaptive_rag_*.db "PRAGMA integrity_check;"` returns `ok`.

---

### Task 4: Improved Health Check
**Why fourth:** Required before any deployment; low risk, high value.

```python
@app.get("/health")
async def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "db": "connected",
        "faiss_indices": len(faiss_store.indices),
        "version": "1.0.0"
    }
```

---

### Task 5: Error Pages (403, 404, 500)
**Why fifth:** Required for user-facing deployment; straightforward HTML templates.

---

### Task 6: Run Full Test Suite, Fix Any Failures
**Why sixth:** Gate before pilot begins.

```bash
python -m pytest --tb=short -q
```

Expected: 109+ tests, 0 failures. If any fail, fix before proceeding.

---

## Success Metrics for Pilot

| Metric | Target | Measurement |
|---|---|---|
| Data isolation failures | 0 | Manual test + workspace isolation tests |
| 500 error rate | < 5% of queries | Application logs |
| Average query latency | < 10s | `QueryLog.latency` field |
| Feedback helpfulness rate | > 60% "Helpful" | `QueryFeedback` table aggregate |
| User-reported citation accuracy | > 70% | Pilot feedback form |

---

*Created: October 2026 | Review before Phase R3*
