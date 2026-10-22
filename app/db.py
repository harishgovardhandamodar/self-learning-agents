"""
SQLite storage for the Self-Learning Agent Lab.

Everything the lab produces is a record: a design (an agent spec), a run
(stream x evolver x policy), the checkpoint metrics, and the mechanism
diagnostics that explain *why* a run behaved the way it did.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = os.environ.get("SLA_DB", str(Path(__file__).resolve().parent.parent / "data" / "sla.db"))
Path(os.path.dirname(DB_PATH)).parent.mkdir(parents=True, exist_ok=True)

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS designs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    domain      TEXT DEFAULT 'general',
    description TEXT DEFAULT '',
    spec        TEXT NOT NULL,          -- full harness + config JSON
    tags        TEXT DEFAULT '[]',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    design_id   TEXT,
    name        TEXT DEFAULT '',
    stream_id   TEXT NOT NULL,
    evolver    TEXT NOT NULL,
    policy     TEXT NOT NULL,          -- 'simulated' | 'live'
    model       TEXT DEFAULT '',
    status      TEXT DEFAULT 'pending', -- pending|running|done|error|cancelled
    progress    REAL DEFAULT 0,
    message     TEXT DEFAULT '',
    config      TEXT NOT NULL,
    result      TEXT,                   -- metrics JSON (on completion)
    error       TEXT DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);

CREATE TABLE IF NOT EXISTS checkpoints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    t           INTEGER NOT NULL,       -- stream position of the checkpoint
    label       TEXT DEFAULT '',
    kind        TEXT DEFAULT 'eval',    -- 'eval' | 'seed'
    metrics     TEXT NOT NULL,          -- per-group accuracy JSON
    diagnostics TEXT DEFAULT '{}',      -- mechanism diagnostics JSON
    snapshot    TEXT DEFAULT '{}',      -- harness snapshot JSON
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ckpt_run ON checkpoints(run_id, t);

CREATE TABLE IF NOT EXISTS traces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    t           INTEGER NOT NULL,
    group_id    TEXT DEFAULT '',
    kind        TEXT DEFAULT 'episode', -- episode|update|eval
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_run ON traces(run_id, t);

CREATE TABLE IF NOT EXISTS streams (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    scenario    TEXT DEFAULT 'custom',
    config      TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS landscape_entries (
    id          TEXT PRIMARY KEY,
    kind        TEXT DEFAULT 'note',     -- paper|product|framework|writeup|blog|note
    title       TEXT NOT NULL,
    authors     TEXT DEFAULT '',
    venue       TEXT DEFAULT '',
    year        INTEGER,
    url         TEXT DEFAULT '',
    summary     TEXT DEFAULT '',
    mechanism   TEXT DEFAULT '',
    align       TEXT DEFAULT '{}',        -- JSON: this lab's alignment
    concepts    TEXT DEFAULT '[]',        -- JSON array of concept ids
    patterns    TEXT DEFAULT '[]',        -- JSON array of pattern ids
    source      TEXT DEFAULT 'local',     -- where it came from
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_landscape_created ON landscape_entries(created_at DESC);
"""


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        _local.conn = conn
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()


# ------------------------------------------------------------------ designs

def save_design(design_id: str, name: str, domain: str, description: str,
                spec: Dict[str, Any], tags: Optional[List[str]] = None) -> Dict[str, Any]:
    conn = get_conn()
    now = time.time()
    conn.execute(
        """INSERT INTO designs (id, name, domain, description, spec, tags, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET
             name=excluded.name, domain=excluded.domain, description=excluded.description,
             spec=excluded.spec, tags=excluded.tags, updated_at=excluded.updated_at""",
        (design_id, name, domain, description, json.dumps(spec),
         json.dumps(tags or []), now, now),
    )
    conn.commit()
    return get_design(design_id)


def get_design(design_id: str) -> Optional[Dict[str, Any]]:
    row = get_conn().execute("SELECT * FROM designs WHERE id=?", (design_id,)).fetchone()
    return _design_row(row) if row else None


def list_designs() -> List[Dict[str, Any]]:
    rows = get_conn().execute(
        "SELECT * FROM designs ORDER BY updated_at DESC").fetchall()
    return [_design_row(r) for r in rows]


def delete_design(design_id: str) -> bool:
    conn = get_conn()
    cur = conn.execute("DELETE FROM designs WHERE id=?", (design_id,))
    conn.commit()
    return cur.rowcount > 0


def _design_row(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["spec"] = json.loads(d.get("spec") or "{}")
    d["tags"] = json.loads(d.get("tags") or "[]")
    return d


# --------------------------------------------------------------------- runs

def create_run(run_id: str, design_id: str, name: str, stream_id: str, evolver: str,
               policy: str, model: str, config: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_conn()
    now = time.time()
    conn.execute(
        """INSERT INTO runs (id, design_id, name, stream_id, evolver, policy, model,
                             status, progress, message, config, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, design_id, name, stream_id, evolver, policy, model, "pending", 0.0,
         "queued", json.dumps(config), now, now),
    )
    conn.commit()
    return get_run(run_id)


def update_run(run_id: str, **fields) -> None:
    if not fields:
        return
    conn = get_conn()
    cols = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE runs SET {cols}, updated_at=? WHERE id=?",
                 list(fields.values()) + [time.time(), run_id])
    conn.commit()


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    row = get_conn().execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return _run_row(row) if row else None


def delete_run(run_id: str) -> bool:
    """Remove a run and everything attached to it."""
    conn = get_conn()
    for table in ("traces", "checkpoints"):
        conn.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))
    cur = conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
    conn.commit()
    return cur.rowcount > 0


def list_runs(limit: int = 100) -> List[Dict[str, Any]]:
    rows = get_conn().execute(
        "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_run_row(r) for r in rows]


def _run_row(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["config"] = json.loads(d.get("config") or "{}")
    d["result"] = json.loads(d["result"]) if d.get("result") else None
    return d


# -------------------------------------------------------------- checkpoints

def add_checkpoint(run_id: str, t: int, label: str, kind: str, metrics: Dict[str, Any],
                   diagnostics: Dict[str, Any], snapshot: Dict[str, Any]) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO checkpoints (run_id, t, label, kind, metrics, diagnostics, snapshot, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (run_id, t, label, kind, json.dumps(metrics), json.dumps(diagnostics),
         json.dumps(snapshot), time.time()),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_checkpoints(run_id: str) -> List[Dict[str, Any]]:
    rows = get_conn().execute(
        "SELECT * FROM checkpoints WHERE run_id=? ORDER BY t ASC", (run_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d["metrics"])
        d["diagnostics"] = json.loads(d["diagnostics"])
        d["snapshot"] = json.loads(d["snapshot"])
        out.append(d)
    return out


# ------------------------------------------------------------------- traces

def add_trace(run_id: str, t: int, group_id: str, kind: str, payload: Dict[str, Any]) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO traces (run_id, t, group_id, kind, payload, created_at) VALUES (?,?,?,?,?,?)",
        (run_id, t, group_id, kind, json.dumps(payload), time.time()),
    )
    conn.commit()


def list_traces(run_id: str, limit: int = 500, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM traces WHERE run_id=?"
    args: List[Any] = [run_id]
    if kind:
        sql += " AND kind=?"
        args.append(kind)
    sql += " ORDER BY id ASC LIMIT ?"
    args.append(limit)
    rows = get_conn().execute(sql, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        out.append(d)
    return out


# ------------------------------------------------------------------ streams

def save_stream(stream_id: str, name: str, scenario: str, config: Dict[str, Any]) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO streams (id, name, scenario, config, created_at) VALUES (?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET name=excluded.name, scenario=excluded.scenario,
             config=excluded.config""",
        (stream_id, name, scenario, json.dumps(config), time.time()),
    )
    conn.commit()


def list_streams() -> List[Dict[str, Any]]:
    rows = get_conn().execute("SELECT * FROM streams ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["config"] = json.loads(d["config"])
        out.append(d)
    return out


def get_stream(stream_id: str) -> Optional[Dict[str, Any]]:
    row = get_conn().execute("SELECT * FROM streams WHERE id=?", (stream_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["config"] = json.loads(d["config"])
    return d


# ------------------------------------------------------------- landscape

def add_landscape_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Add a work to the local knowledge base."""
    conn = get_conn()
    now = time.time()
    conn.execute(
        """INSERT INTO landscape_entries
             (id, kind, title, authors, venue, year, url, summary, mechanism,
              align, concepts, patterns, source, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET
             kind=excluded.kind, title=excluded.title, authors=excluded.authors,
             venue=excluded.venue, year=excluded.year, url=excluded.url,
             summary=excluded.summary, mechanism=excluded.mechanism,
             align=excluded.align, concepts=excluded.concepts,
             patterns=excluded.patterns, source=excluded.source""",
        (entry.get("id", ""), entry.get("kind", "note"), entry.get("title", ""),
         entry.get("authors", ""), entry.get("venue", ""), entry.get("year"),
         entry.get("url", ""), entry.get("summary", ""),
         entry.get("mechanism", ""),
         json.dumps(entry.get("align") or {}),
         json.dumps(entry.get("concepts") or []),
         json.dumps(entry.get("patterns") or []),
         entry.get("source", "local"), now),
    )
    conn.commit()
    return list_landscape_entries_by_id(entry.get("id", "")) or entry


def list_landscape_entries() -> List[Dict[str, Any]]:
    rows = get_conn().execute(
        "SELECT * FROM landscape_entries ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for key in ("align", "concepts", "patterns"):
            try:
                d[key] = json.loads(d.get(key) or ("{}" if key == "align" else "[]"))
            except Exception:
                d[key] = {} if key == "align" else []
        out.append(d)
    return out


def list_landscape_entries_by_id(entry_id: str) -> Optional[Dict[str, Any]]:
    row = get_conn().execute("SELECT * FROM landscape_entries WHERE id=?",
                              (entry_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    for key in ("align", "concepts", "patterns"):
        try:
            d[key] = json.loads(d.get(key) or ("{}" if key == "align" else "[]"))
        except Exception:
            d[key] = {} if key == "align" else []
    return d


def delete_landscape_entry(entry_id: str) -> bool:
    conn = get_conn()
    cur = conn.execute("DELETE FROM landscape_entries WHERE id=?", (entry_id,))
    conn.commit()
    return cur.rowcount > 0
