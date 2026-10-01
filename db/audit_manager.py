import json
import sqlite3
from typing import Any, Dict, List, Optional

from db.connection import open_connection


class AuditLedger:
    """Persist graph snapshots for compliance replay (SQLite; Postgres-compatible DDL)."""

    def __init__(self, db_name: str = "trading_bot.db"):
        self.db_name = db_name

    def _connect(self):
        return open_connection(self.db_name)

    def record_run(self, state_dump: Dict[str, Any], status: str):
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO agent_runs (run_id, thread_id, symbol, trading_mode, status, prompt_version, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    state_json=excluded.state_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    state_dump.get("run_id"),
                    state_dump.get("thread_id"),
                    state_dump.get("symbol"),
                    state_dump.get("trading_mode"),
                    status,
                    state_dump.get("prompt_version"),
                    json.dumps(state_dump),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def record_event(self, run_id: str, event: Dict[str, Any]):
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO agent_events (run_id, node, kind, message, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    event.get("node"),
                    event.get("kind"),
                    event.get("message"),
                    json.dumps(event.get("payload") or {}),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT run_id, thread_id, symbol, trading_mode, status, prompt_version, state_json, created_at, updated_at FROM agent_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if not row:
                return None
            payload = dict(row)
            payload["state"] = json.loads(payload.pop("state_json") or "{}")
            return payload
        finally:
            conn.close()

    def list_pending(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT run_id, thread_id, symbol, trading_mode, status, prompt_version, state_json, created_at FROM agent_runs WHERE status = 'awaiting_human' ORDER BY created_at DESC"
            ).fetchall()
            out = []
            for row in rows:
                item = dict(row)
                item["state"] = json.loads(item.pop("state_json") or "{}")
                out.append(item)
            return out
        finally:
            conn.close()

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT run_id, symbol, trading_mode, status, prompt_version, created_at, updated_at FROM agent_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
