import sqlite3
import threading
from contextlib import contextmanager

DEFAULT_DB_PATH = "trading_bot.db"
_db_lock = threading.RLock()


def open_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled and a write timeout."""
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_session(db_path: str = DEFAULT_DB_PATH):
    """Serialize DB access across FastAPI and websocket threads."""
    with _db_lock:
        conn = open_connection(db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
