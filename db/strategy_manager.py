import json
import sqlite3
from typing import Dict, List, Optional

from db.connection import open_connection


class StrategyManager:
    """
    Manages trading strategies in the SQLite database.
    """

    def __init__(self, db_name: str = "trading_bot.db"):
        self.db_name = db_name

    def _connect(self):
        return open_connection(self.db_name)

    def _row_to_dict(self, row) -> Optional[Dict]:
        if row is None:
            return None
        params = None
        if row[4]:
            try:
                params = json.loads(row[4])
            except (TypeError, json.JSONDecodeError):
                params = None
        return {
            "id": row[0],
            "name": row[1],
            "desc": row[2],
            "strategy_class_name": row[3],
            "params": params,
        }

    def create_strategy(
        self,
        name: str,
        desc: str = None,
        strategy_class_name: str = None,
        params: Optional[Dict] = None,
    ) -> int:
        conn = self._connect()
        cursor = conn.cursor()
        params_json = json.dumps(params) if params is not None else None
        try:
            cursor.execute(
                """
                INSERT INTO strategies (name, desc, strategy_class_name, params)
                VALUES (?, ?, ?, ?)
                """,
                (name, desc, strategy_class_name, params_json),
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise sqlite3.IntegrityError(
                f"Error: A strategy with the name '{name}' already exists."
            ) from e
        finally:
            conn.close()

    def get_strategy(self, strategy_id: int) -> Optional[Dict]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, desc, strategy_class_name, params FROM strategies WHERE id = ?",
            (strategy_id,),
        )
        strategy = self._row_to_dict(cursor.fetchone())
        conn.close()
        return strategy

    def get_all_strategies(self) -> List[Dict]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, desc, strategy_class_name, params FROM strategies"
        )
        strategies = [self._row_to_dict(row) for row in cursor.fetchall()]
        conn.close()
        return strategies

    def update_strategy(
        self,
        strategy_id: int,
        name: str = None,
        desc: str = None,
        strategy_class_name: str = None,
        params: Optional[Dict] = None,
    ) -> bool:
        conn = self._connect()
        cursor = conn.cursor()
        try:
            updates = []
            values = []
            if name is not None:
                updates.append("name = ?")
                values.append(name)
            if desc is not None:
                updates.append("desc = ?")
                values.append(desc)
            if strategy_class_name is not None:
                updates.append("strategy_class_name = ?")
                values.append(strategy_class_name)
            if params is not None:
                updates.append("params = ?")
                values.append(json.dumps(params))
            if not updates:
                conn.close()
                return False
            values.append(strategy_id)
            cursor.execute(
                f"UPDATE strategies SET {', '.join(updates)} WHERE id = ?", values
            )
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise sqlite3.IntegrityError(
                f"Error: A strategy with the name '{name}' already exists."
            ) from e
        finally:
            conn.close()

    def delete_strategy(self, strategy_id: int) -> bool:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM strategies WHERE id = ?", (strategy_id,))
        conn.commit()
        conn.close()
        return cursor.rowcount > 0
