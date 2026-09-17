import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from db.connection import open_connection
from db.create_tables import create_database

logger = logging.getLogger(__name__)


class DeploymentManager:
    def __init__(self, db_path="trading_bot.db"):
        self.db_path = db_path
        create_database(self.db_path)

    def _get_db_connection(self):
        conn = open_connection(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_deployment(
        self,
        portfolio_id: int,
        strategy_id: int,
        token_subscriptions: List[Dict],
        trading_mode: str,
        is_active: bool = False,
    ) -> Optional[int]:
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            token_subscriptions_json = json.dumps(token_subscriptions)
            now = datetime.now()
            cursor.execute(
                """
                INSERT INTO deployments
                    (portfolio_id, strategy_id, token_subscriptions, trading_mode, is_active, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    portfolio_id,
                    strategy_id,
                    token_subscriptions_json,
                    trading_mode,
                    is_active,
                    now,
                ),
            )
            conn.commit()
            deployment_id = cursor.lastrowid
            logger.info(
                "Created deployment ID %s for portfolio %s, strategy %s, mode %s.",
                deployment_id,
                portfolio_id,
                strategy_id,
                trading_mode,
            )
            return deployment_id
        except sqlite3.Error as e:
            logger.error("Database error creating deployment: %s", e, exc_info=True)
            return None
        finally:
            conn.close()

    def get_deployment_by_id(self, deployment_id: int) -> Optional[Dict]:
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM deployments WHERE id = ?", (deployment_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(
                "Database error fetching deployment %s: %s",
                deployment_id,
                e,
                exc_info=True,
            )
            return None
        finally:
            conn.close()

    def get_all_deployments(self) -> List[Dict]:
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM deployments ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error("Database error fetching all deployments: %s", e, exc_info=True)
            return []
        finally:
            conn.close()

    def get_active_deployments(self) -> List[Dict]:
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM deployments WHERE is_active = 1 ORDER BY created_at DESC"
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(
                "Database error fetching active deployments: %s", e, exc_info=True
            )
            return []
        finally:
            conn.close()

    def update_deployment_status(self, deployment_id: int, is_active: bool) -> bool:
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE deployments
                SET is_active = ?, updated_at = ?
                WHERE id = ?
                """,
                (is_active, datetime.now(), deployment_id),
            )
            conn.commit()
            updated_rows = cursor.rowcount
            if updated_rows > 0:
                logger.info(
                    "Deployment %s status updated to is_active=%s.",
                    deployment_id,
                    is_active,
                )
            else:
                logger.warning(
                    "No deployment found with ID %s to update status.", deployment_id
                )
            return updated_rows > 0
        except sqlite3.Error as e:
            logger.error(
                "Database error updating deployment %s status: %s",
                deployment_id,
                e,
                exc_info=True,
            )
            return False
        finally:
            conn.close()

    def update_deployment_tokens(
        self, deployment_id: int, token_subscriptions: List[Dict]
    ) -> bool:
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            token_subscriptions_json = json.dumps(token_subscriptions)
            cursor.execute(
                """
                UPDATE deployments
                SET token_subscriptions = ?, updated_at = ?
                WHERE id = ?
                """,
                (token_subscriptions_json, datetime.now(), deployment_id),
            )
            conn.commit()
            updated_rows = cursor.rowcount
            if updated_rows > 0:
                logger.info("Deployment %s token subscriptions updated.", deployment_id)
            else:
                logger.warning(
                    "No deployment found with ID %s to update tokens.", deployment_id
                )
            return updated_rows > 0
        except sqlite3.Error as e:
            logger.error(
                "Database error updating deployment %s tokens: %s",
                deployment_id,
                e,
                exc_info=True,
            )
            return False
        finally:
            conn.close()

    def delete_deployment(self, deployment_id: int) -> bool:
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM deployments WHERE id = ?", (deployment_id,))
            conn.commit()
            deleted_rows = cursor.rowcount
            if deleted_rows > 0:
                logger.info("Deployment %s deleted.", deployment_id)
            else:
                logger.warning(
                    "No deployment found with ID %s to delete.", deployment_id
                )
            return deleted_rows > 0
        except sqlite3.Error as e:
            logger.error(
                "Database error deleting deployment %s: %s",
                deployment_id,
                e,
                exc_info=True,
            )
            return False
        finally:
            conn.close()
