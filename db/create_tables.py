import sqlite3

from db.connection import open_connection, DEFAULT_DB_PATH


def _column_names(cursor, table: str):
    cursor.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def _table_exists(cursor, table: str) -> bool:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cursor.fetchone() is not None


def create_database(db_name: str = DEFAULT_DB_PATH):
    """
    Creates the trading_bot database and the necessary tables (strategies, portfolios, orders, deployments).
    Also migrates older schemas in place.
    """
    conn = open_connection(db_name)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            desc TEXT,
            strategy_class_name TEXT,
            params TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            strategy_id INTEGER,
            starting_value REAL,
            end_value REAL,
            stock TEXT,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER,
            order_type TEXT,
            transaction_type TEXT,
            price REAL,
            quantity INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            portfolio_value REAL,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS deployments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            strategy_id INTEGER NOT NULL,
            token_subscriptions TEXT NOT NULL,
            trading_mode TEXT NOT NULL CHECK(trading_mode IN ('virtual', 'real')),
            is_active BOOLEAN NOT NULL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id),
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        )
    ''')

    strategy_cols = _column_names(cursor, "strategies")
    if "params" not in strategy_cols:
        cursor.execute("ALTER TABLE strategies ADD COLUMN params TEXT")
    if "strategy_class_name" not in strategy_cols:
        cursor.execute("ALTER TABLE strategies ADD COLUMN strategy_class_name TEXT")

    deployment_cols = _column_names(cursor, "deployments")
    if "updated_at" not in deployment_cols:
        cursor.execute(
            "ALTER TABLE deployments ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP"
        )

    if _table_exists(cursor, "deployed_strategies"):
        try:
            cursor.execute(
                """
                INSERT OR IGNORE INTO deployments
                    (id, portfolio_id, strategy_id, token_subscriptions, trading_mode, is_active, created_at, updated_at)
                SELECT id, portfolio_id, strategy_id, token_subscriptions, trading_mode, is_active, created_at, updated_at
                FROM deployed_strategies
                """
            )
        except sqlite3.Error:
            cursor.execute(
                """
                INSERT OR IGNORE INTO deployments
                    (id, portfolio_id, strategy_id, token_subscriptions, trading_mode, is_active, created_at)
                SELECT id, portfolio_id, strategy_id, token_subscriptions, trading_mode, is_active, created_at
                FROM deployed_strategies
                """
            )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_database()
    print("Database and tables created successfully!")
