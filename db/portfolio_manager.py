import sqlite3
from typing import List, Tuple, Optional

from db.connection import open_connection


class PortfolioManager:
    """
    Manages portfolios in the SQLite database.
    """

    def __init__(self, db_name: str = 'trading_bot.db'):
        """
        Initializes the PortfolioManager.

        Args:
            db_name (str, optional): The name of the database file. Defaults to 'trading_bot.db'.
        """
        self.db_name = db_name

    def _connect(self):
        """
        Establishes a connection to the SQLite database.

        Returns:
            sqlite3.Connection: The database connection object.
        """
        return open_connection(self.db_name)

    def create_portfolio(
        self, name: str, strategy_id: int, starting_value: float, end_value: float = None, stock: str = None
    ) -> int:
        """
        Creates a new portfolio in the database.

        Args:
            name (str): The name of the portfolio (must be unique).
            strategy_id (int): The ID of the strategy associated with the portfolio.
            starting_value (float): The starting value of the portfolio.
            end_value (float, optional): The end value of the portfolio. Defaults to None.
            stock (str, optional): The stock name for this portfolio. Defaults to None

        Returns:
            int: The ID of the newly created portfolio.
        Raises:
            sqlite3.IntegrityError: If a portfolio with the same name already exists or if a foreign key constraint is violated.
        """
        conn = self._connect()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO portfolios (name, strategy_id, starting_value, end_value, stock) VALUES (?, ?, ?, ?, ?)",
                (name, strategy_id, starting_value, end_value, stock),
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise sqlite3.IntegrityError(
                f"Error: A portfolio with the name '{name}' already exists or foreign key constraint violation."
            ) from e
        finally:
            conn.close()

    def get_portfolio(self, portfolio_id: int) -> Optional[Tuple[int, str, int, float, float, str]]:
        """
        Retrieves a portfolio from the database by its ID.

        Args:
            portfolio_id (int): The ID of the portfolio to retrieve.

        Returns:
            Optional[Tuple[int, str, int, float, float, str]]: A tuple containing (id, name, strategy_id, starting_value, end_value, stock) if found, or None if not found.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, name, strategy_id, starting_value, end_value, stock FROM portfolios WHERE id = ?",
            (portfolio_id,),
        )
        portfolio = cursor.fetchone()

        conn.close()
        return portfolio

    def get_all_portfolios(self) -> List[Tuple[int, str, int, float, float, str]]:
        """
        Retrieves all portfolios from the database.

        Returns:
            List[Tuple[int, str, int, float, float, str]]: A list of tuples, where each tuple represents a portfolio (id, name, strategy_id, starting_value, end_value, stock).
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT id, name, strategy_id, starting_value, end_value, stock FROM portfolios")
        portfolios = cursor.fetchall()

        conn.close()
        return portfolios

    def update_portfolio(
        self, portfolio_id: int, name: str = None, strategy_id: int = None, starting_value: float = None, end_value: float = None, stock:str=None
    ) -> bool:
        """
        Updates an existing portfolio in the database.

        Args:
            portfolio_id (int): The ID of the portfolio to update.
            name (str, optional): The new name of the portfolio. Defaults to None.
            strategy_id (int, optional): The new strategy ID. Defaults to None.
            starting_value (float, optional): The new starting value. Defaults to None.
            end_value (float, optional): The new end value. Defaults to None.
            stock (str, optional): The new stock value. Defaults to None.

        Returns:
            bool: True if the portfolio was updated, False if the portfolio was not found.
        Raises:
            sqlite3.IntegrityError: If the updated portfolio name already exists or if a foreign key constraint is violated.
        """
        conn = self._connect()
        cursor = conn.cursor()

        try:
            updates = []
            params = []

            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if strategy_id is not None:
                updates.append("strategy_id = ?")
                params.append(strategy_id)
            if starting_value is not None:
                updates.append("starting_value = ?")
                params.append(starting_value)
            if end_value is not None:
                updates.append("end_value = ?")
                params.append(end_value)
            if stock is not None:
                updates.append("stock = ?")
                params.append(stock)

            if not updates:
                conn.close()
                return False  # Nothing to update

            query = f"UPDATE portfolios SET {', '.join(updates)} WHERE id = ?"
            params.append(portfolio_id)

            cursor.execute(query, params)
            conn.commit()

            return cursor.rowcount > 0
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise sqlite3.IntegrityError(
                f"Error: A portfolio with the name '{name}' already exists or foreign key constraint violation."
            ) from e
        finally:
            conn.close()

    def delete_portfolio(self, portfolio_id: int) -> bool:
        """
        Deletes a portfolio from the database.

        Args:
            portfolio_id (int): The ID of the portfolio to delete.

        Returns:
            bool: True if the portfolio was deleted, False if the portfolio was not found.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))
        conn.commit()
        conn.close()

        return cursor.rowcount > 0


# Example Usage (for testing):
if __name__ == "__main__":
    # Initialize the PortfolioManager
    portfolio_manager = PortfolioManager()

    # Create a portfolio
    try:
        portfolio_id = portfolio_manager.create_portfolio(
            name="My First Portfolio", strategy_id=1, starting_value=10000.0, stock='banknifty'
        )
        print(f"Created portfolio with ID: {portfolio_id}")

        portfolio_id2 = portfolio_manager.create_portfolio(
            name="second portfolio", strategy_id=2, starting_value=15000.0, stock='nifty'
        )
        print(f"Created portfolio with ID: {portfolio_id2}")

        # try creating duplicate portfolio
        #portfolio_id2 = portfolio_manager.create_portfolio(
        #    name="My First Portfolio", strategy_id=2, starting_value=15000.0, stock='nifty'
        #)
        #print(f"Created portfolio with ID: {portfolio_id2}")

    except sqlite3.IntegrityError as e:
        print(e)

    # Get all portfolios
    all_portfolios = portfolio_manager.get_all_portfolios()
    print("All portfolios:", all_portfolios)

    # Get a portfolio by ID
    portfolio = portfolio_manager.get_portfolio(portfolio_id)
    if portfolio:
        print(f"Retrieved portfolio: {portfolio}")

    # Update a portfolio
    updated = portfolio_manager.update_portfolio(portfolio_id, end_value=12000.0)
    if updated:
        print("Portfolio updated successfully.")
    else:
        print("Portfolio not found for update.")

    # Get all portfolios
    all_portfolios = portfolio_manager.get_all_portfolios()
    print("All portfolios:", all_portfolios)

    #update a portfolio with duplicate name
    try:
        updated = portfolio_manager.update_portfolio(portfolio_id, name="second portfolio")
    except sqlite3.IntegrityError as e:
        print(e)

    # Delete a portfolio
    deleted = portfolio_manager.delete_portfolio(portfolio_id)
    if deleted:
        print("Portfolio deleted successfully.")
    else:
        print("Portfolio not found for deletion.")

    # Get all portfolios
    all_portfolios = portfolio_manager.get_all_portfolios()
    print("All portfolios:", all_portfolios)
