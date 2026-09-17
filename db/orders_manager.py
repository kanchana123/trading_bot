# in db/orders_manager.py

import sqlite3
from typing import List, Tuple, Optional, Dict
from datetime import datetime

from db.connection import open_connection


class OrdersManager:
    """
    Manages orders in the SQLite database.
    """

    def __init__(self, db_name: str = 'trading_bot.db'):
        """
        Initializes the OrdersManager.

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

    def create_order(
        self,
        portfolio_id: int,
        order_type: str,
        transaction_type: str,
        price: float,
        quantity: int,
        metadata: str = None,
        portfolio_value: float = None,
        timestamp: datetime = None
    ) -> int:
        """
        Creates a new order in the database.

        Args:
            portfolio_id (int): The ID of the portfolio the order belongs to.
            order_type (str): The type of the order ('backtest', 'virtual', 'real').
            transaction_type (str): The type of transaction ('buy', 'sell').
            price (float): The price of the order.
            quantity (int): quantity of the order.
            metadata (str, optional): Additional metadata for the order. Defaults to None.
            portfolio_value (float, optional): total value of the portfolio. Defaults to None.
            timestamp(datetime, optional): timestamp of the transaction. Defaults to None

        Returns:
            int: The ID of the newly created order.
        Raises:
            ValueError: if order type or transaction type is not valid.
            sqlite3.IntegrityError: If a foreign key constraint is violated.
        """
        conn = self._connect()
        cursor = conn.cursor()
        
        if order_type not in ('backtest', 'virtual', 'real'):
            raise ValueError(f"Invalid order_type: {order_type}. Must be one of 'backtest', 'virtual', 'real'.")
        if transaction_type not in ('buy', 'sell'):
            raise ValueError(f"Invalid transaction_type: {transaction_type}. Must be one of 'buy', 'sell'.")

        try:
            stored_timestamp = timestamp
            if stored_timestamp is not None and hasattr(stored_timestamp, "isoformat"):
                stored_timestamp = stored_timestamp.isoformat(sep=" ", timespec="seconds")
            if stored_timestamp is None:
                cursor.execute(
                    "INSERT INTO orders (portfolio_id, order_type, transaction_type, price, quantity, metadata, portfolio_value) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (portfolio_id, order_type, transaction_type, price, quantity, metadata, portfolio_value),
                )
            else:
                 cursor.execute(
                    "INSERT INTO orders (portfolio_id, order_type, transaction_type, price, quantity, metadata, portfolio_value, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (portfolio_id, order_type, transaction_type, price, quantity, metadata, portfolio_value, stored_timestamp),
                )

            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise sqlite3.IntegrityError(f"Error: Foreign key constraint violation. Check if portfolio_id exists.") from e
        finally:
            conn.close()

    def get_order(self, order_id: int) -> Optional[Tuple]:
        """
        Retrieves an order from the database by its ID.

        Args:
            order_id (int): The ID of the order to retrieve.

        Returns:
            Optional[Tuple]: A tuple containing the order details if found, or None if not found.
            (id, portfolio_id, order_type, transaction_type, price, quantity, timestamp, metadata, portfolio_value)
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, portfolio_id, order_type, transaction_type, price, quantity, timestamp, metadata, portfolio_value FROM orders WHERE id = ?",
            (order_id,),
        )
        order = cursor.fetchone()

        conn.close()
        return order

    def get_all_orders(self) -> List[Tuple]:
        """
        Retrieves all orders from the database.

        Returns:
            List[Tuple]: A list of tuples, where each tuple represents an order.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT id, portfolio_id, order_type, transaction_type, price, quantity, timestamp, metadata, portfolio_value FROM orders")
        orders = cursor.fetchall()

        conn.close()
        return orders

    def update_order(
        self,
        order_id: int,
        portfolio_id: int = None,
        order_type: str = None,
        transaction_type: str = None,
        price: float = None,
        quantity: int = None,
        metadata: str = None,
        portfolio_value: float = None,
        timestamp: datetime = None
    ) -> bool:
        """
        Updates an existing order in the database.

        Args:
            order_id (int): The ID of the order to update.
            portfolio_id (int, optional): The new portfolio ID. Defaults to None.
            order_type (str, optional): The new order type. Defaults to None.
            transaction_type (str, optional): The new transaction type. Defaults to None.
            price (float, optional): The new price. Defaults to None.
            quantity (int, optional): The new quantity. Defaults to None.
            metadata (str, optional): The new metadata. Defaults to None.
            portfolio_value (float, optional): total value of the portfolio. Defaults to None.
            timestamp(datetime, optional): timestamp of the transaction. Defaults to None

        Returns:
            bool: True if the order was updated, False if the order was not found.
        Raises:
            ValueError: if order type or transaction type is not valid.
            sqlite3.IntegrityError: If a foreign key constraint is violated.
        """
        conn = self._connect()
        cursor = conn.cursor()
        
        if order_type is not None and order_type not in ('backtest', 'virtual', 'real'):
            raise ValueError(f"Invalid order_type: {order_type}. Must be one of 'backtest', 'virtual', 'real'.")
        if transaction_type is not None and transaction_type not in ('buy', 'sell'):
            raise ValueError(f"Invalid transaction_type: {transaction_type}. Must be one of 'buy', 'sell'.")

        try:
            updates = []
            params = []

            if portfolio_id is not None:
                updates.append("portfolio_id = ?")
                params.append(portfolio_id)
            if order_type is not None:
                updates.append("order_type = ?")
                params.append(order_type)
            if transaction_type is not None:
                updates.append("transaction_type = ?")
                params.append(transaction_type)
            if price is not None:
                updates.append("price = ?")
                params.append(price)
            if quantity is not None:
                updates.append("quantity = ?")
                params.append(quantity)
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(metadata)
            if portfolio_value is not None:
                updates.append("portfolio_value = ?")
                params.append(portfolio_value)
            if timestamp is not None:
                updates.append("timestamp = ?")
                params.append(timestamp)

            if not updates:
                conn.close()
                return False  # Nothing to update

            query = f"UPDATE orders SET {', '.join(updates)} WHERE id = ?"
            params.append(order_id)

            cursor.execute(query, params)
            conn.commit()

            return cursor.rowcount > 0
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise sqlite3.IntegrityError(f"Error: Foreign key constraint violation. Check if portfolio_id exists.") from e
        finally:
            conn.close()

    def delete_order(self, order_id: int) -> bool:
        """
        Deletes an order from the database.

        Args:
            order_id (int): The ID of the order to delete.

        Returns:
            bool: True if the order was deleted, False if the order was not found.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM orders WHERE id = ?", (order_id,))
        conn.commit()
        conn.close()

        return cursor.rowcount > 0


# Example Usage (for testing):
if __name__ == "__main__":
    # Initialize the OrdersManager
    orders_manager = OrdersManager()

    # Create an order
    try:
        order_id = orders_manager.create_order(
            portfolio_id=1,
            order_type="backtest",
            transaction_type="buy",
            price=150.50,
            metadata="{'strategy': 'MA crossover'}",
            portfolio_value=1000.0,
            timestamp= datetime.now()
        )
        print(f"Created order with ID: {order_id}")

        order_id2 = orders_manager.create_order(
            portfolio_id=1,
            order_type="virtual",
            transaction_type="sell",
            price=160.50,
            metadata="{'strategy': 'MA crossover'}",
            portfolio_value=1100.0
        )
        print(f"Created order with ID: {order_id2}")
        
        # try to create an order with wrong order type
        # order_id2 = orders_manager.create_order(
        #     portfolio_id=1,
        #     order_type="wrong_type",
        #     transaction_type="sell",
        #     price=160.50,
        #     metadata="{'strategy': 'MA crossover'}",
        #     portfolio_value=1100.0
        # )
        # print(f"Created order with ID: {order_id2}")

    except (sqlite3.IntegrityError, ValueError) as e:
        print(e)
    
    # Get all orders
    all_orders = orders_manager.get_all_orders()
    print("All orders:", all_orders)

    # Get an order by ID
    order = orders_manager.get_order(order_id)
    if order:
        print(f"Retrieved order: {order}")

    # Update an order
    updated = orders_manager.update_order(order_id, price=155.75, metadata="{'strategy': 'MA crossover', 'comment': 'updated'}")
    if updated:
        print("Order updated successfully.")
    else:
        print("Order not found for update.")

    # Get all orders
    all_orders = orders_manager.get_all_orders()
    print("All orders:", all_orders)
    
    # update order with wrong type
    try:
        updated = orders_manager.update_order(order_id, transaction_type="wrong_type")
    except (sqlite3.IntegrityError, ValueError) as e:
        print(e)
        
    # Delete an order
    deleted = orders_manager.delete_order(order_id)
    if deleted:
        print("Order deleted successfully.")
    else:
        print("Order not found for deletion.")

    # Get all orders
    all_orders = orders_manager.get_all_orders()
    print("All orders:", all_orders)

