from datetime import datetime

class Order:
    """Represents a trading order."""

    def __init__(self, order_type: str, transaction_type: str, price: float, quantity: int, metadata: str = None, portfolio_value: float = None, timestamp:datetime=None):
        self.order_type = order_type  # 'backtest', 'virtual', 'real'
        self.transaction_type = transaction_type  # 'buy', 'sell'
        self.price = price
        self.quantity = quantity
        self.metadata = metadata
        self.portfolio_value = portfolio_value
        self.timestamp = timestamp

    def __str__(self):
        return f"Order(type={self.order_type}, transaction={self.transaction_type}, price={self.price}, quantity={self.quantity}, time={self.timestamp}, portfolio_value={self.portfolio_value}), metadata={self.metadata})"
