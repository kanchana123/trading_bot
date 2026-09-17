from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

class RealTimeStrategy(ABC):
    """
    Base class for real-time trading strategies that operate on tick data.
    """
    def __init__(self, name: str, params: Optional[Dict] = None):
        self.name = name
        self.params = params if params is not None else {}
        self.desc: str = "Base Real-Time Strategy" # Subclasses should override

        # These will be set by RealtimeTrader when an instance is created for a deployment
        self.deployment_id: Optional[int] = None
        self.portfolio_id: Optional[int] = None
        self.strategy_id_db: Optional[int] = None # ID of the strategy definition in DB
        self.trading_mode: Optional[str] = None # 'virtual' or 'real'
        self.token_info: Optional[Dict] = None # {'instrumentToken': '...', 'symbol': '...', 'exchange': '...'}

    @abstractmethod
    def on_new_tick(self, tick_data: Dict, token_details: Dict) -> Optional[Dict[str, Any]]:
        """
        Process a new tick for the subscribed token.
        
        Args:
            tick_data (Dict): The raw tick data from the WebSocket.
                              Example: {'tk': '2885', 'ltp': '2900.50', ...}
                                    or {'token': '2885', 'last_traded_price': '2900.50', ...}
            token_details (Dict): Detailed information about the token this tick belongs to.
                                  Example: {'instrumentToken': '2885', 'symbol': 'RELIANCE-EQ', 'exchange': 'NSE_EQ'}

        Returns:
            Optional[Dict[str, Any]]: A dictionary with trade signal parameters if a trade is to be made,
                                      None otherwise.
                                      Example: {'action': 'buy', 'quantity': 10, 'price': 2900.50, 
                                                'order_type': 'LIMIT', 'product_type': 'INTRADAY'}
        """
        pass

    def generate_desc(self) -> str:
        """
        Generates a description for the strategy based on its parameters.
        Subclasses should implement this.
        """
        return f"{self.name} with params: {self.params}"

    # Helper methods common to RT strategies could go here
    # e.g., managing position state for this specific token if not handled by portfolio manager globally
    # def get_current_position(self) -> int:
    #     # This would require interaction with a position tracking mechanism
    #     pass
