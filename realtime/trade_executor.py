import json
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Places virtual or real orders and persists them."""

    def __init__(self, orders_manager, angel_api_client=None):
        self.orders_manager = orders_manager
        self.angel_api_client = angel_api_client

    def execute_order(
        self,
        portfolio_id: int,
        strategy_id: Optional[int],
        token_details: Dict,
        transaction_type: str,
        quantity: int,
        price: float,
        order_type: str = "LIMIT",
        trading_mode: str = "virtual",
        product_type: str = "INTRADAY",
        variety: str = "NORMAL",
    ) -> Dict:
        action = (transaction_type or "").lower()
        if action not in ("buy", "sell"):
            return {"success": False, "error": f"Invalid transaction_type: {transaction_type}"}
        if quantity is None or int(quantity) <= 0:
            return {"success": False, "error": "Quantity must be positive."}
        if trading_mode not in ("virtual", "real", "backtest"):
            return {"success": False, "error": f"Invalid trading_mode: {trading_mode}"}

        broker_order_id = None
        if trading_mode == "real":
            if not self.angel_api_client or not self.angel_api_client.is_session_active():
                return {"success": False, "error": "Angel API client is not available for real orders."}
            result = self.angel_api_client.place_order(
                tradingsymbol=token_details.get("symbol"),
                symboltoken=token_details.get("instrumentToken"),
                transaction_type=action,
                quantity=int(quantity),
                price=float(price),
                exchange=token_details.get("exchange", "NSE"),
                order_type=order_type,
                product_type=product_type,
                variety=variety,
            )
            if not result.get("success"):
                logger.error("Real order rejected: %s", result)
                return result
            broker_order_id = result.get("broker_order_id")

        metadata = {
            "strategy_id": strategy_id,
            "symbol": token_details.get("symbol"),
            "instrumentToken": token_details.get("instrumentToken"),
            "exchange": token_details.get("exchange"),
            "order_type": order_type,
            "product_type": product_type,
            "broker_order_id": broker_order_id,
        }
        try:
            order_id = self.orders_manager.create_order(
                portfolio_id=portfolio_id,
                order_type=trading_mode if trading_mode != "backtest" else "backtest",
                transaction_type=action,
                price=float(price),
                quantity=int(quantity),
                metadata=json.dumps(metadata),
                portfolio_value=None,
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error("Failed to persist order: %s", e, exc_info=True)
            return {"success": False, "error": f"Order may have been sent but was not saved: {e}"}

        logger.info(
            "Executed %s %s x%s @ %s (mode=%s, db_order_id=%s, broker_order_id=%s)",
            action,
            token_details.get("symbol"),
            quantity,
            price,
            trading_mode,
            order_id,
            broker_order_id,
        )
        return {
            "success": True,
            "order_id": order_id,
            "broker_order_id": broker_order_id,
        }
