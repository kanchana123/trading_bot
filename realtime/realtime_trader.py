import logging
import json
from threading import Thread, RLock
from typing import Dict, List, Optional

from db.deployment_manager import DeploymentManager
from db.orders_manager import OrdersManager
from db.strategy_manager import StrategyManager
from realtime.angel_websocket_client import AngelWebSocketClient
from realtime.trade_executor import TradeExecutor
from base_models.angel_api import AngelAPI
from strategies.action_price.KernelTrader import KernelStrategy

logger = logging.getLogger(__name__)

STRATEGY_CLASS_MAP = {
    "KernelMomentum": KernelStrategy,
}


class RealtimeTrader:
    def __init__(
        self,
        deployment_manager: DeploymentManager,
        strategy_manager: StrategyManager,
        orders_manager: OrdersManager,
        trade_executor: TradeExecutor,
        angel_api_http_client: AngelAPI,
    ):
        self.deployment_manager = deployment_manager
        self.strategy_manager = strategy_manager
        self.orders_manager = orders_manager
        self.trade_executor = trade_executor
        self.angel_api_http_client = angel_api_http_client

        self.ws_client: Optional[AngelWebSocketClient] = None
        self.active_deployments: List[Dict] = []
        self.strategy_instances: Dict[str, List[object]] = {}
        self.token_details_map: Dict[str, Dict] = {}

        self._is_running = False
        self._lock = RLock()
        self._thread: Optional[Thread] = None

    def _get_angel_creds_and_feed_token(self):
        client = self.angel_api_http_client
        if not client:
            return None
        if not client.is_session_active():
            logger.info("Angel API session not active. Attempting to log in.")
            try:
                client.login()
            except Exception as e:
                logger.error("Angel API login failed: %s", e, exc_info=True)
                return None

        auth_token = client.jwt_token
        api_key = client.api_key
        client_code = client.client_code
        feed_token = client.feed_token

        if not all([auth_token, api_key, client_code, feed_token]):
            missing = [
                name
                for name, val in zip(
                    ["auth_token", "api_key", "client_code", "feed_token"],
                    [auth_token, api_key, client_code, feed_token],
                )
                if not val
            ]
            logger.error("Could not retrieve websocket credentials. Missing: %s", missing)
            return None
        return {
            "auth_token": auth_token,
            "api_key": api_key,
            "client_code": client_code,
            "feed_token": feed_token,
        }

    def start_trading(self):
        with self._lock:
            if self._is_running:
                logger.info("RealtimeTrader is already running.")
                return False
            self._is_running = True
            self._thread = Thread(target=self._run_loop, name="RealtimeTrader", daemon=True)
            self._thread.start()
            logger.info("RealtimeTrader thread started.")
            return True

    def _run_loop(self):
        creds = self._get_angel_creds_and_feed_token()
        if not creds:
            logger.error("Failed to start RealtimeTrader: Could not get Angel credentials/feed token.")
            self._is_running = False
            return

        self.ws_client = AngelWebSocketClient(
            auth_token=creds["auth_token"],
            api_key=creds["api_key"],
            client_code=creds["client_code"],
            feed_token=creds["feed_token"],
            tick_callback=self._handle_tick_data,
            on_open_callback=self._load_and_subscribe_deployments,
        )
        try:
            self.ws_client.connect()
        except Exception as e:
            logger.error("RealtimeTrader websocket loop ended: %s", e, exc_info=True)
        finally:
            self._is_running = False
            self.ws_client = None

    def _load_and_subscribe_deployments(self):
        with self._lock:
            self.active_deployments = self.deployment_manager.get_active_deployments()
            if not self.active_deployments:
                logger.info("No active deployments found.")
                return

            all_tokens_to_subscribe_info = []
            self.strategy_instances.clear()
            self.token_details_map.clear()

            for dep in self.active_deployments:
                deployment_id = dep["id"]
                portfolio_id = dep["portfolio_id"]
                strategy_id_db = dep["strategy_id"]
                trading_mode = dep["trading_mode"]

                try:
                    token_subscriptions_list = json.loads(dep["token_subscriptions"])
                except (TypeError, json.JSONDecodeError):
                    logger.error(
                        "Failed to parse token_subscriptions for deployment %s. Skipping.",
                        deployment_id,
                    )
                    continue

                strategy_details_db = self.strategy_manager.get_strategy(strategy_id_db)
                if not strategy_details_db:
                    logger.error(
                        "Strategy with ID %s not found for deployment %s. Skipping.",
                        strategy_id_db,
                        deployment_id,
                    )
                    continue

                strategy_name_db = (
                    strategy_details_db.get("strategy_class_name")
                    or strategy_details_db.get("name")
                )
                strat_params = strategy_details_db.get("params")
                StrategyClass = STRATEGY_CLASS_MAP.get(strategy_name_db)
                if not StrategyClass:
                    logger.error(
                        "Strategy class for '%s' not found in STRATEGY_CLASS_MAP. Skipping deployment %s.",
                        strategy_name_db,
                        deployment_id,
                    )
                    continue

                for token_info in token_subscriptions_list:
                    instrument_token_str = str(token_info["instrumentToken"])
                    self.token_details_map[instrument_token_str] = token_info
                    try:
                        strategy_instance = StrategyClass(params=strat_params)
                        strategy_instance.deployment_id = deployment_id
                        strategy_instance.portfolio_id = portfolio_id
                        strategy_instance.strategy_id_db = strategy_id_db
                        strategy_instance.trading_mode = trading_mode
                        strategy_instance.token_info = token_info
                        self.strategy_instances.setdefault(instrument_token_str, []).append(
                            strategy_instance
                        )
                        logger.info(
                            "Instantiated strategy '%s' for token %s (Deployment: %s)",
                            strategy_name_db,
                            token_info.get("symbol"),
                            deployment_id,
                        )
                    except Exception as e:
                        logger.error(
                            "Error instantiating strategy %s for token %s: %s",
                            strategy_name_db,
                            token_info.get("symbol"),
                            e,
                            exc_info=True,
                        )
                        continue

                    if not any(
                        t["instrumentToken"] == token_info["instrumentToken"]
                        for t in all_tokens_to_subscribe_info
                    ):
                        all_tokens_to_subscribe_info.append(token_info)

            if all_tokens_to_subscribe_info and self.ws_client:
                logger.info(
                    "Subscribing to %s unique tokens via WebSocket.",
                    len(all_tokens_to_subscribe_info),
                )
                self.ws_client.subscribe_tokens(all_tokens_to_subscribe_info)

    def _handle_tick_data(self, tick_data: Dict):
        instrument_token = (
            tick_data.get("tk") or tick_data.get("token") or tick_data.get("symbolToken")
        )
        if not instrument_token:
            return
        instrument_token_str = str(instrument_token)

        with self._lock:
            if instrument_token_str not in self.strategy_instances:
                return
            if instrument_token_str not in self.token_details_map:
                logger.warning(
                    "Received tick for token %s but no details found in map.",
                    instrument_token_str,
                )
                return

            token_full_details = self.token_details_map[instrument_token_str]
            for strategy_instance in self.strategy_instances[instrument_token_str]:
                try:
                    trade_signal = strategy_instance.on_new_tick(
                        tick_data, token_full_details
                    )
                    if not trade_signal:
                        continue
                    logger.info(
                        "Strategy '%s' (Dep: %s) signaled %s for %s",
                        type(strategy_instance).__name__,
                        getattr(strategy_instance, "deployment_id", None),
                        trade_signal.get("action"),
                        token_full_details.get("symbol"),
                    )
                    result = self.trade_executor.execute_order(
                        portfolio_id=strategy_instance.portfolio_id,
                        strategy_id=strategy_instance.strategy_id_db,
                        token_details=token_full_details,
                        transaction_type=trade_signal["action"],
                        quantity=trade_signal["quantity"],
                        price=trade_signal["price"],
                        order_type=trade_signal.get("order_type", "LIMIT"),
                        trading_mode=strategy_instance.trading_mode,
                        product_type=trade_signal.get("product_type", "INTRADAY"),
                        variety=trade_signal.get("variety", "NORMAL"),
                    )
                    if result.get("success") and hasattr(strategy_instance, "confirm_fill"):
                        strategy_instance.confirm_fill(trade_signal)
                    elif not result.get("success"):
                        logger.error("Order execution failed: %s", result)
                except Exception as e:
                    logger.error(
                        "Error processing tick in strategy %s for token %s: %s",
                        type(strategy_instance).__name__,
                        instrument_token_str,
                        e,
                        exc_info=True,
                    )

    def stop_trading(self):
        with self._lock:
            if not self._is_running and not self.ws_client:
                logger.info("RealtimeTrader is not running.")
                return
            logger.info("Stopping RealtimeTrader...")
            if self.ws_client:
                tokens_to_unsubscribe_info = list(self.token_details_map.values())
                if tokens_to_unsubscribe_info:
                    self.ws_client.unsubscribe_tokens(tokens_to_unsubscribe_info)
                self.ws_client.disconnect()
                self.ws_client = None
            self.strategy_instances.clear()
            self.token_details_map.clear()
            self.active_deployments.clear()
            self._is_running = False

    def reload_deployments(self):
        with self._lock:
            if not self._is_running:
                logger.info("RealtimeTrader is not running. Cannot reload deployments.")
                return
            logger.info("Reloading deployments...")
            current_tokens_info = list(self.token_details_map.values())
            if current_tokens_info and self.ws_client and self.ws_client.is_connected():
                self.ws_client.unsubscribe_tokens(current_tokens_info)
            self.strategy_instances.clear()
            self.token_details_map.clear()
            self.active_deployments.clear()
        self._load_and_subscribe_deployments()
