import logging
import json
import time
from typing import Callable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2 as SmartWebSocket
except ImportError:
    try:
        from smartapi import SmartWebSocket
    except ImportError:
        SmartWebSocket = None


class AngelWebSocketClient:
    def __init__(
        self,
        auth_token: str,
        api_key: str,
        client_code: str,
        feed_token: str,
        tick_callback: Callable,
        on_open_callback: Optional[Callable] = None,
    ):
        self.auth_token = auth_token
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self.tick_callback = tick_callback
        self.on_open_callback = on_open_callback
        self.sws = None
        self.subscribed_tokens = set()
        self._pending_token_info: List[dict] = []
        self._connected = False
        self._should_run = True
        self._reconnect_delay = 5

    def is_connected(self) -> bool:
        return self._connected and self.sws is not None

    def connect(self):
        """Connect and stay connected, reconnecting until disconnect() is called."""
        if SmartWebSocket is None:
            raise ImportError(
                "SmartWebSocket is not installed. Install smartapi-python to use realtime trading."
            )
        while self._should_run:
            try:
                self._create_socket()
                logger.info("Attempting to connect to Angel WebSocket...")
                self.sws.connect()
            except Exception as e:
                logger.error("Error connecting to Angel WebSocket: %s", e, exc_info=True)
                self._connected = False
            if self._should_run:
                logger.info("Reconnecting WebSocket in %s seconds...", self._reconnect_delay)
                time.sleep(self._reconnect_delay)

    def _create_socket(self):
        self.sws = SmartWebSocket(
            self.auth_token, self.api_key, self.client_code, self.feed_token
        )
        self.sws.on_open = self._on_open
        self.sws.on_data = self._on_data
        self.sws.on_error = self._on_error
        self.sws.on_close = self._on_close

    def _on_open(self, *args):
        self._connected = True
        logger.info("Angel WebSocket connection opened.")
        tokens = list(self._pending_token_info)
        if tokens:
            self.subscribe_tokens(tokens)
        if self.on_open_callback:
            try:
                self.on_open_callback()
            except Exception as e:
                logger.error("Error in websocket on_open callback: %s", e, exc_info=True)

    def _on_data(self, *args):
        message = args[-1] if args else None
        if isinstance(message, list):
            for tick in message:
                self._emit_tick(tick)
        elif isinstance(message, dict):
            self._emit_tick(message)
        elif isinstance(message, (bytes, str)):
            try:
                parsed = json.loads(message)
                self._on_data(parsed)
            except (TypeError, json.JSONDecodeError):
                logger.debug("Unhandled websocket payload: %s", message)

    def _emit_tick(self, tick):
        if not isinstance(tick, dict):
            return
        if tick.get("tk") or tick.get("token") or tick.get("symbolToken"):
            if "token" not in tick and "tk" in tick:
                tick["token"] = tick["tk"]
            if "last_traded_price" not in tick and "ltp" in tick:
                tick["last_traded_price"] = tick["ltp"]
            self.tick_callback(tick)

    def _on_error(self, *args):
        logger.error("Angel WebSocket error: %s", args[-1] if args else "unknown")

    def _on_close(self, *args):
        self._connected = False
        logger.info("Angel WebSocket connection closed: %s", args)

    def subscribe_tokens(self, token_info_list: list):
        self._pending_token_info = list(token_info_list or [])
        if not self.is_connected():
            logger.warning("WebSocket not connected yet. Tokens will be subscribed on open.")
            return

        exchange_map = {
            "NSE_EQ": 1,
            "NSE_FO": 2,
            "NSE_CD": 7,
            "BSE_EQ": 3,
            "BSE_FO": 4,
            "MCX_FO": 5,
            "NSE": 1,
            "NFO": 2,
            "BSE": 3,
            "MCX": 5,
        }

        api_token_list_by_exchange = {}
        for info in token_info_list:
            exchange_type = exchange_map.get(str(info.get("exchange", "")).upper())
            if exchange_type is None:
                logger.warning(
                    "Unknown exchange segment: %s for token %s. Skipping.",
                    info.get("exchange"),
                    info.get("instrumentToken"),
                )
                continue
            api_token_list_by_exchange.setdefault(exchange_type, []).append(
                str(info["instrumentToken"])
            )

        final_token_list_for_api = [
            {"exchangeType": ex_type, "tokens": tokens}
            for ex_type, tokens in api_token_list_by_exchange.items()
        ]
        if not final_token_list_for_api:
            logger.info("No valid tokens to subscribe.")
            return

        try:
            correlation_id = f"sub_{self.client_code}_{pd.Timestamp.now().timestamp()}"
            mode = 1
            logger.info(
                "Subscribing with mode %s to: %s",
                mode,
                json.dumps(final_token_list_for_api),
            )
            self.sws.subscribe(correlation_id, mode, final_token_list_for_api)
            for item_list in final_token_list_for_api:
                for token_str in item_list["tokens"]:
                    self.subscribed_tokens.add(token_str)
        except Exception as e:
            logger.error("Error during token subscription: %s", e, exc_info=True)

    def unsubscribe_tokens(self, token_info_list: list):
        if not self.is_connected():
            logger.warning("WebSocket not connected. Cannot unsubscribe.")
            return

        exchange_map = {
            "NSE_EQ": 1,
            "NSE_FO": 2,
            "BSE_EQ": 3,
            "MCX_FO": 5,
            "NSE": 1,
            "NFO": 2,
            "BSE": 3,
            "MCX": 5,
        }
        api_token_list_by_exchange = {}
        for info in token_info_list:
            exchange_type = exchange_map.get(str(info.get("exchange", "")).upper())
            if exchange_type is None:
                continue
            api_token_list_by_exchange.setdefault(exchange_type, []).append(
                str(info["instrumentToken"])
            )
        final_token_list_for_api = [
            {"exchangeType": ex_type, "tokens": tokens}
            for ex_type, tokens in api_token_list_by_exchange.items()
        ]
        if not final_token_list_for_api:
            return

        try:
            correlation_id = f"unsub_{self.client_code}_{pd.Timestamp.now().timestamp()}"
            mode = 1
            self.sws.unsubscribe(correlation_id, mode, final_token_list_for_api)
            for item_list in final_token_list_for_api:
                for token_str in item_list["tokens"]:
                    self.subscribed_tokens.discard(token_str)
        except Exception as e:
            logger.error("Error unsubscribing from tokens: %s", e, exc_info=True)

    def disconnect(self):
        self._should_run = False
        self._connected = False
        if self.sws:
            try:
                if hasattr(self.sws, "close_connection"):
                    self.sws.close_connection()
                elif hasattr(self.sws, "close"):
                    self.sws.close()
            except Exception as e:
                logger.error("Error closing websocket: %s", e)
        self.subscribed_tokens.clear()
