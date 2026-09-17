try:
    from SmartApi import SmartConnect
except ImportError:
    SmartConnect = None
import pyotp
import pandas as pd
import os
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import math

from base_models.ohlc import normalize_ohlc_columns

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKEN_FILE = os.path.join(SCRIPT_DIR, "token_symbol_list.txt")
load_dotenv()

EXCHANGE_MAP = {
    "NSE_EQ": "NSE",
    "NSE_FO": "NFO",
    "NSE_CD": "CDS",
    "BSE_EQ": "BSE",
    "BSE_FO": "BFO",
    "MCX_FO": "MCX",
    "NSE": "NSE",
    "BSE": "BSE",
    "NFO": "NFO",
    "MCX": "MCX",
}


class AngelAPI:
    def __init__(self, connect: bool = True):
        """
        Initializes the AngelAPI class. Login is attempted when credentials are present.
        Missing credentials do not raise, so backtests and the API can start offline.
        """
        self.api_key = os.getenv("ANGEL_API_KEY")
        self.client_code = os.getenv("ANGEL_CLIENT_CODE")
        self.password = os.getenv("ANGEL_PASSWORD")
        self.totp = os.getenv("ANGEL_TOTP")
        self.smartApi = None
        self.feed_token = None
        self.jwt_token = None
        self.refreshToken = None

        self.INTERVALS_MAX_DAYS = {
            "ONE_MINUTE": 30,
            "THREE_MINUTE": 6,
            "FIVE_MINUTE": 30,
            "TEN_MINUTE": 30,
            "FIFTEEN_MINUTE": 30,
            "THIRTY_MINUTE": 30,
            "ONE_HOUR": 100,
            "ONE_DAY": 365,
        }

        if connect:
            if SmartConnect is None:
                logger.warning("smartapi-python is not installed. Broker features are disabled.")
            elif not all([self.api_key, self.client_code, self.password, self.totp]):
                logger.warning(
                    "Missing Angel API credentials. Broker features are disabled until they are set."
                )
            else:
                try:
                    self._connect()
                except Exception as e:
                    logger.error("Angel API login failed: %s", e, exc_info=True)

    def has_credentials(self) -> bool:
        return bool(all([self.api_key, self.client_code, self.password, self.totp]))

    def is_session_active(self) -> bool:
        return bool(self.smartApi and self.jwt_token)

    def login(self):
        """Establish or refresh a broker session."""
        if not self.has_credentials():
            raise ValueError("Missing one or more Angel API credentials in environment variables.")
        self._connect()
        return self.is_session_active()

    def _connect(self):
        if SmartConnect is None:
            raise ImportError("smartapi-python is not installed.")
        try:
            self.smartApi = SmartConnect(api_key=self.api_key)
            totp_obj = pyotp.TOTP(self.totp)
            totp_token = totp_obj.now()
            data = self.smartApi.generateSession(self.client_code, self.password, totp_token)

            if not data or data.get("status") is False:
                error_message = "Session generation returned None."
                if data:
                    error_message = data.get("message", "No error message in response.")
                raise Exception(
                    f"Angel API login failed. Please check your credentials. Error: {error_message}"
                )

            session_data = data.get("data") or {}
            self.feed_token = self.smartApi.getfeedToken() or session_data.get("feedToken")
            self.jwt_token = session_data.get("jwtToken")
            self.refreshToken = session_data.get("refreshToken")
            logger.info("Angel API connection successful.")
        except Exception as e:
            logger.error("Error connecting to Angel API: %s", e)
            self.smartApi = None
            self.jwt_token = None
            self.feed_token = None
            raise

    def _resolve_token_filepath(self, filepath=None) -> str:
        path = filepath or DEFAULT_TOKEN_FILE
        if os.path.isabs(path) and os.path.exists(path):
            return path
        candidate = os.path.join(SCRIPT_DIR, os.path.basename(path))
        if os.path.exists(candidate):
            return candidate
        return path

    def get_tokens_for_symbols(self, symbols, filepath=None):
        symbol_tokens = {}
        path = self._resolve_token_filepath(filepath)
        try:
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        token, symbol = parts
                        if symbol in symbols:
                            symbol_tokens[symbol] = token
        except FileNotFoundError:
            logger.error("Token file not found at %s", path)
            return {}
        except Exception as e:
            logger.error("An error occurred while reading the token file: %s", e)
            return {}
        return symbol_tokens

    def place_order(
        self,
        tradingsymbol: str,
        symboltoken: str,
        transaction_type: str,
        quantity: int,
        price: float,
        exchange: str = "NSE",
        order_type: str = "LIMIT",
        product_type: str = "INTRADAY",
        variety: str = "NORMAL",
    ) -> dict:
        if not self.is_session_active():
            return {"success": False, "error": "Angel API session is not active."}

        mapped_exchange = EXCHANGE_MAP.get(exchange, exchange)
        orderparams = {
            "variety": variety,
            "tradingsymbol": tradingsymbol,
            "symboltoken": str(symboltoken),
            "transactiontype": transaction_type.upper(),
            "exchange": mapped_exchange,
            "ordertype": order_type.upper(),
            "producttype": product_type.upper(),
            "duration": "DAY",
            "price": str(price if order_type.upper() != "MARKET" else "0"),
            "quantity": str(int(quantity)),
        }
        try:
            order_id = self.smartApi.placeOrder(orderparams)
            if not order_id:
                return {"success": False, "error": "Broker returned an empty order id."}
            return {"success": True, "broker_order_id": order_id}
        except Exception as e:
            logger.error("Failed to place Angel order: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}

    def download_historical_data(
        self,
        exchange,
        instrument_symbol,
        interval="ONE_MINUTE",
        days=None,
        from_date=None,
        to_date=None,
        filepath=None,
    ):
        if not self.is_session_active():
            logger.error("Not connected to Angel API. Cannot download data.")
            return None

        token_dict = self.get_tokens_for_symbols([instrument_symbol], filepath)
        instrument_token = token_dict.get(instrument_symbol) if token_dict else None
        if not instrument_token:
            logger.error("Could not get the token from file for %s", instrument_symbol)
            return None

        if days is not None:
            to_date_obj = datetime.now()
            from_date_obj = to_date_obj - timedelta(days=days)
        elif from_date is not None and to_date is not None:
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    from_date_obj = datetime.strptime(from_date, fmt)
                    to_date_obj = datetime.strptime(to_date, fmt)
                    break
                except ValueError:
                    from_date_obj = None
                    to_date_obj = None
            if from_date_obj is None:
                logger.error("Could not parse from_date/to_date: %s %s", from_date, to_date)
                return None
        else:
            logger.error("Provide either 'days' or both 'from_date' and 'to_date'.")
            return None

        max_days = self.INTERVALS_MAX_DAYS.get(interval, 30)
        total_days = max((to_date_obj - from_date_obj).days, 1)
        num_calls = math.ceil(total_days / max_days)
        data_frames = []

        for i in range(num_calls):
            current_from_date_obj = from_date_obj + timedelta(days=i * max_days)
            current_to_date_obj = min(
                to_date_obj, from_date_obj + timedelta(days=(i + 1) * max_days)
            )
            current_from_date = current_from_date_obj.strftime("%Y-%m-%d %H:%M")
            current_to_date = current_to_date_obj.strftime("%Y-%m-%d %H:%M")

            try:
                logger.info(
                    "Downloading historical data %s %s %s -> %s %s",
                    exchange,
                    instrument_token,
                    current_from_date,
                    current_to_date,
                    interval,
                )
                historical_data = self.smartApi.getCandleData(
                    {
                        "exchange": exchange,
                        "symboltoken": instrument_token,
                        "interval": interval,
                        "fromdate": current_from_date,
                        "todate": current_to_date,
                    }
                )

                if historical_data and historical_data.get("status") is True:
                    df = pd.DataFrame(historical_data["data"])
                    df.rename(
                        columns={
                            0: "Date",
                            1: "Open",
                            2: "High",
                            3: "Low",
                            4: "Close",
                            5: "Volume",
                        },
                        inplace=True,
                    )
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                    df = normalize_ohlc_columns(df)
                    data_frames.append(df)
                else:
                    message = (
                        historical_data.get("message", "Unknown error")
                        if historical_data
                        else "Empty response"
                    )
                    logger.error("Error getting candle data: %s", message)
                    return None
            except Exception as e:
                logger.error("Error downloading historical data: %s", e)
                return None

        if not data_frames:
            return None

        combined_df = pd.concat(data_frames)
        combined_df = combined_df[~combined_df.index.duplicated(keep="first")]
        return combined_df


if __name__ == "__main__":
    angel_api = AngelAPI()
    exchange = "NSE"
    instrument_tokens = ["ICICIBANK-EQ"]
    interval = "ONE_MINUTE"
    days = 40
    data = {}
    for symbol in instrument_tokens:
        data[symbol] = angel_api.download_historical_data(
            exchange, symbol, interval=interval, days=days
        )
    if data:
        print("historical data:", instrument_tokens)
        print(data)
