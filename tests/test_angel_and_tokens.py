from base_models.angel_api import AngelAPI
from base_models.get_symboltoken import get_tokens_for_symbols


def test_angel_api_starts_offline_without_credentials(monkeypatch):
    for key in ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_PASSWORD", "ANGEL_TOTP"):
        monkeypatch.delenv(key, raising=False)
    api = AngelAPI(connect=True)
    assert api.is_session_active() is False
    assert api.has_credentials() is False
    assert api.download_historical_data("NSE", "ICICIBANK-EQ", days=1) is None
    result = api.place_order(
        tradingsymbol="ICICIBANK-EQ",
        symboltoken="1",
        transaction_type="buy",
        quantity=1,
        price=100,
    )
    assert result["success"] is False


def test_get_tokens_for_symbols_from_file(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("2885 RELIANCE-EQ\n4963 ICICIBANK-EQ\n")
    found = get_tokens_for_symbols(["ICICIBANK-EQ", "MISSING-EQ"], filepath=str(path))
    assert found == {"ICICIBANK-EQ": "4963"}


def test_get_tokens_missing_file_returns_empty(tmp_path):
    assert get_tokens_for_symbols(["X"], filepath=str(tmp_path / "nope.txt")) == {}
