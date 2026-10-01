from strategies.action_price.KernelTrader import KernelBacktestAdapter, KernelStrategy
from strategies.bollingerBandStrategy import BollingerBandStrategy
from strategies.low_risk_combo import LowRiskComboStrategy
from strategies.rsi_reversion import RSIReversionStrategy
from strategies.sma_crossover import SMACrossoverStrategy

STRATEGY_CATALOG = {
    "BollingerBand": {
        "label": "Bollinger mean reversion",
        "backtest": BollingerBandStrategy,
        "realtime": BollingerBandStrategy,
        "modes": ["backtest", "virtual", "real"],
        "default_params": {
            "window": 20,
            "std_multiplier": 2,
            "position_fraction": 0.2,
            "trend_sma": 50,
            "rsi_period": 14,
            "rsi_max_buy": 55,
            "stop_loss_pct": 1.0,
            "bar_interval_minutes": 1,
        },
    },
    "KernelMomentum": {
        "label": "Kernel momentum breakout",
        "backtest": KernelBacktestAdapter,
        "realtime": KernelStrategy,
        "modes": ["backtest", "virtual", "real"],
        "default_params": {
            "kernel": [-2, -1, 0, 1, 2],
            "threshold": 1.5,
            "quantity": 1,
            "stop_loss_pct": 0.8,
            "bar_interval_minutes": 1,
        },
    },
    "SMACrossover": {
        "label": "SMA crossover trend follow",
        "backtest": SMACrossoverStrategy,
        "realtime": SMACrossoverStrategy,
        "modes": ["backtest", "virtual", "real"],
        "default_params": {
            "fast": 10,
            "slow": 30,
            "quantity": 1,
            "stop_loss_pct": 1.5,
            "bar_interval_minutes": 1,
        },
    },
    "RSIReversion": {
        "label": "RSI oversold/overbought reversion",
        "backtest": RSIReversionStrategy,
        "realtime": RSIReversionStrategy,
        "modes": ["backtest", "virtual", "real"],
        "default_params": {
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "trend_sma": 50,
            "quantity": 1,
            "stop_loss_pct": 1.0,
            "bar_interval_minutes": 1,
        },
    },
    "LowRiskCombo": {
        "label": "Kernel + Bollinger + RSI filter",
        "backtest": LowRiskComboStrategy,
        "realtime": LowRiskComboStrategy,
        "modes": ["backtest", "virtual", "real"],
        "default_params": {
            "window": 20,
            "std_multiplier": 2,
            "kernel": [-2, -1, 0, 1, 2],
            "threshold": 1.5,
            "rsi_period": 14,
            "rsi_max_buy": 55,
            "rsi_sell": 70,
            "position_fraction": 0.15,
            "stop_loss_pct": 0.5,
            "bar_interval_minutes": 1,
        },
    },
}

BACKTEST_STRATEGY_CLASSES = {
    key: spec["backtest"] for key, spec in STRATEGY_CATALOG.items()
}
STRATEGY_CLASS_MAP = {
    key: spec["realtime"] for key, spec in STRATEGY_CATALOG.items()
}


def catalog_payload():
    return [
        {
            "name": key,
            "label": spec["label"],
            "modes": spec["modes"],
            "default_params": spec["default_params"],
        }
        for key, spec in STRATEGY_CATALOG.items()
    ]


def supports_realtime(class_name: str) -> bool:
    spec = STRATEGY_CATALOG.get(class_name) or {}
    return "virtual" in spec.get("modes", []) or "real" in spec.get("modes", [])
