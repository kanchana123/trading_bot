import pytest

from strategies.low_risk_llm_equity.strategy_sandbox import compile_strategy_code


SAFE_CODE = """
def calculate_indicators(df):
    return df

def should_buy(row, kernel_signal, news_context):
    return False

def should_sell(row, kernel_signal, news_context):
    return False
"""


def test_sandbox_compiles_safe_strategy():
    _, scope = compile_strategy_code(SAFE_CODE, {})
    assert callable(scope["should_buy"])
    assert scope["should_buy"](None, "HOLD", {}) is False


def test_sandbox_allows_pandas_import():
    code = """
import pandas as pd
def calculate_indicators(df):
    return df
def should_buy(row, kernel_signal, news_context):
    return False
def should_sell(row, kernel_signal, news_context):
    return False
"""
    _, scope = compile_strategy_code(code, {})
    assert "calculate_indicators" in scope


def test_sandbox_rejects_os_import():
    bad = SAFE_CODE.replace("return df", "import os\n    return df")
    with pytest.raises(ValueError):
        compile_strategy_code(bad, {})


def test_sandbox_rejects_eval():
    bad = """
def calculate_indicators(df):
    eval("1+1")
    return df
def should_buy(row, kernel_signal, news_context):
    return False
def should_sell(row, kernel_signal, news_context):
    return False
"""
    with pytest.raises(ValueError):
        compile_strategy_code(bad, {})


def test_sandbox_rejects_open_and_dunder_attr():
    bad_open = """
def calculate_indicators(df):
    open("/tmp/x", "w")
    return df
def should_buy(row, kernel_signal, news_context):
    return False
def should_sell(row, kernel_signal, news_context):
    return False
"""
    with pytest.raises(ValueError):
        compile_strategy_code(bad_open, {})


def test_sandbox_requires_all_three_functions():
    incomplete = """
def calculate_indicators(df):
    return df
"""
    with pytest.raises(ValueError, match="missing functions"):
        compile_strategy_code(incomplete, {})
