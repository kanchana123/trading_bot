import pandas as pd
import pytest

from strategies.low_risk_llm_equity.kernel_signal_generator import KernelSignalGenerator
from tests.helpers import ohlc_frame


def test_kernel_signal_generator_is_causal():
    data = ohlc_frame(n=30)
    gen = KernelSignalGenerator(params={"kernel": [-2, -1, 0, 1, 2], "threshold": 0.1})
    original = gen.process_data(data.copy())["kernel_momentum"].copy()

    mutated = data.copy()
    mutated.iloc[-1, mutated.columns.get_loc("Close")] = 999
    updated = gen.process_data(mutated)["kernel_momentum"]
    pd.testing.assert_series_equal(
        original.iloc[:-1].reset_index(drop=True),
        updated.iloc[:-1].reset_index(drop=True),
    )


def test_kernel_signal_generator_buy_and_sell_crossover():
    gen = KernelSignalGenerator(params={"threshold": 1.5})
    gen.data_with_momentum = pd.DataFrame(
        {"kernel_momentum": [0.0, -1.0, -2.0, -1.0, 1.0, 2.0]}
    )
    assert gen.should_buy(2)
    assert not gen.should_sell(2)
    assert gen.should_sell(5)
    assert not gen.should_buy(5)


def test_kernel_signal_generator_rejects_empty_and_missing_columns():
    gen = KernelSignalGenerator()
    with pytest.raises(ValueError, match="empty"):
        gen.process_data(pd.DataFrame())
    with pytest.raises(ValueError, match="required"):
        gen.process_data(pd.DataFrame({"Close": [1, 2, 3]}))
