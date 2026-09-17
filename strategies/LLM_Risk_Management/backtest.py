def backtest(df, initial_capital=100000):
    """
    Backtests the trading strategy.

    Args:
        df: pandas DataFrame with OHLCV data and indicators.
        initial_capital: The initial capital for the portfolio.

    Returns:
        pandas DataFrame with order history.
    """
    portfolio = {"cash": initial_capital, "holdings": {}}
    current_positions = {}
    order_history = []
    
    for i in range(len(df)):
        current_data = df.iloc[i]
        
        # Prepare data for LLM
        current_portfolio = portfolio.copy()
        current_portfolio["holdings"] = current_positions.copy()
        current_news = ["No news available"]  # Replace with actual news if available
        current_ohlcv = {
            "ICICIBANK": {
                "open": df["Open"].iloc[max(0, i - 2):i].tolist(),
                "high": df["High"].iloc[max(0, i - 2):i].tolist(),
                "low": df["Low"].iloc[max(0, i - 2):i].tolist(),
                "close": df["Close"].iloc[max(0, i - 2):i].tolist(),
                "volume": df["Volume"].iloc[max(0, i - 2):i].tolist(),
            }
        }

        # Get LLM advice
        llm_response = get_llm_trading_advice(current_portfolio, current_positions, current_news, current_ohlcv)
        orders = parse_llm_response(llm_response)

        # Process orders
        for order in orders:
            if order["stock"] == "ICICIBANK":
                current_price = current_data["Close"]
                order_type = order["buy/sell"]
                entry_price = order["entry_price"]
                stop_loss = order["stop_loss"]
                
                if order_type == "buy":
                    if portfolio["cash"] >= entry_price:
                        quantity = int(portfolio["cash"] / entry_price)
                        portfolio["cash"] -= quantity * entry_price
                        current_positions["ICICIBANK"] = {
                            "quantity": quantity,
                            "avg_price": entry_price,
                            "stop_loss": stop_loss
                        }
                        order_history.append({
                            "Date": df.index[i],
                            "Stock": "ICICIBANK",
                            "Type": "buy",
                            "Entry Price": entry_price,
                            "Stop Loss": stop_loss,
                            "Quantity": quantity,
                            "Explanation": order["explanation"],
                            "Portfolio Value": portfolio["cash"] + (current_positions.get("ICICIBANK", {}).get("quantity", 0) * current_price)
                        })
                elif order_type == "sell":
                    if "ICICIBANK" in current_positions:
                        quantity = current_positions["ICICIBANK"]["quantity"]
                        portfolio["cash"] += quantity * entry_price
                        del current_positions["ICICIBANK"]
                        order_history.append({
                            "Date": df.index[i],
                            "Stock": "ICICIBANK",
                            "Type": "sell",
                            "Entry Price": entry_price,
                            "Stop Loss": stop_loss,
                            "Quantity": quantity,
                            "Explanation": order["explanation"],
                            "Portfolio Value": portfolio["cash"] + (current_positions.get("ICICIBANK", {}).get("quantity", 0) * current_price)
                        })
                
                # Check stop loss
                if "ICICIBANK" in current_positions:
                    if current_price <= current_positions["ICICIBANK"]["stop_loss"]:
                        quantity = current_positions["ICICIBANK"]["quantity"]
                        portfolio["cash"] += quantity * current_price
                        del current_positions["ICICIBANK"]
                        order_history.append({
                            "Date": df.index[i],
                            "Stock": "ICICIBANK",
                            "Type": "stop_loss_sell",
                            "Entry Price": current_price,
                            "Stop Loss": current_positions["ICICIBANK"]["stop_loss"],
                            "Quantity": quantity,
                            "Explanation": "Stop loss triggered",
                            "Portfolio Value": portfolio["cash"] + (current_positions.get("ICICIBANK", {}).get("quantity", 0) * current_price)
                        })

    return pd.DataFrame(order_history)
