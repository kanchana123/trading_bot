Most simple and low risk strategy
- create a strategy that will analyze 5 equity stocks data
- measure technical indicators and identify signals
- use kernel method to create a signal
- get real-time news sentiment
- create llm prompt with llm's goal, portfolio value, current position, each stock with its ohlcv data, technical indicators, kernel method signal, news sentiment
- llm output in json to buy or sell orders with stop loss, target, entry/exit price, note about current situation and expectation
- place orders save orders in database with flag real order, virtual, backtest

create main.py that will import this modules, get stock names, get their historical data, calculate technical indicators, kernel method, backtest, while backtest get news sentiment and generate llm prompt, call llm api, place orders, save orders in csv file first once strategy is called from the dashboard to backtest then save orders to database. 