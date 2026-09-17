## Goal
I want to train agent to learn intraday trading strategy

Agent should take chart image and ohlcv, different strategy trading signals, and current position as input and output should be buy, hold, sell.

image size is 100x100
open, high, low, close, volume, close_H, open_R, close_R, bb_upper, bb_middle, bb_lower, coppock, trix, mfi, rsi, macd1, macd2, supertrend, time_of_day, portfolio, current_position

network should use CNN, RNN, and transformers. it should see candlestick patterns and associate it with actions and rewards. it should be able to remember the previous candles and take action accordingly.

Agent should run 100 episodes on 1 days data, store state-actions-next_state-rewards in experience store 

each episode should run intraday trading session, with starting balance, buy, hold or sell based on current position and balance. it should choose action using epsilon greedy strategy

generate training data and train the network at the end of the episode

while generating trading data:
    - create environment
    - read image file and transform images, store images in environment
    - one episode
        - generate greedy action for each state or row in dataframe directly passing entire dataframe to model in complete batch
        - loop through dataframe
            - based on epsilon, choose greedy action from row or random action, choose valid action
            - based on action, track current position, current balance and calculate reward
            - update current position, current balance, reward at row index
            - at the end of the episode sell and make sure there is no current position
    - train network from training data from episode
