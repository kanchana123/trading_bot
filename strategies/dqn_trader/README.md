# Goal
Create reinforcement learning model to trade based on candlestick chart with bollinger bands.
Here are the steps to follow:
1. Get 1 years data from angel api for one minute interval and generate images for training. The image will have candlestick chart with bollinger band, ranko chart, heikinishi chart. each image will have one day data. for example the data could be from yesterday's 12pm to today 12pm. then next image will be from yesterday 12:01 to today 12:01. After reading the data, go through the loop, slice data, make chart of the sliced data, and save chart with the timestamp. also save image name, with ohlc data, volume, technical indicators.
2. generate deep q network that will take image, time of the day, ohlc data, volume, other technical indicators as input and have 3 output actions: buy, hold, sell
3. processed images and data to train the network. 
4. action can be buy or hold if there is no current order, if there is current order then the action can be hold or sell.
5. evaluate the model.

File structure:
- generate data class
- generate network class
- data
- reinforcement training and testing class
- main.py 