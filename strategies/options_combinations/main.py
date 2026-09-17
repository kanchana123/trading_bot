import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

# Black-Scholes function for European Call option price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Black-Scholes function for European Put option price
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Generate synthetic options data (mock)
def generate_mock_options():
    strikes = [1000, 1050, 1100, 1150, 1200]  # Example strikes
    expirations = [30, 60, 90]  # Expirations in days
    option_types = ['call', 'put']  # Option types
    options = []
    
    for strike in strikes:
        for exp in expirations:
            for opt_type in option_types:
                options.append({'strike': strike, 'type': opt_type, 'T': exp / 365, 'r': 0.01, 'sigma': 0.2})
    
    return options

# Simulate the payoff surface over stock prices and time
def simulate_payoff(options, stock_prices, times):
    payoff_surface = np.zeros((len(stock_prices), len(times)))
    
    for opt in options:
        for i, stock_price in enumerate(stock_prices):
            for j, time in enumerate(times):
                if opt['type'] == 'call':
                    payoff_surface[i, j] += max(stock_price - opt['strike'], 0)
                else:
                    payoff_surface[i, j] += max(opt['strike'] - stock_price, 0)
    
    return payoff_surface

# Plotting function for 3D surface
def plot_3d_surface(stock_prices, times, payoff_surface, ax):
    X, Y = np.meshgrid(times, stock_prices)
    ax.clear()
    ax.plot_surface(X, Y, payoff_surface, cmap='viridis')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_zlabel('Payoff')
    ax.set_title('3D Payoff Surface')
    plt.draw()
    plt.pause(0.5)

# Main function to run the valley-growing process
def grow_valley():
    # Initialize stock prices, times, and the surface
    current_stock_price = 1100  # Current stock price (you can change this as needed)
    stock_prices = np.linspace(current_stock_price - 100, current_stock_price + 100, 50)  # Stock price range around current stock price
    times = np.linspace(0, 1, 30)  # Time from 0 to 1 year (in years)
    
    # Generate mock options
    options = generate_mock_options()
    
    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize the payoff surface as flat
    payoff_surface = np.zeros((len(stock_prices), len(times)))
    
    # Iterate over adding options to create the "valley"
    for i in range(1, len(options) + 1):
        current_options = options[:i]
        payoff_surface = simulate_payoff(current_options, stock_prices, times)
        
        # Print the current option positions
        print(f"Step {i}: Current Option Positions:")
        for opt in current_options:
            print(f"  {opt['type'].capitalize()} Option: Strike = {opt['strike']}, T = {opt['T']*365} days, Sigma = {opt['sigma']}")
        
        # Update the plot to show the valley growing
        plot_3d_surface(stock_prices, times, payoff_surface, ax)
    
    plt.show()

# Run the valley-growing simulation
grow_valley()
