import numpy as np
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from functools import lru_cache

# Black-Scholes option pricing with caching
@lru_cache(maxsize=None)
def cached_black_scholes(S, K, T, r, sigma, option_type):
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Precompute surfaces for all options
def precompute_surfaces(options, stock_prices, time_points):
    surfaces = []
    for opt in options:
        surface = np.zeros((len(time_points), len(stock_prices)))
        for i, t in enumerate(time_points):
            T_remaining = max(opt['T'] - t, 0)
            for j, S in enumerate(stock_prices):
                if T_remaining <= 0:
                    payoff = max(S - opt['strike'], 0) if opt['type'] == 'call' else max(opt['strike'] - S, 0)
                    surface[i, j] = payoff if opt['position'] == 'long' else -payoff
                else:
                    price = cached_black_scholes(S, opt['strike'], T_remaining, opt['r'], opt['sigma'], opt['type'])
                    surface[i, j] = price if opt['position'] == 'long' else -price
        surfaces.append(surface)
    return surfaces

# Fitness function for 3D surface
def fitness_3d_revenue(surface):
    min_revenue = np.min(surface)
    avg_revenue = np.mean(surface)
    return avg_revenue + 5 * min_revenue

# BFS optimizer with memoized partial surfaces
def bfs_optimizer_3d(options, option_surfaces, stock_prices, time_points, beam_width=3, max_depth=4):
    queue = [([], np.zeros((len(time_points), len(stock_prices))))]
    best_results = []
    for depth in range(max_depth):
        candidates = []
        for combo, surface in queue:
            used_ids = {id(opt) for opt in combo}
            for idx, opt in enumerate(options):
                if id(opt) in used_ids:
                    continue
                new_combo = combo + [opt]
                new_surface = surface + option_surfaces[idx]
                score = fitness_3d_revenue(new_surface)
                candidates.append((score, new_combo, new_surface))
        top_candidates = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])
        queue = [(combo, surface) for _, combo, surface in top_candidates]
        best_results.append(top_candidates[0])
    return best_results

# 3D visualization of payoff surface
def plot_3d_surface(stock_prices, time_points, surface, title):
    X, Y = np.meshgrid(stock_prices, time_points)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y * 365, surface, cmap='viridis')
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Days Held")
    ax.set_zlabel("Portfolio Value")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Sample usage
if __name__ == '__main__':
    current_price = 1100
    stock_prices = np.linspace(950, 1250, 40)  # Reduced for speed
    expiries = [30, 60]  # in days
    r = 0.01
    sigma = 0.2

    # Generate option chain
    option_chain = []
    strikes = np.linspace(current_price - 200, current_price + 200, 25)
    for strike in strikes:
        for expiry in expiries:
            T = expiry / 365
            for typ in ['call', 'put']:
                for position in ['long', 'short']:
                    option_chain.append({
                        'strike': strike,
                        'T': T,
                        'r': r,
                        'sigma': sigma,
                        'type': typ,
                        'position': position
                    })

    max_days = max(expiries)
    time_points = np.linspace(0, max_days / 365, 20)  # Reduced for speed
    option_surfaces = precompute_surfaces(option_chain, stock_prices, time_points)
    results = bfs_optimizer_3d(option_chain, option_surfaces, stock_prices, time_points)

    for i, (score, combo, surface) in enumerate(results):
        title = f"[3D Payoff] Depth {i+1} - Score: {score:.2f} - {len(combo)} options"
        print(title)
        for opt in combo:
            print(f"  {opt['position']} {opt['type']} - Strike: {opt['strike']}, Expiry: {opt['T']*365:.1f}d")
        plot_3d_surface(stock_prices, time_points, surface, title)
