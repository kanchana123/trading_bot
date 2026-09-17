import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from functools import lru_cache
import random

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

# Evolutionary Algorithm
def evolutionary_optimizer(options, option_surfaces, stock_prices, time_points, population_size=30, generations=10, mutation_rate=0.2):
    def create_individual():
        indices = np.random.choice(len(options), size=np.random.randint(2, 6), replace=False)
        return list(indices)

    def crossover(parent1, parent2):
        cut = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child = list(set(parent1[:cut] + parent2[cut:]))
        return child

    def mutate(individual):
        if random.random() < mutation_rate:
            if len(individual) > 0 and random.random() < 0.5:
                individual.pop(random.randint(0, len(individual) - 1))
            else:
                new_gene = random.randint(0, len(options) - 1)
                if new_gene not in individual:
                    individual.append(new_gene)
        return individual

    population = [create_individual() for _ in range(population_size)]
    best_individual = None
    best_score = float('-inf')
    best_surface = None

    for gen in range(generations):
        scored_population = []
        for individual in population:
            surface = np.sum([option_surfaces[i] for i in individual], axis=0)
            score = fitness_3d_revenue(surface)
            scored_population.append((score, individual, surface))
        scored_population.sort(reverse=True, key=lambda x: x[0])
        best_candidate = scored_population[0]
        if best_candidate[0] > best_score:
            best_score, best_individual, best_surface = best_candidate
        top_individuals = [ind for _, ind, _ in scored_population[:population_size // 2]]
        population = top_individuals[:]
        while len(population) < population_size:
            parent1, parent2 = random.sample(top_individuals, 2)
            child = mutate(crossover(parent1, parent2))
            population.append(child)

    return best_score, [options[i] for i in best_individual], best_surface

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
    stock_prices = np.linspace(950, 1250, 40)
    expiries = [30, 60]  # in days
    r = 0.01
    sigma = 0.2

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
    time_points = np.linspace(0, max_days / 365, 20)
    option_surfaces = precompute_surfaces(option_chain, stock_prices, time_points)

    score, combo, surface = evolutionary_optimizer(option_chain, option_surfaces, stock_prices, time_points)
    title = f"[EA Payoff] Score: {score:.2f} - {len(combo)} options"
    print(title)
    for opt in combo:
        print(f"  {opt['position']} {opt['type']} - Strike: {opt['strike']}, Expiry: {opt['T']*365:.1f}d")
    plot_3d_surface(stock_prices, time_points, surface, title)
