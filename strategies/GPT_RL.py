import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class TradingEnvironment:
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        return self._get_state()
    
    def _get_state(self) -> Dict:
        """Create a market state description for the model"""
        current_data = self.data.iloc[self.current_step]
        
        # Calculate technical indicators
        recent_prices = self.data.iloc[max(0, self.current_step-20):self.current_step+1]
        sma_20 = recent_prices['close'].mean()
        rsi = self._calculate_rsi(recent_prices['close'])
        
        state_description = f"""
        Current Market State:
        Price: ${current_data['close']:.2f}
        24h Change: {((current_data['close'] / current_data['open'] - 1) * 100):.2f}%
        Volume: ${current_data['volume']:.2f}
        RSI: {rsi:.2f}
        20 SMA: ${sma_20:.2f}
        
        Portfolio State:
        Current Balance: ${self.balance:.2f}
        Position: {'Long' if self.position == 1 else 'Short' if self.position == -1 else 'None'}
        
        What action should be taken? Respond with 'BUY', 'SELL', or 'HOLD'.
        """
        return state_description
    
    def _calculate_rsi(self, prices, periods=14):
        # Simple RSI calculation
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        prev_value = self.balance + self.position * self.data.iloc[self.current_step]['close']
        
        # Execute action
        current_price = self.data.iloc[self.current_step]['close']
        if action == 'BUY' and self.position <= 0:
            self.position = 1
            self.balance -= current_price
            self.trades.append(('BUY', current_price))
        elif action == 'SELL' and self.position >= 0:
            self.position = -1
            self.balance += current_price
            self.trades.append(('SELL', current_price))
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calculate reward
        current_value = self.balance + self.position * current_price
        reward = (current_value - prev_value) / prev_value  # Percentage return
        self.portfolio_values.append(current_value)
        
        return self._get_state(), reward, done, {'portfolio_value': current_value}

class TradingRL:
    def __init__(self, model_name: str = 'gpt2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Add special tokens for trading
        special_tokens = ['<|BUY|>', '<|SELL|>', '<|HOLD|>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def get_action(self, state: str) -> str:
        """Get trading action from the model"""
        inputs = self.tokenizer.encode(state, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 20,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.7,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract action from response
        if '<|BUY|>' in response:
            return 'BUY'
        elif '<|SELL|>' in response:
            return 'SELL'
        else:
            return 'HOLD'
    
    def train_step(self, state: str, action: str, reward: float, next_state: str):
        """Update model using PPO"""
        inputs = self.tokenizer.encode(state, return_tensors='pt').to(self.device)
        action_tokens = self.tokenizer.encode(f'<|{action}|>', return_tensors='pt').to(self.device)
        
        # Calculate action probabilities
        with torch.no_grad():
            old_outputs = self.model(inputs)
            old_probs = F.softmax(old_outputs.logits[:, -1, :], dim=-1)
        
        # PPO update
        outputs = self.model(inputs)
        probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
        
        # Calculate PPO loss
        ratio = probs / (old_probs + 1e-8)
        clip_param = 0.2
        surr1 = ratio * reward
        surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * reward
        loss = -torch.min(surr1, surr2).mean()
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def train_trading_bot(
    env: TradingEnvironment,
    agent: TradingRL,
    episodes: int = 100,
    learning_rate: float = 1e-5
):
    """Train the trading bot using RL"""
    agent.optimizer = torch.optim.AdamW(agent.model.parameters(), lr=learning_rate)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        trades = []
        
        while not done:
            # Get action from model
            action = agent.get_action(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Train model
            loss = agent.train_step(state, action, reward, next_state)
            
            episode_reward += reward
            trades.append((action, info['portfolio_value']))
            state = next_state
            
        # Episode summary
        print(f"Episode {episode + 1}/{episodes}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Final Portfolio Value: ${info['portfolio_value']:.2f}")
        print("------------------------")

# Real-time trading implementation
class RealTimeTrader:
    def __init__(self, agent: TradingRL, exchange_api):
        self.agent = agent
        self.exchange = exchange_api
        self.position = 0
        
    def get_market_data(self) -> Dict:
        """Fetch current market data from exchange"""
        # Implement based on your exchange API
        pass
    
    def create_state_description(self, market_data: Dict) -> str:
        """Create state description for the model"""
        # Similar to TradingEnvironment._get_state
        pass
    
    def execute_trade(self, action: str):
        """Execute trade on exchange"""
        # Implement based on your exchange API
        pass
    
    def trade_loop(self, interval: int = 60):
        """Main trading loop"""
        while True:
            try:
                # Get current market data
                market_data = self.get_market_data()
                
                # Create state description
                state = self.create_state_description(market_data)
                
                # Get model's action
                action = self.agent.get_action(state)
                
                # Execute trade if needed
                self.execute_trade(action)
                
                # Wait for next interval
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait before retrying

# Example usage
def main():
    # Load historical data
    data = pd.read_csv('historical_data.csv')
    
    # Create environment and agent
    env = TradingEnvironment(data)
    agent = TradingRL()
    
    # Train the agent
    train_trading_bot(env, agent)
    
    # Save the trained model
    agent.model.save_pretrained('trained_trading_model')
    
    # Initialize real-time trader
    # exchange_api = initialize_exchange_api()  # Your exchange API implementation
    # trader = RealTimeTrader(agent, exchange_api)
    # trader.trade_loop()

if __name__ == "__main__":
    main()
