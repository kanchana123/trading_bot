import json
import os
import requests
from typing import Dict, List, Tuple
from huggingface_hub import HfApi

HF_API_TOKEN = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

def get_llm_trading_advice_hf(
    portfolio: Dict, current_positions: Dict, news_headlines: List[str], ohlcv_data: Dict
) -> str:
    """
    Gets trading advice from the Gemma 7B Instruct model using the Hugging Face Inference API.

    Args:
        portfolio: Dictionary representing the current portfolio.
        current_positions: Dictionary representing current open positions.
        news_headlines: List of news headlines.
        ohlcv_data: Dictionary of OHLCV data for various stocks.

    Returns:
        A string representing the LLM's trading advice in a specific format.
    """
    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

    # api_url = "https://api-inference.huggingface.co/models/google/gemma-3-27b-it"  # Changed the model
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

    # Construct the prompt for the LLM
    prompt = f"""
    You are a financial expert and a trading bot with a goal to make profit of at least 100 Rs with complete buy-sell trade. You are only allowed to trade the stock ICICIBANK by considering given historical data in the prompt. Analyze the following information and provide trading orders in JSON format.
    Each order should include:
    - stock: The stock ticker symbol (e.g., ICICIBANK).
    - transaction_type: The type of order (e.g., limit, market, stop).
    - buy/sell: Whether to buy or sell.
    - entry_price: The price at which to enter the trade.
    - quantity: The quantity of the stock to buy or sell. Maximum quantity is 20.
    - stop_loss: The stop-loss price.
    - trend_confidence: percentage of confidence in the upward or downward future trend
    - explanation: A brief explanation of the reasoning behind the order and how it helps with risk management.

    Stock: ICICIBANK

    Portfolio: {portfolio}
    Current Positions: {current_positions}
    News Headlines: {news_headlines}
    OHLCV Data: {ohlcv_data}

    Respond only with a JSON array of orders for ICICIBANK.
    If it is not good time to buy or sell response with empty JSON array.
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,  # Adjust as needed
            "temperature": 0.7,  # Adjust for creativity/determinism
            "top_p": 0.9,  # Adjust for diversity
            "return_full_text": False,
        },
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        llm_response = response.json()[0]["generated_text"]
        return llm_response
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return "[]"  # Return an empty list if there's an error
    except (KeyError, IndexError) as e:
        print(f"Error parsing API response: {e}")
        return "[]"
