import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKEN_FILE = os.path.join(SCRIPT_DIR, "token_symbol_list.txt")


def get_tokens_for_symbols(symbols, filepath=None):
    """
    Reads a file containing token and symbol pairs, and returns the tokens
    for the given symbols.

    Args:
        symbols (list): A list of symbols (strings) for which to retrieve tokens.
        filepath (str): The path to the file containing token-symbol mappings.

    Returns:
        dict: A dictionary where keys are the input symbols and values are their
              corresponding tokens. If a symbol is not found, it will not be included
              in the output dictionary.
    """
    path = filepath or DEFAULT_TOKEN_FILE
    symbol_tokens = {}

    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    token, symbol = parts  # Corrected order: token first, then symbol
                    if symbol in symbols:
                        symbol_tokens[symbol] = token
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return {}
    return symbol_tokens


# Example usage (you can put this in a separate test file):
if __name__ == "__main__":
    filepath = f'{os.path.abspath(".")}/base_models/token_symbol_list.txt'

    # Example symbols
    symbols_to_find = [
        "ICICIBANK-EQ",
        "RELIANCE-EQ",
        "TCS-EQ",
        "XYZ-EQ",  # Example of a symbol that doesn't exist
        "SBIN-EQ",
    ]

    tokens = get_tokens_for_symbols(symbols_to_find, filepath)

    if tokens:
        print("Tokens found:")
        for symbol, token in tokens.items():
            print(f"  {symbol}: {token}")
    else:
        print("No tokens found or an error occurred.")

