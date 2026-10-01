import requests
from bs4 import BeautifulSoup

def fetch_data(url):
    """
    Fetch data from Google doc and parse it into a list of points with their coordinates and characters.

    Args:
        url (str): URL of the Google doc.
    
    Returns:
        list: List of points with their coordinates and characters.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        rows = soup.find_all('tr')[1:]
        
        points = []
        max_x = max_y = 0

        for row in rows: 
            cols = row.find_all('td')
            x = int(cols[0].text.strip())
            y = int(cols[2].text.strip())
            character = cols[1].text.strip()
            points.append((x, y, character))
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        return points, max_x, max_y
    
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    
def decode_msg(url):
    """
    Decode the message from a Google Doc URL.

    Args:
        url (str): URL of the Google doc containing the message data.
    
    Returns:
        None: Prints the decoded message in a grid format.
    """
    data = fetch_data(url)

    if not data:
        return None
    
    points, max_x, max_y = data

    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]  

    for x, y, char in points:
        grid[y][x] = char

    for row in grid:
        print(''.join(row))

decode_msg("https://docs.google.com/document/d/e/2PACX-1vRMx5YQlZNa3ra8dYYxmv-QIQ3YJe8tbI3kqcuC7lQiZm-CSEznKfN_HYNSpoXcZIV3Y_O3YoUB1ecq/pub")
