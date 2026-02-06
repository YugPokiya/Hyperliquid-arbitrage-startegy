import asyncio
import sys
import os

# Ensure Python can find the 'src' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.connection.ws_client import HyperLiquidWS
from src.config import config

async def strategy_loop(data_queue):
    """
    This loop processes market data as it arrives.
    """
    print(f"Starting Strategy Loop for {config.COIN}...")
    
    while True:
        # Wait for new data from WebSocket
        market_data = await data_queue.get()
        data = market_data.get('data', {})
        
        if 'levels' in data:
            # Extract Best Bid/Ask
            # Hyperliquid L2 data structure: data['levels'][0] is bids, [1] is asks
            bids = data['levels'][0]
            asks = data['levels'][1]
            
            if bids and asks:
                best_bid = float(bids[0]['px'])
                best_ask = float(asks[0]['px'])
                mid_price = (best_bid + best_ask) / 2
                
                # --- YOUR LOGIC GOES HERE ---
                # 1. Update Historical Data List
                # 2. Calculate Fisher Transform (using src.utils.indicators)
                # 3. Run PyTorch Inference
                # 4. Execute Trade
                
                print(f"Price Update: {mid_price} | Processing Strategy...")

async def main():
    # Create a queue to share data between WS and Strategy
    data_queue = asyncio.Queue()
    
    # Initialize WebSocket Client
    ws_client = HyperLiquidWS(data_queue)
    
    # Run both the WS listener and the Strategy loop concurrently
    await asyncio.gather(
        ws_client.connect(),
        strategy_loop(data_queue)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user.")
