import asyncio
import json
import logging
import websockets
from src.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperLiquidWS:
    def __init__(self, data_queue):
        """
        :param data_queue: An asyncio.Queue to push market data into.
        """
        self.url = config.WS_URL
        self.data_queue = data_queue
        self.is_running = False
        self.ws = None

    async def connect(self):
        """Main loop to maintain the connection."""
        self.is_running = True
        
        while self.is_running:
            try:
                logger.info(f"Connecting to Hyperliquid WS: {self.url}...")
                
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
                    self.ws = ws
                    logger.info("WebSocket Connected.")
                    
                    # 1. Subscribe to Order Book (L2)
                    await self._subscribe(ws, "l2Book", config.COIN)
                    
                    # 2. Subscribe to Trades (Optional, good for volume analysis)
                    await self._subscribe(ws, "trades", config.COIN)

                    # 3. Listen for messages
                    await self._listen(ws)
                    
            except Exception as e:
                logger.error(f"WS Connection Error: {e}")
                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def _subscribe(self, ws, channel, coin):
        """Helper to send subscription JSON."""
        msg = {
            "method": "subscribe",
            "subscription": {
                "type": channel,
                "coin": coin
            }
        }
        await ws.send(json.dumps(msg))
        logger.info(f"Subscribed to {channel} for {coin}")

    async def _listen(self, ws):
        """Inner loop to process messages."""
        async for message in ws:
            try:
                data = json.loads(message)
                channel = data.get("channel")

                # Filter specifically for Order Book updates
                if channel == "l2Book":
                    # Determine if it's a snapshot or an update
                    # Hyperliquid sends 'isSnapshot': true on first connect
                    await self.data_queue.put({
                        "type": "l2Book",
                        "data": data["data"]
                    })
                
                # Handle Trade updates if needed
                elif channel == "trades":
                    # You can process trades here if your strategy needs them
                    pass

                elif channel == "subscriptionResponse":
                    logger.info(f"Subscription Confirmed: {data['data']}")

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    def stop(self):
        """Stops the WebSocket loop."""
        self.is_running = False
        if self.ws:
            asyncio.create_task(self.ws.close())
