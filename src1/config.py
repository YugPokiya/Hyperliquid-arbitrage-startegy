import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    # API Keys (Loaded from environment variables)
    PRIVATE_KEY = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")
    
    # Trading Settings
    COIN = "ETH"  # Change to "HYPE" or "BTC" as needed
    LEVERAGE = 5
    TIMEFRAME = "1m"
    
    # ML Model Settings
    MODEL_PATH = "model_checkpoints/gru_fisher_v1.pth"
    FISHER_LEN_1 = 9
    FISHER_LEN_2 = 21

    # URLs
    WS_URL = "wss://api.hyperliquid.xyz/ws"
    API_URL = "https://api.hyperliquid.xyz"

config = Config()
