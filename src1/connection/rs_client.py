import logging
import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from src.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperLiquidRestClient:
    def __init__(self):
        """
        Initializes the Exchange and Info clients using the private key from config.
        """
        try:
            # 1. Setup Account (Wallet)
            self.account = eth_account.Account.from_key(config.PRIVATE_KEY)
            logger.info(f"Initialized REST Client for wallet: {self.account.address}")

            # 2. Initialize Info Client (for fetching prices/meta)
            self.info = Info(constants.MAINNET_API_URL, skip_ws=True)

            # 3. Initialize Exchange Client (for placing orders)
            # Note: We use the account object here for automatic signing
            self.exchange = Exchange(self.account, constants.MAINNET_API_URL, self.account)
            
        except Exception as e:
            logger.error(f"Failed to initialize REST client: {e}")
            raise e

    def get_account_value(self):
        """Fetches the current account margin summary."""
        try:
            user_state = self.info.user_state(self.account.address)
            return user_state["marginSummary"]["accountValue"]
        except Exception as e:
            logger.error(f"Error fetching account value: {e}")
            return None

    def place_order(self, coin: str, is_buy: bool, price: float, size: float):
        """
        Places a Limit order.
        """
        try:
            logger.info(f"Placing Order: {coin} | {'Buy' if is_buy else 'Sell'} | Price: {price} | Size: {size}")
            
            # The SDK handles the complex 'action' dictionary and signing for you
            order_result = self.exchange.order(
                name=coin,
                is_buy=is_buy,
                sz=size,
                limit_px=price,
                order_type={"limit": {"tif": "Gtc"}},  # Good Till Cancelled
                reduce_only=False
            )
            
            status = order_result["response"]["data"]["statuses"][0]
            if "resting" in status:
                oid = status["resting"]["oid"]
                logger.info(f"Order Placed Successfully. OID: {oid}")
                return oid
            elif "filled" in status:
                logger.info("Order Immediately Filled.")
                return "filled"
            else:
                logger.error(f"Order Failed: {status}")
                return None

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def cancel_all_orders(self):
        """Cancels all open orders for the account."""
        try:
            logger.info("Cancelling all open orders...")
            # Fetch open orders first
            open_orders = self.info.open_orders(self.account.address)
            
            if open_orders:
                result = self.exchange.cancel_all_orders()
                logger.info(f"Cancel Result: {result}")
            else:
                logger.info("No open orders to cancel.")
                
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

# Create a singleton instance to be imported elsewhere
rest_client = HyperLiquidRestClient()
