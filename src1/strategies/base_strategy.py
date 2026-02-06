from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    def __init__(self, rest_client):
        """
        :param rest_client: Instance of HyperLiquidRestClient to execute trades.
        """
        self.rest_client = rest_client
        self.is_active = True

    @abstractmethod
    async def process_data(self, market_data):
        """
        Abstract method to process incoming market data.
        Must be implemented by the child class.
        """
        pass

    def stop(self):
        """Stops the strategy execution."""
        self.is_active = False
        logger.info("Strategy stopped.")
