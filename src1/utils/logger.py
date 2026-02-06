import logging
from rich.logging import RichHandler

def setup_logger(name: str = "CryptoBot", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance with Rich formatting.
    
    Args:
        name (str): The name of the logger (usually __name__).
        level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if function is called repeatedly
    if not logger.handlers:
        # Create Rich handler
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False
        )
        
        # Set format (Rich handles the heavy lifting, but we can set a basic format)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
    return logger
