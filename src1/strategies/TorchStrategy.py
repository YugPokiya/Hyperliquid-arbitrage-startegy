import warnings
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
console = Console()

# ==========================================
# 1. PYTORCH STRATEGY ENGINE
# ==========================================
class TorchStrategy:
    """
    Implements the Pine Script logic using PyTorch Tensors for high-performance
    vectorized calculation.
    """
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def _to_tensor(self, series):
        """Helper to convert pandas series to torch tensor."""
        # Fill NaNs with 0 or previous values to avoid torch errors
        clean_series = series.fillna(0).values
        return torch.tensor(clean_series, dtype=torch.float32, device=self.device)

    def calculate_fisher(self, high, low, length=9):
        """
        PyTorch implementation of Fisher Transform.
        Pine Script logic:
        value1 = 0.66 * ((hl2 - min) / (max - min) - 0.5) + 0.67 * value1[1]
        fisher = 0.5 * log((1 + value1) / (1 - value1)) + 0.5 * fisher[1]
        """
        # Note: Recursive calculations (EMA-like) are hard to fully vectorize in pure Torch 
        # without a loop or compiled kernel. For hybrid speed, we use a loop for the recursive 
        # parts but keep data on device.
        
        n = len(high)
        hl2 = (high + low) / 2
        
        # Rolling min/max (efficiently done via unfolding or pandas first)
        # We will use the passed pandas series for the rolling window to keep it simple, 
        # then switch to torch for the transform.
        high_roll = high.rolling(length).max().fillna(high)
        low_roll = low.rolling(length).min().fillna(low)
        
        # Convert to tensors for the math
        t_hl2 = self._to_tensor(hl2)
        t_high = self._to_tensor(high_roll)
        t_low = self._to_tensor(low_roll)
        
        # Initialize output tensors
        value1 = torch.zeros_like(t_hl2)
        fisher = torch.zeros_like(t_hl2)
        
        # We must loop for the recursive component (Pine Script: nz(value1[1]))
        # Using TorchScript (JIT) here would be the next level of optimization.
        eps = 1e-9
        
        for i in range(1, n):
            # Normalize
            denom = t_high[i] - t_low[i]
            if denom == 0: denom = eps
            
            raw_val = 0.66 * ((t_hl2[i] - t_low[i]) / denom - 0.5) + 0.67 * value1[i-1]
            
            # Clamp to avoid log(negative) or log(0) errors (Pine Script limit is 0.999)
            raw_val = torch.clamp(raw_val, -0.999, 0.999)
            value1[i] = raw_val
            
            fisher[i] = 0.5 * torch.log((1 + raw_val) / (1 - raw_val)) + 0.5 * fisher[i-1]
            
        return fisher

    def run_strategy(self, df):
        """
        Main execution block.
        """
        # --- A. Pre-calculate Indicators using Pandas-TA (Fast C-wrappers) ---
        # We mix Pandas-TA for standard indicators and PyTorch for custom logic
        df.ta.ema(length=5, append=True)
        df.ta.ema(length=8, append=True)
        df.ta.ema(length=13, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.psar(append=True) # Adds PSARl_0.02_0.2 and PSARs...
        df.ta.ema(length=200, append=True)

        # Drop NaNs created by indicators
        df.dropna(inplace=True)
        if len(df) < 200: return None

        # --- B. Move Data to PyTorch Tensors ---
        # Close, High, Low
        close = self._to_tensor(df['Close'])
        high = self._to_tensor(df['High'])
        low = self._to_tensor(df['Low'])
        
        # Indicators
        ema5 = self._to_tensor(df['EMA_5'])
        ema8 = self._to_tensor(df['EMA_8'])
        ema13 = self._to_tensor(df['EMA_13'])
        ema200 = self._to_tensor(df['EMA_200'])
        rsi = self._to_tensor(df['RSI_14'])
        
        # PSAR (Pandas TA creates combined columns or separate long/short columns)
        # We need a single PSAR column. 
        psar_cols = [c for c in df.columns if 'PSAR' in c]
        # Combine PSAR columns usually 'PSARl' and 'PSARs'
        psar_vals = df[psar_cols[0]].fillna(0) + df[psar_cols[1]].fillna(0)
        psar = self._to_tensor(psar_vals)

        # --- C. Custom PyTorch Fisher Calculation ---
        fisher = self.calculate_fisher(df['High'], df['Low'])
        fisher_trigger = torch.roll(fisher, 1) # trigger = fisher[1]

        # --- D. Vectorized Boolean Logic (The "Brain") ---
        
        # 1. 5-8-13 Logic
        # --------------------------------
        # Crossovers (Using shifting)
        ema5_prev = torch.roll(ema5, 1)
        ema8_prev = torch.roll(ema8, 1)
        ema13_prev = torch.roll(ema13, 1)
        
        cross_5_8 = ((ema5 > ema8) & (ema5_prev <= ema8_prev)) | (ema5 > ema8)
        cross_5_13 = ((ema5 > ema13) & (ema5_prev <= ema13_prev)) | (ema5 > ema13)
        
        # PSAR Direction (1 if PSAR < Close else -1)
        psar_dir = torch.where(psar < close, 1.0, -1.0)
        
        # "Strong Buy" Condition (from your Pine Script)
        # long5813 condition
        bullish_alignment = (close > ema5) & (close > ema8) & (close > ema13)
        price_increasing = close > torch.roll(close, 1)
        
        strong_buy_signal = (
            cross_5_8 & 
            cross_5_13 & 
            (psar_dir == 1.0) & 
            price_increasing & 
            bullish_alignment
        )

        # 2. Fisher Anomaly Logic
        # --------------------------------
        anomaly_threshold = 2.0
        is_oversold_anomaly = fisher < -anomaly_threshold
        is_overbought_anomaly = fisher > anomaly_threshold
        
        bullish_trend = close > ema200
        
        # Fisher Entry: Crossover(fisher, trigger) AND previous was anomaly
        fisher_cross_up = (fisher > fisher_trigger) & (torch.roll(fisher, 1) <= fisher_trigger)
        fisher_entry_long = fisher_cross_up & torch.roll(is_oversold_anomaly, 1) & bullish_trend

        # --- E. Compile Results ---
        # We extract the indices where signals are True
        # Using the last candle for "Live" status
        last_idx = -1
        
        result = {
            "Fisher_Val": fisher[last_idx].item(),
            "RSI": rsi[last_idx].item(),
            "Trend_200": "Bullish" if bullish_trend[last_idx] else "Bearish",
            "Signal_5813": "BUY" if strong_buy_signal[last_idx] else "NEUTRAL",
            "Signal_Fisher": "ENTRY LONG" if fisher_entry_long[last_idx] else "NEUTRAL"
        }
        
        return result

# ==========================================
# 2. MULTIPROCESSING WORKER
# ==========================================
def process_ticker(ticker):
    """
    Worker function to fetch data and run strategy for a single ticker.
    """
    try:
        # Fetch Data (New tool: yfinance with threads disabled to be safe in multiprocessing)
        df = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
        
        if df.empty:
            return None

        # Initialize Strategy Engine
        # Note: In multiprocessing, we initialize class inside to avoid pickling issues with CUDA tensors
        engine = TorchStrategy() 
        results = engine.run_strategy(df)
        
        if results:
            results['Ticker'] = ticker
            return results
        return None
        
    except Exception as e:
        return None

# ==========================================
# 3. MAIN ORCHESTRATOR
# ==========================================
def main():
    # List of tickers (Mix of Tech, Crypto, Indices for demo)
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "INTC", 
        "BTC-USD", "ETH-USD", "SOL-USD", "SPY", "QQQ", "IWM"
    ]
    
    console.print("[bold cyan]🚀 Initializing AI-Powered Strategy Engine...[/bold cyan]")
    console.print(f"[yellow]Device detected: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}[/yellow]")
    
    results_list = []
    
    # Modern Progress Bar context
    with Progress() as progress:
        task = progress.add_task("[green]Scanning Market...[/green]", total=len(tickers))
        
        # Multiprocessing Pool
        # We leverage all CPU cores to fetch and process independent tickers
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_ticker, t): t for t in tickers}
            
            for future in futures:
                res = future.result()
                if res:
                    results_list.append(res)
                progress.advance(task)

    # Display Results in a Rich Table
    table = Table(title="Strategy Analysis Results (Latest Candle)")

    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Fisher Value", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("200 Trend", justify="center")
    table.add_column("5-8-13 Signal", justify="center", style="bold")
    table.add_column("Fisher Signal", justify="center", style="bold")

    for row in results_list:
        # Color coding the output
        fisher_style = "red" if row['Fisher_Val'] > 2 else ("green" if row['Fisher_Val'] < -2 else "white")
        trend_style = "green" if row['Trend_200'] == "Bullish" else "red"
        
        sig_5813 = row['Signal_5813']
        sig_5813_style = "green reverse" if sig_5813 == "BUY" else "dim"
        
        sig_fish = row['Signal_Fisher']
        sig_fish_style = "green reverse" if "ENTRY" in sig_fish else "dim"

        table.add_row(
            row['Ticker'],
            f"[{fisher_style}]{row['Fisher_Val']:.2f}[/{fisher_style}]",
            f"{row['RSI']:.2f}",
            f"[{trend_style}]{row['Trend_200']}[/{trend_style}]",
            f"[{sig_5813_style}]{sig_5813}[/{sig_5813_style}]",
            f"[{sig_fish_style}]{sig_fish}[/{sig_fish_style}]"
        )

    console.print(table)
    console.print("[bold green]Scan Complete.[/bold green]")

if __name__ == "__main__":
    # Required for Windows Multiprocessing support
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
