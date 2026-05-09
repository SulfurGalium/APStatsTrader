import os

SYMBOL = "SPY"          # S&P 500 ETF
TICKERS = "Q.SPY"

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

FRED_API_KEY = os.getenv("FRED_API_KEY")


HISTORICAL_CSV = "ES_5Years_8_11_2024.csv"
SIMULATION_CSV = "sp500_with_yield_5m.csv"
MACRO_CACHE_CSV = "macro_cache.csv"
FRED_YIELD_SERIES = "DGS10"

MODEL_PREFIX = "diffusion_macro_v2"
MODEL_PATH = f"{MODEL_PREFIX}.pth"
SCALER_PATH = f"{MODEL_PREFIX}_scaler.pkl"

CONTEXT_SIZE = 60
HORIZON = 12
BATCH_SIZE = 64
EPOCHS = 80
DIFFUSION_STEPS = 30
BETA_SCHEDULE = "cosine"

DEFAULT_BRANCHES = 100
DEFAULT_EQUITY = 10_000.0
DEFAULT_RISK_PCT = 0.05
DEFAULT_EMA_SPAN = 20
DEFAULT_MOMENTUM_WEIGHT = 3
DEFAULT_NUM_SAMPLES = 64

# Risk controls for backtest & live
MIN_LOCAL_VOL = 0.5        # minimum $ volatility used for stop sizing
MAX_POSITION_NOTIONAL = 5_000.0  # cap per-trade notional exposure
CONFIDENCE_THRESHOLD = 0.05       # minimum abs(mu) in scaled space to trade
SIGNAL_TO_NOISE_THRESHOLD = 1.5   # minimum |forecast_mu| / forecast_sigma to trade
MIN_EDGE_TO_STOP_RATIO = 1.25     # required expected move divided by stop distance
REQUIRE_TREND_CONFIRMATION = True

# Live paper trading
LIVE_RISK_PCT = DEFAULT_RISK_PCT
LIVE_STOP_MULTIPLIER = 1.25   # fraction of recent volatility
LIVE_BAR_TIMEFRAME_MIN = 5
