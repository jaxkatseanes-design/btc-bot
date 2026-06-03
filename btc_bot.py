"""
BTC Day Trading Bot — Alpaca Crypto
=====================================
Strategy : RSI + MACD + candle pattern + volume confluence
Timeframe : 1-minute candles (5-min for trend filter)
Exchange  : Alpaca paper trading (crypto)
Deploy    : Render free tier + cron-job.org keep-alive ping

Environment variables required (set in Render dashboard):
  ALPACA_API_KEY
  ALPACA_SECRET_KEY
  ALPACA_BASE_URL   (https://paper-api.alpaca.markets)
"""

import os
import time
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ─────────────────────────────────────────────
# CONFIG — edit these before deploying
# ─────────────────────────────────────────────
SYMBOL          = "BTC/USD"
ACCOUNT_BALANCE = 10000.0       # Starting paper balance — update periodically
RISK_PER_TRADE  = 0.01          # 1% of account per trade
MAX_POSITIONS   = 3             # Max concurrent open trades
DAILY_DD_LIMIT  = -0.03         # Stop trading if down 3% on the day
MIN_SCORE       = 5             # Minimum confluence score to enter (out of 8)
FULL_SIZE_SCORE = 5             # Score needed for full position (else half size)
ATR_PERIOD      = 14
RSI_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
VOLUME_AVG_PERIOD = 20  # volume disabled in scoring
STOP_ATR_MULT   = 1.5           # Stop = entry - (1.5 × ATR)
TARGET_ATR_MULT = 3.0           # Target = entry + (3.0 × ATR)
TRAIL_ACTIVATE  = 0.01          # Activate trailing stop at +1% profit
TRAIL_ATR_MULT  = 1.0           # Trail at 1 ATR below highest close
MAX_HOLD_HOURS  = 3             # Force close after 3 hours
WHALE_WICK_RATIO = 3.0          # Wick > 3× body triggers whale filter
WHALE_COOLDOWN  = 5             # Minutes to skip after whale signal
FOMO_BLOCK_PCT  = 1.5           # Skip if already moved 1.5% in last 5 candles
ATR_SPIKE_MULT  = 3.0           # Skip if current ATR > 3× its average

# Trade log file — Render ephemeral disk, resets on deploy
# For persistence use Render disk add-on or external DB later
TRADE_LOG = "trade_log.json"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ALPACA CLIENTS
# ─────────────────────────────────────────────
API_KEY    = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

trading = TradingClient(API_KEY, SECRET_KEY, paper=True)
data    = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
def get_bars(symbol: str, timeframe: TimeFrame, limit: int) -> list[dict]:
    """Fetch recent bars. Returns list of dicts with o/h/l/c/v keys."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(minutes=limit * 2)   # overshoot, then trim
    req = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        limit=limit
    )
    df = data.get_crypto_bars(req).df
    if df.empty:
        return []
    # Flatten multi-index if present
    if hasattr(df.index, 'levels'):
        df = df.loc[symbol] if symbol in df.index.get_level_values(0) else df
    bars = []
    for _, row in df.tail(limit).iterrows():
        bars.append({
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
            "v": float(row["volume"]),
        })
    return bars


# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────
def calc_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[-period + i - 1] - closes[-period + i - 2] if i > 1 else closes[-period] - closes[-period - 1]
        (gains if diff >= 0 else losses).append(abs(diff))
    # Use only the last `period` changes
    changes = [closes[i] - closes[i-1] for i in range(len(closes)-period, len(closes))]
    g = sum(x for x in changes if x > 0) / period
    l = sum(abs(x) for x in changes if x < 0) / period
    if l == 0:
        return 100.0
    rs = g / l
    return 100 - (100 / (1 + rs))


def calc_ema(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return [values[-1]] * len(values)
    k = 2 / (period + 1)
    ema = [sum(values[:period]) / period]
    for v in values[period:]:
        ema.append(v * k + ema[-1] * (1 - k))
    # Pad front
    pad = len(values) - len(ema)
    return [ema[0]] * pad + ema


def calc_macd(closes: list[float]):
    """Returns (macd_line, signal_line, histogram) for latest bar."""
    if len(closes) < MACD_SLOW + MACD_SIGNAL:
        return 0.0, 0.0, 0.0
    fast   = calc_ema(closes, MACD_FAST)
    slow   = calc_ema(closes, MACD_SLOW)
    macd   = [f - s for f, s in zip(fast, slow)]
    signal = calc_ema(macd, MACD_SIGNAL)
    hist   = [m - s for m, s in zip(macd, signal)]
    return macd[-1], signal[-1], hist[-1]


def calc_atr(bars: list[dict], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["h"], bars[i]["l"], bars[i-1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period


def calc_volume_avg(bars: list[dict], period: int = 20) -> float:
    vols = [b["v"] for b in bars[-period:]]
    return sum(vols) / len(vols) if vols else 0.0


# ─────────────────────────────────────────────
# CANDLE PATTERN DETECTION
# ─────────────────────────────────────────────
def candle_body(b: dict) -> float:
    return abs(b["c"] - b["o"])

def candle_range(b: dict) -> float:
    return b["h"] - b["l"] if b["h"] != b["l"] else 0.0001

def lower_wick(b: dict) -> float:
    return min(b["o"], b["c"]) - b["l"]

def upper_wick(b: dict) -> float:
    return b["h"] - max(b["o"], b["c"])

def is_bull(b: dict) -> bool:
    return b["c"] >= b["o"]

def is_hammer(b: dict) -> bool:
    body = candle_body(b)
    lw   = lower_wick(b)
    uw   = upper_wick(b)
    return body > 0 and lw >= 2 * body and uw <= 0.5 * body

def is_bullish_engulfing(bars: list[dict]) -> bool:
    if len(bars) < 2:
        return False
    prev, curr = bars[-2], bars[-1]
    return (
        not is_bull(prev) and is_bull(curr)
        and curr["o"] < prev["c"]
        and curr["c"] > prev["o"]
        and candle_body(curr) > candle_body(prev)
    )

def is_three_green(bars: list[dict]) -> bool:
    if len(bars) < 3:
        return False
    c1, c2, c3 = bars[-3], bars[-2], bars[-1]
    return (
        is_bull(c1) and is_bull(c2) and is_bull(c3)
        and c2["c"] > c1["c"]
        and c3["c"] > c2["c"]
        and c2["o"] >= c1["o"]
        and c3["o"] >= c2["o"]
    )

def is_morning_star(bars: list[dict]) -> bool:
    if len(bars) < 3:
        return False
    c1, c2, c3 = bars[-3], bars[-2], bars[-1]
    mid_body = candle_body(c2)
    return (
        not is_bull(c1) and candle_body(c1) > 0
        and mid_body < candle_body(c1) * 0.4
        and is_bull(c3)
        and c3["c"] > (c1["o"] + c1["c"]) / 2
    )

def is_shooting_star(b: dict) -> bool:
    body = candle_body(b)
    uw   = upper_wick(b)
    lw   = lower_wick(b)
    return body > 0 and uw >= 2 * body and lw <= 0.5 * body

def is_bearish_engulfing(bars: list[dict]) -> bool:
    if len(bars) < 2:
        return False
    prev, curr = bars[-2], bars[-1]
    return (
        is_bull(prev) and not is_bull(curr)
        and curr["o"] > prev["c"]
        and curr["c"] < prev["o"]
        and candle_body(curr) > candle_body(prev)
    )

def detect_bullish_pattern(bars: list[dict]) -> str | None:
    if is_three_green(bars):       return "three_green"
    if is_bullish_engulfing(bars): return "engulfing"
    if is_morning_star(bars):      return "morning_star"
    if is_hammer(bars[-1]):        return "hammer"
    return None

def detect_bearish_pattern(bars: list[dict]) -> str | None:
    if is_bearish_engulfing(bars): return "bear_engulf"
    if is_shooting_star(bars[-1]): return "shooting_star"
    return None


# ─────────────────────────────────────────────
# WHALE / MANIPULATION FILTER
# ─────────────────────────────────────────────
_whale_cooldown_until: datetime | None = None

def whale_detected(bars: list[dict], atr: float, avg_vol: float) -> str | None:
    """Returns reason string if manipulation detected, else None."""
    global _whale_cooldown_until

    # Active cooldown from prior detection
    if _whale_cooldown_until and datetime.now(timezone.utc) < _whale_cooldown_until:
        mins = (_whale_cooldown_until - datetime.now(timezone.utc)).seconds // 60
        return f"whale_cooldown ({mins}m remaining)"

    b = bars[-1]

    # 1 — Stop hunt: massive lower wick on high volume, price recovered
    body = candle_body(b)
    lw   = lower_wick(b)
    if body > 0 and lw > WHALE_WICK_RATIO * body and b["v"] > avg_vol * 2:
        _whale_cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=WHALE_COOLDOWN)
        return "stop_hunt_wick"

    # 2 — Volume spike with tiny body (distribution / fake move)
    if b["v"] > avg_vol * 3 and candle_body(b) < b["c"] * 0.003:
        _whale_cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=WHALE_COOLDOWN)
        return "volume_without_commitment"

    # 3 — ATR spike (news / cascade event)
    recent_atr = calc_atr(bars[-30:], 14) if len(bars) >= 30 else atr
    if len(bars) >= 60:
        older_atr = calc_atr(bars[-60:-30], 14)
        if older_atr > 0 and recent_atr > older_atr * ATR_SPIKE_MULT:
            return "atr_spike_news_event"

    # 4 — FOMO block: already moved 1.5%+ in last 5 candles
    if len(bars) >= 5:
        move_pct = abs(bars[-1]["c"] - bars[-5]["c"]) / bars[-5]["c"] * 100
        if move_pct > FOMO_BLOCK_PCT:
            return f"fomo_block ({move_pct:.1f}% in 5 candles)"

    return None


# ─────────────────────────────────────────────
# CONFLUENCE SCORING
# ─────────────────────────────────────────────
def score_setup(bars_1m: list[dict], bars_5m: list[dict]) -> dict:
    """
    Score the current setup 0–8. Returns dict with score and breakdown.
    Positive signals max out at 8. Negative signals reduce score.
    """
    closes_1m  = [b["c"] for b in bars_1m]
    closes_5m  = [b["c"] for b in bars_5m]
    score      = 0
    breakdown  = {}

    # ── RSI (1-min) ──────────────────────────
    rsi = calc_rsi(closes_1m, RSI_PERIOD)
    breakdown["rsi"] = round(rsi, 1)
    if rsi < 30:
        score += 2; breakdown["rsi_signal"] = "oversold +2"
    elif rsi < 45:
        score += 1; breakdown["rsi_signal"] = "recovering +1"
    elif rsi > 70:
        score -= 1; breakdown["rsi_signal"] = "overbought -1"
    else:
        breakdown["rsi_signal"] = "neutral 0"

    # ── RSI divergence (basic: RSI rising while price flat/down) ──
    if len(closes_1m) >= 10:
        price_move = closes_1m[-1] - closes_1m[-10]
        rsi_now    = calc_rsi(closes_1m, RSI_PERIOD)
        rsi_prev   = calc_rsi(closes_1m[:-5], RSI_PERIOD)
        if price_move < 0 and rsi_now > rsi_prev:
            score += 2; breakdown["rsi_div"] = "bullish_divergence +2"
        elif price_move > 0 and rsi_now < rsi_prev:
            score -= 2; breakdown["rsi_div"] = "bearish_divergence -2"

    # ── MACD (1-min) ─────────────────────────
    macd, sig, hist = calc_macd(closes_1m)
    breakdown["macd"] = round(macd, 2)
    breakdown["macd_signal"] = round(sig, 2)
    breakdown["macd_hist"] = round(hist, 2)

    # Check crossover: macd crossed above signal in last 3 bars
    if len(closes_1m) >= MACD_SLOW + MACD_SIGNAL + 3:
        prev_macd, prev_sig, _ = calc_macd(closes_1m[:-1])
        if macd > sig and prev_macd <= prev_sig:
            score += 2; breakdown["macd_signal_str"] = "bullish_crossover +2"
        elif macd > sig and hist > 0:
            score += 1; breakdown["macd_signal_str"] = "above_signal +1"
        elif macd < sig and prev_macd >= prev_sig:
            score -= 2; breakdown["macd_signal_str"] = "bearish_crossover -2"
        else:
            breakdown["macd_signal_str"] = "neutral 0"

    # ── Candle pattern (1-min) ────────────────
    bull_pat = detect_bullish_pattern(bars_1m)
    bear_pat = detect_bearish_pattern(bars_1m)
    if bull_pat:
        score += 1; breakdown["pattern"] = f"{bull_pat} +1"
    elif bear_pat:
        score -= 2; breakdown["pattern"] = f"{bear_pat} -2"
    else:
        breakdown["pattern"] = "none 0"

    # ── Volume ───────────────────────────────
    avg_vol = calc_volume_avg(bars_1m, VOLUME_AVG_PERIOD)
    cur_vol = bars_1m[-1]["v"]
    vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
    breakdown["volume_ratio"] = round(vol_ratio, 2)
    if vol_ratio >= 2.0 and is_bull(bars_1m[-1]):
        score += 2; breakdown["vol_signal"] = "high_vol_green +2"
    elif vol_ratio >= 1.5 and is_bull(bars_1m[-1]):
        score += 1; breakdown["vol_signal"] = "above_avg_green +1"
    elif vol_ratio >= 2.0 and not is_bull(bars_1m[-1]):
        score -= 2; breakdown["vol_signal"] = "high_vol_red -2"
    else:
        breakdown["vol_signal"] = "normal 0"

    # ── 5-min trend filter ───────────────────
    if len(bars_5m) >= 20:
        ema20_5m = calc_ema(closes_5m, 20)
        if closes_5m[-1] > ema20_5m[-1]:
            score += 1; breakdown["trend_5m"] = "above_20ema +1"
        else:
            score -= 1; breakdown["trend_5m"] = "below_20ema -1"

    # Cap score at 8, floor at 0
    score = max(0, min(8, score))
    breakdown["total_score"] = score
    return breakdown


# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────
def calc_position_size(
    entry: float,
    stop: float,
    account_balance: float,
    score: int
) -> float:
    """
    Returns BTC quantity to buy based on 1% risk rule.
    Uses half size if score < FULL_SIZE_SCORE.
    """
    risk_pct   = RISK_PER_TRADE * (1.0 if score >= FULL_SIZE_SCORE else 0.5)
    max_risk   = account_balance * risk_pct
    risk_per_unit = entry - stop
    if risk_per_unit <= 0:
        return 0.0
    qty = max_risk / risk_per_unit
    return round(qty, 4)


# ─────────────────────────────────────────────
# TRADE LOG
# ─────────────────────────────────────────────
def load_log() -> dict:
    if not os.path.exists(TRADE_LOG):
        return {"trades": [], "daily_pnl": 0.0, "last_reset": str(datetime.now(timezone.utc).date())}
    with open(TRADE_LOG) as f:
        return json.load(f)

def save_log(log_data: dict):
    with open(TRADE_LOG, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

def reset_daily_pnl_if_new_day(log_data: dict) -> dict:
    today = str(datetime.now(timezone.utc).date())
    if log_data.get("last_reset") != today:
        log_data["daily_pnl"] = 0.0
        log_data["last_reset"] = today
        log.info("New trading day — daily P&L reset")
    return log_data

def log_trade(log_data: dict, trade: dict):
    log_data["trades"].append(trade)
    save_log(log_data)


# ─────────────────────────────────────────────
# ORDER MANAGEMENT
# ─────────────────────────────────────────────
def get_open_positions() -> list:
    try:
        positions = trading.get_all_positions()
        return [p for p in positions if p.symbol == SYMBOL.replace("/", "")]
    except Exception as e:
        log.error(f"get_open_positions error: {e}")
        return []

def get_account_balance() -> float:
    try:
        account = trading.get_account()
        return float(account.cash)
    except Exception as e:
        log.error(f"get_account_balance error: {e}")
        return ACCOUNT_BALANCE

def place_buy(qty: float, entry: float, stop: float, target: float) -> bool:
    """Place market buy. Stop and target tracked manually via log."""
    try:
        req = MarketOrderRequest(
            symbol=SYMBOL.replace("/", ""),
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )
        order = trading.submit_order(req)
        log.info(f"BUY placed | qty={qty} | stop={stop:.2f} | target={target:.2f} | id={order.id}")
        return True
    except Exception as e:
        log.error(f"place_buy error: {e}")
        return False

def place_sell(qty: float, reason: str = "exit") -> bool:
    """Market sell to close position."""
    try:
        req = MarketOrderRequest(
            symbol=SYMBOL.replace("/", ""),
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        )
        order = trading.submit_order(req)
        log.info(f"SELL placed | qty={qty} | reason={reason} | id={order.id}")
        return True
    except Exception as e:
        log.error(f"place_sell error: {e}")
        return False


# ─────────────────────────────────────────────
# POSITION MANAGEMENT (stop / target / trail)
# ─────────────────────────────────────────────
def manage_open_positions(bars: list[dict], log_data: dict, atr: float):
    """
    Check each open position against stop loss, take profit,
    trailing stop, and time-based exit. Uses trade log for entry data.
    """
    positions = get_open_positions()
    if not positions:
        return

    current_price = bars[-1]["c"]
    highest_close  = max(b["c"] for b in bars[-180:])  # 3-hour window

    for pos in positions:
        qty    = abs(float(pos.qty))
        cost   = float(pos.avg_entry_price)
        pnl_pct = (current_price - cost) / cost * 100

        # Find matching trade in log for stop/target
        trade_rec = next(
            (t for t in reversed(log_data["trades"])
             if t.get("status") == "open" and t.get("symbol") == SYMBOL),
            None
        )

        stop_price   = trade_rec["stop"] if trade_rec else cost * (1 - STOP_ATR_MULT * atr / cost)
        target_price = trade_rec["target"] if trade_rec else cost * (1 + TARGET_ATR_MULT * atr / cost)
        entry_time   = datetime.fromisoformat(trade_rec["entry_time"]) if trade_rec else datetime.now(timezone.utc)

        # ── Trailing stop: activate at +1% ──
        if pnl_pct >= TRAIL_ACTIVATE * 100:
            trail_stop = highest_close - (TRAIL_ATR_MULT * atr)
            if trail_stop > stop_price:
                stop_price = trail_stop
                log.info(f"Trail stop updated → {stop_price:.2f}")
                if trade_rec:
                    trade_rec["stop"] = stop_price
                    save_log(log_data)

        # ── Partial exit at target (sell 50%) ──
        if current_price >= target_price and trade_rec and not trade_rec.get("partial_exit"):
            half_qty = round(qty / 2, 4)
            if place_sell(half_qty, reason="target_50pct"):
                log.info(f"Partial exit 50% at target {target_price:.2f}")
                if trade_rec:
                    trade_rec["partial_exit"] = True
                    trade_rec["stop"] = cost  # Move stop to breakeven after partial
                    save_log(log_data)

        # ── Stop loss hit ──
        elif current_price <= stop_price:
            if place_sell(qty, reason="stop_loss"):
                pnl = (current_price - cost) * qty
                log_data["daily_pnl"] = log_data.get("daily_pnl", 0) + pnl
                log.info(f"Stop hit | P&L: ${pnl:.2f} | daily_pnl: ${log_data['daily_pnl']:.2f}")
                if trade_rec:
                    trade_rec["status"] = "closed"
                    trade_rec["exit_price"] = current_price
                    trade_rec["pnl"] = round(pnl, 2)
                save_log(log_data)

        # ── Bearish reversal → early exit ──
        elif detect_bearish_pattern(bars[-3:]):
            pat = detect_bearish_pattern(bars[-3:])
            if place_sell(qty, reason=f"bearish_pattern_{pat}"):
                pnl = (current_price - cost) * qty
                log_data["daily_pnl"] = log_data.get("daily_pnl", 0) + pnl
                log.info(f"Early exit on {pat} | P&L: ${pnl:.2f}")
                if trade_rec:
                    trade_rec["status"] = "closed"
                    trade_rec["exit_price"] = current_price
                    trade_rec["pnl"] = round(pnl, 2)
                save_log(log_data)

        # ── Time-based exit (3 hours) ──
        elif (datetime.now(timezone.utc) - entry_time).total_seconds() > MAX_HOLD_HOURS * 3600:
            if place_sell(qty, reason="time_exit_3hr"):
                pnl = (current_price - cost) * qty
                log_data["daily_pnl"] = log_data.get("daily_pnl", 0) + pnl
                log.info(f"Time exit 3hr | P&L: ${pnl:.2f}")
                if trade_rec:
                    trade_rec["status"] = "closed"
                    trade_rec["exit_price"] = current_price
                    trade_rec["pnl"] = round(pnl, 2)
                save_log(log_data)

        else:
            log.info(
                f"Position OK | price={current_price:.2f} | cost={cost:.2f} | "
                f"pnl={pnl_pct:.2f}% | stop={stop_price:.2f} | target={target_price:.2f}"
            )


# ─────────────────────────────────────────────
# KEEP-ALIVE ENDPOINT (for cron-job.org ping)
# ─────────────────────────────────────────────
def start_keepalive_server():
    """
    Minimal HTTP server so cron-job.org can ping Render
    and keep the free instance awake. Runs in background thread.
    """
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"btc_bot alive")
        def log_message(self, *args):
            pass  # Suppress default HTTP logs

    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info(f"Keep-alive server running on port {port}")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def run():
    log.info("=" * 60)
    log.info("BTC Bot starting up")
    log.info("=" * 60)

    start_keepalive_server()

    # State
    log_data = load_log()

    while True:
        try:
            cycle_start = time.time()
            log_data = reset_daily_pnl_if_new_day(log_data)

            # ── Circuit breaker: daily drawdown limit ──
            if log_data.get("daily_pnl", 0) <= ACCOUNT_BALANCE * DAILY_DD_LIMIT:
                log.warning(
                    f"Daily drawdown limit hit: ${log_data['daily_pnl']:.2f}. "
                    "Sleeping until midnight UTC."
                )
                now = datetime.now(timezone.utc)
                midnight = (now + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                time.sleep((midnight - now).total_seconds())
                continue

            # ── Fetch data ──────────────────────────
            bars_1m = get_bars(SYMBOL, TimeFrame.Minute, 100)
            bars_5m = get_bars(SYMBOL, TimeFrame(5, TimeFrameUnit.Minute), 50)

            if len(bars_1m) < 50:
                log.warning("Not enough 1-min bars yet — waiting")
                time.sleep(30)
                continue

            atr     = calc_atr(bars_1m, ATR_PERIOD)
            avg_vol = calc_volume_avg(bars_1m, VOLUME_AVG_PERIOD)
            current_price = bars_1m[-1]["c"]

            log.info(f"BTC={current_price:.2f} | ATR={atr:.2f} | vol_ratio={bars_1m[-1]['v']/avg_vol:.2f}x")

            # ── Manage existing positions ────────────
            manage_open_positions(bars_1m, log_data, atr)

            # ── Check position count ─────────────────
            open_positions = get_open_positions()
            if len(open_positions) >= MAX_POSITIONS:
                log.info(f"Max positions ({MAX_POSITIONS}) reached — no new entries")
                time.sleep(60)
                continue

            # ── Whale / manipulation filter ──────────
            whale = whale_detected(bars_1m, atr, avg_vol)
            if whale:
                log.info(f"Whale filter triggered: {whale} — skipping candle")
                time.sleep(60)
                continue

            # ── Confluence scoring ───────────────────
            result   = score_setup(bars_1m, bars_5m)
            score    = result["total_score"]
            log.info(f"Confluence score: {score}/8 | {result}")

            if score < MIN_SCORE:
                log.info(f"Score {score} < {MIN_SCORE} minimum — no trade")
                time.sleep(60)
                continue

            # ── Calculate stop and target ────────────
            entry  = current_price
            stop   = round(entry - STOP_ATR_MULT * atr, 2)
            target = round(entry + TARGET_ATR_MULT * atr, 2)
            rr     = (target - entry) / (entry - stop) if entry > stop else 0

            if rr < 1.5:
                log.info(f"R/R too low ({rr:.2f}) — skipping trade")
                time.sleep(60)
                continue

            # ── Position size ────────────────────────
            balance = get_account_balance()
            qty     = calc_position_size(entry, stop, balance, score)

            if qty <= 0:
                log.warning("Position size calculated as 0 — skipping")
                time.sleep(60)
                continue

            log.info(
                f"ENTRY SIGNAL | score={score}/8 | entry={entry:.2f} | "
                f"stop={stop:.2f} | target={target:.2f} | rr={rr:.2f} | qty={qty}"
            )

            # ── Place order ──────────────────────────
            success = place_buy(qty, entry, stop, target)

            if success:
                trade_record = {
                    "symbol":     SYMBOL,
                    "entry_time": str(datetime.now(timezone.utc)),
                    "entry":      entry,
                    "stop":       stop,
                    "target":     target,
                    "qty":        qty,
                    "score":      score,
                    "signals":    result,
                    "status":     "open",
                    "partial_exit": False,
                    "exit_price": None,
                    "pnl":        None,
                }
                log_trade(log_data, trade_record)

            # ── Sleep until next minute ──────────────
            elapsed = time.time() - cycle_start
            sleep_time = max(60 - elapsed, 5)
            log.info(f"Cycle complete in {elapsed:.1f}s — sleeping {sleep_time:.0f}s")
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            log.info("Bot stopped by user")
            break
        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)
            time.sleep(30)


if __name__ == "__main__":
    run()
