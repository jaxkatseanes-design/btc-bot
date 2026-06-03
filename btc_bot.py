"""
BTC Day Trading Bot — Support/Resistance Range Strategy
========================================================
Strategy : Find support from last 3 major lows, wait for price to
           touch support, confirm with 2 consecutive green candles,
           buy on 3rd candle, target midpoint, stop below lowest wick.
Timeframe : 1-minute candles
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
from datetime import datetime, timezone, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SYMBOL            = "BTC/USD"
RISK_PER_TRADE    = 0.01       # 1% of account per trade
MAX_POSITIONS     = 3          # Max concurrent open trades
DAILY_DD_LIMIT    = -0.03      # Stop trading if down 3% on the day
SUPPORT_LOOKBACK  = 100        # Candles to look back for support/resistance
SUPPORT_TOUCH_PCT = 0.003      # Price within 0.3% of support = at support
WHALE_FLUSH_MULT  = 3.0        # Skip if dip is 3x larger than average dip
MAX_HOLD_HOURS    = 3          # Force close after 3 hours
TRAIL_ACTIVATE    = 0.005      # Activate trailing stop at +0.5% profit
TRAIL_PCT         = 0.003      # Trail stop 0.3% below highest close
TRADE_LOG         = "trade_log.json"

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

trading = TradingClient(API_KEY, SECRET_KEY, paper=True)
data    = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
def get_bars(limit: int = 120) -> list[dict]:
    """Fetch recent 1-min BTC bars."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(minutes=limit * 2)
    req = CryptoBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start,
        end=end,
        limit=limit
    )
    try:
        df = data.get_crypto_bars(req).df
        if df.empty:
            return []
        if hasattr(df.index, 'levels'):
            sym = SYMBOL.replace("/", "")
            if sym in df.index.get_level_values(0):
                df = df.loc[sym]
            elif SYMBOL in df.index.get_level_values(0):
                df = df.loc[SYMBOL]
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
    except Exception as e:
        log.error(f"get_bars error: {e}")
        return []


# ─────────────────────────────────────────────
# SUPPORT & RESISTANCE
# ─────────────────────────────────────────────
def find_major_lows(bars: list[dict], n: int = 3) -> list[float]:
    """
    Find the n most significant lows in the bar set.
    A low is significant if it is lower than the 5 candles on either side.
    """
    lows = []
    for i in range(5, len(bars) - 5):
        window = bars[i-5:i] + bars[i+1:i+6]
        if all(bars[i]["l"] <= b["l"] for b in window):
            lows.append((bars[i]["l"], i))

    # Sort by price ascending, take the n lowest unique levels
    lows.sort(key=lambda x: x[0])
    unique_lows = []
    for low, idx in lows:
        if not any(abs(low - ul) / ul < 0.005 for ul in unique_lows):
            unique_lows.append(low)
        if len(unique_lows) == n:
            break
    return unique_lows


def find_major_highs(bars: list[dict], n: int = 3) -> list[float]:
    """Find the n most significant highs in the bar set."""
    highs = []
    for i in range(5, len(bars) - 5):
        window = bars[i-5:i] + bars[i+1:i+6]
        if all(bars[i]["h"] >= b["h"] for b in window):
            highs.append((bars[i]["h"], i))

    highs.sort(key=lambda x: x[0], reverse=True)
    unique_highs = []
    for high, idx in highs:
        if not any(abs(high - uh) / uh < 0.005 for uh in unique_highs):
            unique_highs.append(high)
        if len(unique_highs) == n:
            break
    return unique_highs


def calc_support(bars: list[dict]) -> float:
    """Average of the 3 major lows."""
    lows = find_major_lows(bars, 3)
    return sum(lows) / len(lows) if lows else bars[-1]["l"]


def calc_resistance(bars: list[dict]) -> float:
    """Average of the 3 major highs."""
    highs = find_major_highs(bars, 3)
    return sum(highs) / len(highs) if highs else bars[-1]["h"]


def calc_avg_dip_size(bars: list[dict]) -> float:
    """Average size of dips to give whale flush context."""
    lows = find_major_lows(bars, 5)
    if len(lows) < 2:
        return 0.0
    dips = []
    for i in range(1, len(lows)):
        dips.append(abs(lows[i] - lows[i-1]))
    return sum(dips) / len(dips) if dips else 0.0


# ─────────────────────────────────────────────
# CANDLE HELPERS
# ─────────────────────────────────────────────
def is_green(b: dict) -> bool:
    return b["c"] > b["o"]

def is_two_green(bars: list[dict]) -> bool:
    """Last 2 candles are both green and each closes higher."""
    if len(bars) < 2:
        return False
    c1, c2 = bars[-2], bars[-1]
    return is_green(c1) and is_green(c2) and c2["c"] > c1["c"]

def lowest_wick(bars: list[dict], lookback: int = 5) -> float:
    """Lowest low in the last n candles — used for stop placement."""
    return min(b["l"] for b in bars[-lookback:])


# ─────────────────────────────────────────────
# WHALE FLUSH FILTER
# ─────────────────────────────────────────────
def is_whale_flush(bars: list[dict], support: float) -> bool:
    """
    Returns True if the current dip to support is unusually large
    compared to the average dip — likely a manipulative flush.
    """
    current_dip = bars[-1]["c"] - support
    avg_dip = calc_avg_dip_size(bars)
    if avg_dip <= 0:
        return False
    if abs(current_dip) > WHALE_FLUSH_MULT * avg_dip:
        log.info(f"Whale flush detected — dip {abs(current_dip):.2f} vs avg {avg_dip:.2f}")
        return True
    return False


# ─────────────────────────────────────────────
# ENTRY SIGNAL
# ─────────────────────────────────────────────
def check_entry(bars: list[dict]) -> dict | None:
    """
    Returns entry dict if all conditions met, else None.

    Conditions:
    1. Price touched support (within 0.3%)
    2. Not a whale flush
    3. Last 2 candles are green and each closes higher (turnaround confirmed)
    4. Stop would be below lowest wick of the dip
    5. Target (midpoint) gives at least 1.5:1 reward:risk
    """
    if len(bars) < 20:
        return None

    support    = calc_support(bars)
    resistance = calc_resistance(bars)
    current    = bars[-1]["c"]
    midpoint   = (support + resistance) / 2

    log.info(
        f"Levels — support={support:.2f} | resistance={resistance:.2f} | "
        f"midpoint={midpoint:.2f} | current={current:.2f}"
    )

    # 1 — Price at support
    distance_pct = abs(current - support) / support
    if distance_pct > SUPPORT_TOUCH_PCT:
        log.info(f"Not at support — distance {distance_pct*100:.2f}% > {SUPPORT_TOUCH_PCT*100:.1f}%")
        return None

    # 2 — Whale flush check
    if is_whale_flush(bars, support):
        return None

    # 3 — Two green candles confirming turnaround
    if not is_two_green(bars):
        log.info("Waiting for 2 green candles to confirm turnaround")
        return None

    # 4 — Stop placement
    stop = lowest_wick(bars, 5) * 0.999  # just below lowest wick

    # 5 — Reward:risk check
    risk   = current - stop
    reward = midpoint - current
    if risk <= 0:
        return None
    rr = reward / risk
    if rr < 1.5:
        log.info(f"R/R too low — {rr:.2f} (need 1.5+)")
        return None

    log.info(
        f"ENTRY SIGNAL ✓ | current={current:.2f} | support={support:.2f} | "
        f"stop={stop:.2f} | target={midpoint:.2f} | R/R={rr:.2f}"
    )

    return {
        "entry":      current,
        "stop":       round(stop, 2),
        "target":     round(midpoint, 2),
        "support":    round(support, 2),
        "resistance": round(resistance, 2),
        "rr":         round(rr, 2),
    }


# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────
def get_account_balance() -> float:
    try:
        account = trading.get_account()
        return float(account.cash)
    except Exception as e:
        log.error(f"get_account_balance error: {e}")
        return 10000.0

def calc_position_size(entry: float, stop: float, balance: float) -> float:
    max_risk      = balance * RISK_PER_TRADE
    risk_per_unit = entry - stop
    if risk_per_unit <= 0:
        return 0.0
    qty = max_risk / risk_per_unit
    return round(qty, 6)


# ─────────────────────────────────────────────
# ORDER EXECUTION
# ─────────────────────────────────────────────
def place_buy(qty: float) -> bool:
    try:
        req = MarketOrderRequest(
            symbol=SYMBOL.replace("/", ""),
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )
        order = trading.submit_order(req)
        log.info(f"BUY order placed | qty={qty} | id={order.id}")
        return True
    except Exception as e:
        log.error(f"place_buy error: {e}")
        return False

def place_sell(qty: float, reason: str = "exit") -> bool:
    try:
        req = MarketOrderRequest(
            symbol=SYMBOL.replace("/", ""),
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        )
        order = trading.submit_order(req)
        log.info(f"SELL order placed | qty={qty} | reason={reason} | id={order.id}")
        return True
    except Exception as e:
        log.error(f"place_sell error: {e}")
        return False

def get_open_positions() -> list:
    try:
        positions = trading.get_all_positions()
        return [p for p in positions if p.symbol == SYMBOL.replace("/", "")]
    except Exception as e:
        log.error(f"get_open_positions error: {e}")
        return []


# ─────────────────────────────────────────────
# TRADE LOG
# ─────────────────────────────────────────────
def load_log() -> dict:
    if not os.path.exists(TRADE_LOG):
        return {
            "trades": [],
            "daily_pnl": 0.0,
            "last_reset": str(datetime.now(timezone.utc).date())
        }
    with open(TRADE_LOG) as f:
        return json.load(f)

def save_log(log_data: dict):
    with open(TRADE_LOG, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

def reset_daily_if_new_day(log_data: dict) -> dict:
    today = str(datetime.now(timezone.utc).date())
    if log_data.get("last_reset") != today:
        log_data["daily_pnl"] = 0.0
        log_data["last_reset"] = today
        log.info("New day — daily P&L reset to 0")
    return log_data


# ─────────────────────────────────────────────
# POSITION MANAGEMENT
# ─────────────────────────────────────────────
def manage_positions(bars: list[dict], log_data: dict):
    positions = get_open_positions()
    if not positions:
        return

    current = bars[-1]["c"]
    highest_close = max(b["c"] for b in bars[-60:])

    for pos in positions:
        qty  = abs(float(pos.qty))
        cost = float(pos.avg_entry_price)
        pnl_pct = (current - cost) / cost

        # Find trade record
        trade = next(
            (t for t in reversed(log_data["trades"])
             if t.get("status") == "open"),
            None
        )

        stop   = trade["stop"]   if trade else cost * 0.995
        target = trade["target"] if trade else cost * 1.01
        entry_time = datetime.fromisoformat(trade["entry_time"]) if trade else datetime.now(timezone.utc)

        # Trailing stop — activate at +0.5%
        if pnl_pct >= TRAIL_ACTIVATE:
            trail_stop = highest_close * (1 - TRAIL_PCT)
            if trail_stop > stop:
                stop = round(trail_stop, 2)
                log.info(f"Trail stop updated → {stop:.2f}")
                if trade:
                    trade["stop"] = stop
                    save_log(log_data)

        log.info(
            f"Position | price={current:.2f} | cost={cost:.2f} | "
            f"pnl={pnl_pct*100:.2f}% | stop={stop:.2f} | target={target:.2f}"
        )

        # Target hit — full exit
        if current >= target:
            if place_sell(qty, reason="target_hit"):
                pnl = (current - cost) * qty
                log_data["daily_pnl"] = log_data.get("daily_pnl", 0) + pnl
                log.info(f"TARGET HIT ✓ | P&L: ${pnl:.2f}")
                if trade:
                    trade["status"]     = "closed"
                    trade["exit_price"] = current
                    trade["exit_reason"]= "target"
                    trade["pnl"]        = round(pnl, 2)
                save_log(log_data)

        # Stop hit
        elif current <= stop:
            if place_sell(qty, reason="stop_loss"):
                pnl = (current - cost) * qty
                log_data["daily_pnl"] = log_data.get("daily_pnl", 0) + pnl
                log.info(f"STOP HIT | P&L: ${pnl:.2f}")
                if trade:
                    trade["status"]     = "closed"
                    trade["exit_price"] = current
                    trade["exit_reason"]= "stop"
                    trade["pnl"]        = round(pnl, 2)
                save_log(log_data)

        # Time exit — 3 hours max
        elif (datetime.now(timezone.utc) - entry_time).total_seconds() > MAX_HOLD_HOURS * 3600:
            if place_sell(qty, reason="time_exit"):
                pnl = (current - cost) * qty
                log_data["daily_pnl"] = log_data.get("daily_pnl", 0) + pnl
                log.info(f"TIME EXIT | P&L: ${pnl:.2f}")
                if trade:
                    trade["status"]     = "closed"
                    trade["exit_price"] = current
                    trade["exit_reason"]= "time"
                    trade["pnl"]        = round(pnl, 2)
                save_log(log_data)


# ─────────────────────────────────────────────
# KEEP-ALIVE SERVER
# ─────────────────────────────────────────────
def start_keepalive():
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"btc_bot alive")
        def log_message(self, *args):
            pass

    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info(f"Keep-alive server on port {port}")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def run():
    log.info("=" * 60)
    log.info("BTC Support/Resistance Bot starting")
    log.info("Strategy: buy the dip at support, 2 green candle confirm")
    log.info("=" * 60)

    start_keepalive()
    log_data = load_log()

    while True:
        try:
            cycle_start = time.time()
            log_data = reset_daily_if_new_day(log_data)

            # Daily circuit breaker
            balance = get_account_balance()
            if log_data.get("daily_pnl", 0) <= balance * DAILY_DD_LIMIT:
                log.warning("Daily loss limit hit — pausing until midnight UTC")
                now      = datetime.now(timezone.utc)
                midnight = (now + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                time.sleep((midnight - now).total_seconds())
                continue

            # Fetch bars
            bars = get_bars(SUPPORT_LOOKBACK)
            if len(bars) < 20:
                log.warning("Not enough bars — waiting")
                time.sleep(30)
                continue

            current = bars[-1]["c"]
            log.info(f"BTC={current:.2f} | balance=${balance:.2f} | daily_pnl=${log_data.get('daily_pnl',0):.2f}")

            # Manage open positions first
            manage_positions(bars, log_data)

            # Check position count
            open_pos = get_open_positions()
            if len(open_pos) >= MAX_POSITIONS:
                log.info(f"Max positions ({MAX_POSITIONS}) open — no new entries")
                time.sleep(60)
                continue

            # Check for entry signal
            signal = check_entry(bars)

            if signal:
                qty = calc_position_size(signal["entry"], signal["stop"], balance)
                if qty <= 0:
                    log.warning("Position size 0 — skipping")
                    time.sleep(60)
                    continue

                success = place_buy(qty)
                if success:
                    trade = {
                        "symbol":     SYMBOL,
                        "entry_time": str(datetime.now(timezone.utc)),
                        "entry":      signal["entry"],
                        "stop":       signal["stop"],
                        "target":     signal["target"],
                        "support":    signal["support"],
                        "resistance": signal["resistance"],
                        "rr":         signal["rr"],
                        "qty":        qty,
                        "status":     "open",
                        "exit_price": None,
                        "exit_reason":None,
                        "pnl":        None,
                    }
                    log_data["trades"].append(trade)
                    save_log(log_data)

            # Sleep until next minute
            elapsed    = time.time() - cycle_start
            sleep_time = max(60 - elapsed, 5)
            log.info(f"Sleeping {sleep_time:.0f}s")
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            log.info("Bot stopped")
            break
        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)
            time.sleep(30)


if __name__ == "__main__":
    run()
