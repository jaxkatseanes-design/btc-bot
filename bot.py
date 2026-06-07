"""
BTC Paper Trading Bot — Render Web Service
==========================================
Strategy: v10/v11 winner
  - MACD body 0.55 filter
  - EMA pullback + liquidity sweep entries
  - sweep/pull ATR stop 1.0x, MACD stop 1.3x
  - Target 7x ATR
  - RSI max 55 (MACD), 60 (pull/sweep)
  - Vol filter 1.2x ATR avg
  - Trail 1% after 2R
  - 4% risk per trade, compounding
  - Longs only, Alpaca crypto paper trading

Deployment: Render free tier web service
Trigger:    cron-job.org pings /run every 4 hours

Endpoints:
  GET /          — health check, shows current state
  GET /run       — triggered by cron, runs strategy check
  GET /status    — full trade log and metrics
  GET /reset     — emergency: cancel open orders, clear state
"""

import os
import json
import math
import logging
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify

# ── Alpaca ───────────────────────────────────────────────────────────────────
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_KEY    = os.environ.get("ALPACA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
PAPER      = True   # Always paper trading

SYMBOL        = "BTC/USD"
SYMBOL_TRADE  = "BTCUSD"   # Alpaca trading symbol (no slash)
RISK_PCT      = 0.04        # 4% risk per trade
TRAIL_PCT     = 0.010       # 1% trailing stop after 2R
TARGET_ATR    = 7.0
MACD_STOP_ATR = 1.3
PULL_STOP_ATR = 1.0
SWEEP_STOP_ATR= 1.0
RSI_MAX_MACD  = 55
RSI_MAX_OTHER = 60
VOL_MULT      = 1.2
MACD_BODY_MIN = 0.55
PULL_BODY_MIN = 0.40
SWEEP_BODY_MIN= 0.40
MAX_HOLD_BARS = 84          # 14 days on 4h candles
LOOKBACK      = 210         # bars needed for indicators

STATE_FILE    = "state.json"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

app = Flask(__name__)


# ─────────────────────────────────────────────
# STATE MANAGEMENT
# Persists open trade info between cron pings.
# On Render free tier, file resets on redeploy —
# bot re-syncs from Alpaca positions on every /run.
# ─────────────────────────────────────────────
def load_state():
    default = {
        "open_trade": None,      # dict or None
        "high_water": 0.0,       # highest price seen in open trade
        "trade_log": [],         # list of closed trade dicts
        "run_count": 0,
        "last_run": None,
        "errors": [],
    }
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                saved = json.load(f)
                default.update(saved)
    except Exception as e:
        log.warning(f"State load failed: {e} — using default")
    return default

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        log.error(f"State save failed: {e}")

def log_error(state, msg):
    state["errors"].append({"time": str(datetime.now(timezone.utc)), "msg": msg})
    state["errors"] = state["errors"][-20:]  # keep last 20


# ─────────────────────────────────────────────
# ALPACA CLIENTS
# ─────────────────────────────────────────────
def get_clients():
    data_client  = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
    trade_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
    return data_client, trade_client

def get_account_balance(trade_client):
    """Return current portfolio value from Alpaca paper account."""
    try:
        account = trade_client.get_account()
        return float(account.portfolio_value)
    except Exception as e:
        log.error(f"Failed to get account balance: {e}")
        return None

def get_open_position(trade_client):
    """Return open BTC position from Alpaca, or None."""
    try:
        positions = trade_client.get_all_positions()
        for p in positions:
            if p.symbol == SYMBOL_TRADE:
                return {
                    "qty": float(p.qty),
                    "avg_entry": float(p.avg_entry_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                }
        return None
    except Exception as e:
        log.error(f"Failed to get positions: {e}")
        return None

def cancel_all_orders(trade_client):
    try:
        trade_client.cancel_orders()
        log.info("All open orders cancelled")
    except Exception as e:
        log.error(f"Cancel orders failed: {e}")


# ─────────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────────
def fetch_bars(data_client):
    """Fetch last LOOKBACK+10 4h BTC candles."""
    try:
        end   = datetime.now(timezone.utc)
        start = end - timedelta(hours=(LOOKBACK + 10) * 4)
        req = CryptoBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame(4, TimeFrameUnit.Hour),
            start=start, end=end)
        df = data_client.get_crypto_bars(req).df
        if df.empty: return []
        if hasattr(df.index, "levels"):
            sym = SYMBOL.replace("/", "")
            if sym in df.index.get_level_values(0):   df = df.loc[sym]
            elif SYMBOL in df.index.get_level_values(0): df = df.loc[SYMBOL]
        bars = []
        for _, row in df.iterrows():
            bars.append({
                "o": float(row["open"]),  "h": float(row["high"]),
                "l": float(row["low"]),   "c": float(row["close"]),
                "v": float(row["volume"])
            })
        log.info(f"Fetched {len(bars)} 4h candles")
        return bars
    except Exception as e:
        log.error(f"fetch_bars failed: {e}")
        return []


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def ema(bars, period):
    closes = [b["c"] for b in bars]
    if len(closes) < period: return closes[-1]
    k = 2 / (period + 1); e = sum(closes[:period]) / period
    for c in closes[period:]: e = c * k + e * (1 - k)
    return e

def rsi(bars, period=14):
    closes = [b["c"] for b in bars[-(period + 2):]]
    if len(closes) < period + 1: return 50.0
    ch = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    g = sum(x for x in ch if x > 0) / period
    l = sum(abs(x) for x in ch if x < 0) / period
    if l == 0: return 100.0
    return 100 - (100 / (1 + g / l))

def atr(bars, period=14):
    if len(bars) < period + 1: return bars[-1]["h"] - bars[-1]["l"]
    trs = [max(bars[i]["h"] - bars[i]["l"],
               abs(bars[i]["h"] - bars[i-1]["c"]),
               abs(bars[i]["l"] - bars[i-1]["c"])) for i in range(1, len(bars))]
    return sum(trs[-period:]) / period

def atr_avg(bars, period=14, avg_period=20):
    if len(bars) < period + avg_period + 2: return atr(bars, period)
    atrs = [atr(bars[:len(bars) - i], period) for i in range(avg_period)]
    return sum(atrs) / len(atrs)

def macd_vals(bars, fast=12, slow=26, signal=9):
    closes = [b["c"] for b in bars]
    if len(closes) < slow + signal + 2: return 0, 0, 0, 0
    def _ema(v, p):
        k = 2 / (p + 1); e = sum(v[:p]) / p
        for x in v[p:]: e = x * k + e * (1 - k)
        return e
    def _ema_s(v, p):
        k = 2 / (p + 1); e = sum(v[:p]) / p; r = [e]
        for x in v[p:]: e = x * k + e * (1 - k); r.append(e)
        return r
    fs = _ema_s(closes, fast); ss = _ema_s(closes, slow)
    ml = min(len(fs), len(ss))
    md = [fs[-ml + i] - ss[-ml + i] for i in range(ml)]
    if len(md) < signal + 2: return md[-1], md[-2] if len(md) > 1 else 0, 0, 0
    sn = _ema(md, signal); sp = _ema(md[:-1], signal)
    return md[-1], md[-2], sn, sp

def swing_lows(bars, n=3, w=4):
    lows = []
    for i in range(w, len(bars) - w):
        if all(bars[i]["l"] <= bars[j]["l"] for j in range(i - w, i + w + 1) if j != i):
            lows.append(bars[i]["l"])
    lows.sort()
    u = []
    for l in lows:
        if not any(abs(l - x) / x < 0.005 for x in u): u.append(l)
        if len(u) == n: break
    return u

def is_green(b):  return b["c"] > b["o"]
def strong(b, p): return abs(b["c"] - b["o"]) / max(b["h"] - b["l"], 0.0001) >= p


# ─────────────────────────────────────────────
# SIGNAL DETECTION
# ─────────────────────────────────────────────
def check_signal(bars):
    """
    Returns signal dict or None.
    Signal: {type, entry, stop, target, atr}
    """
    if len(bars) < LOOKBACK: return None
    window = bars[-LOOKBACK:]
    current = window[-1]["c"]
    a = atr(window, 14)
    if a == 0: return None

    e20  = ema(window, 20)
    e50  = ema(window, 50)
    e200 = ema(window, 200) if len(window) >= 200 else e50
    r    = rsi(window, 14)

    # Vol filter
    avg_a = atr_avg(window, 14, 20)
    if a > VOL_MULT * avg_a:
        log.info(f"Vol filter: ATR {a:.0f} > {VOL_MULT}x avg {avg_a:.0f} — skip")
        return None

    # ── MACD ────────────────────────────────────────────────────────────────
    ml, ml_p, ms, ms_p = macd_vals(window, 12, 26, 9)
    if (current >= e200 * 0.998 and r <= RSI_MAX_MACD and
            ml > ms and ml_p <= ms_p and
            is_green(window[-1]) and strong(window[-1], MACD_BODY_MIN) and
            (ml - ms) > (ml_p - ms_p)):
        stop   = current - MACD_STOP_ATR * a
        target = current + TARGET_ATR * a
        risk   = current - stop
        if risk > 0 and (target - current) / risk >= 1.5:
            log.info(f"MACD signal: entry={current:.0f} stop={stop:.0f} target={target:.0f}")
            return {"type": "macd", "entry": current, "stop": stop,
                    "target": target, "atr": a}

    # ── EMA PULLBACK ─────────────────────────────────────────────────────────
    if (current >= e50 and e20 >= e50 and
            e20 * 0.998 <= current <= e20 * 1.008 and r <= RSI_MAX_OTHER and
            is_green(window[-1]) and strong(window[-1], PULL_BODY_MIN) and
            is_green(window[-2])):
        stop   = current - PULL_STOP_ATR * a
        target = current + TARGET_ATR * a
        risk   = current - stop
        if risk > 0 and (target - current) / risk >= 1.5:
            log.info(f"EMA pull signal: entry={current:.0f} stop={stop:.0f} target={target:.0f}")
            return {"type": "ema_pull", "entry": current, "stop": stop,
                    "target": target, "atr": a}

    # ── LIQUIDITY SWEEP ───────────────────────────────────────────────────────
    lows = swing_lows(window[:-3], 3, w=4)
    if lows:
        support = sum(lows) / len(lows)
        swept = any(window[-i]["l"] < support * 0.998 for i in range(2, 6))
        if (swept and current > support and r < RSI_MAX_OTHER and
                is_green(window[-1]) and strong(window[-1], SWEEP_BODY_MIN) and
                is_green(window[-2])):
            stop   = current - SWEEP_STOP_ATR * a
            target = current + TARGET_ATR * a
            risk   = current - stop
            if risk > 0 and (target - current) / risk >= 1.5:
                log.info(f"Sweep signal: entry={current:.0f} stop={stop:.0f} target={target:.0f}")
                return {"type": "sweep", "entry": current, "stop": stop,
                        "target": target, "atr": a}

    log.info(f"No signal — BTC={current:.0f} RSI={r:.1f} ATR={a:.0f}")
    return None


# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────
def calc_qty(balance, entry, stop):
    """
    Risk RISK_PCT of balance on this trade.
    qty = (balance * risk_pct) / (entry - stop)
    Round down to 6 decimal places (Alpaca minimum).
    """
    risk_per_unit = entry - stop
    if risk_per_unit <= 0: return 0
    dollar_risk = balance * RISK_PCT
    qty = dollar_risk / risk_per_unit
    return math.floor(qty * 1e6) / 1e6  # floor to 6dp


# ─────────────────────────────────────────────
# ORDER EXECUTION
# ─────────────────────────────────────────────
def place_buy(trade_client, qty, signal, balance):
    """Place market buy order."""
    try:
        order = trade_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL_TRADE,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )
        )
        log.info(f"BUY ORDER placed: {qty} BTC @ ~{signal['entry']:.0f} | "
                 f"stop={signal['stop']:.0f} target={signal['target']:.0f} | "
                 f"order_id={order.id}")
        return str(order.id)
    except Exception as e:
        log.error(f"Buy order failed: {e}")
        return None

def place_sell(trade_client, qty, reason):
    """Place market sell order to close position."""
    try:
        order = trade_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL_TRADE,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            )
        )
        log.info(f"SELL ORDER placed: {qty} BTC | reason={reason} | order_id={order.id}")
        return str(order.id)
    except Exception as e:
        log.error(f"Sell order failed: {e}")
        return None


# ─────────────────────────────────────────────
# CORE STRATEGY LOOP
# Called on every /run ping
# ─────────────────────────────────────────────
def run_strategy():
    log.info("=" * 60)
    log.info("Strategy run started")

    state = load_state()
    state["run_count"] += 1
    state["last_run"] = str(datetime.now(timezone.utc))

    if not API_KEY or not SECRET_KEY:
        msg = "API keys not set"
        log.error(msg)
        log_error(state, msg)
        save_state(state)
        return {"status": "error", "msg": msg}

    try:
        data_client, trade_client = get_clients()
    except Exception as e:
        msg = f"Client init failed: {e}"
        log.error(msg)
        log_error(state, msg)
        save_state(state)
        return {"status": "error", "msg": msg}

    # ── Sync with Alpaca: check actual open position ──────────────────────────
    alpaca_position = get_open_position(trade_client)
    balance = get_account_balance(trade_client)

    if balance is None:
        msg = "Could not fetch account balance"
        log.error(msg)
        log_error(state, msg)
        save_state(state)
        return {"status": "error", "msg": msg}

    log.info(f"Account balance: ${balance:,.2f}")

    # Reconcile state with Alpaca
    # If Alpaca shows no position but state thinks we're in a trade,
    # the position was closed externally — clear state trade
    if state["open_trade"] and alpaca_position is None:
        log.warning("State had open trade but Alpaca shows no position — clearing")
        state["open_trade"] = None
        state["high_water"] = 0.0

    # If Alpaca shows position but state has no trade (e.g. after redeploy),
    # reconstruct minimal trade state from Alpaca data
    if alpaca_position and state["open_trade"] is None:
        log.warning("Alpaca has position but no state — reconstructing from Alpaca")
        avg_entry = alpaca_position["avg_entry"]
        # Can't recover stop/target without original signal — use safe defaults
        a_est = avg_entry * 0.02  # estimate ATR as 2% of price
        state["open_trade"] = {
            "type": "unknown",
            "entry": avg_entry,
            "stop": avg_entry - MACD_STOP_ATR * a_est,
            "target": avg_entry + TARGET_ATR * a_est,
            "init_stop": avg_entry - MACD_STOP_ATR * a_est,
            "qty": alpaca_position["qty"],
            "entry_time": str(datetime.now(timezone.utc)),
            "bars_held": 0,
        }
        state["high_water"] = avg_entry

    # ── Fetch market data ─────────────────────────────────────────────────────
    bars = fetch_bars(data_client)
    if len(bars) < LOOKBACK:
        msg = f"Not enough bars: {len(bars)} < {LOOKBACK}"
        log.warning(msg)
        save_state(state)
        return {"status": "ok", "msg": msg}

    current_price = bars[-1]["c"]
    log.info(f"Current BTC price: ${current_price:,.0f}")

    # ── MANAGE OPEN TRADE ─────────────────────────────────────────────────────
    if state["open_trade"]:
        trade = state["open_trade"]
        cost      = trade["entry"]
        stop      = trade["stop"]
        target    = trade["target"]
        init_stop = trade["init_stop"]
        qty       = trade["qty"]
        init_risk = cost - init_stop

        # Update high water mark
        if current_price > state["high_water"]:
            state["high_water"] = current_price

        # Increment bars held
        trade["bars_held"] = trade.get("bars_held", 0) + 1

        # Update trailing stop after 2R
        if init_risk > 0 and (current_price - cost) / init_risk >= 2.0:
            trail_stop = state["high_water"] * (1 - TRAIL_PCT)
            if trail_stop > stop:
                old_stop = stop
                trade["stop"] = trail_stop
                stop = trail_stop
                log.info(f"Trail stop updated: {old_stop:.0f} → {stop:.0f} "
                         f"(HWM={state['high_water']:.0f})")

        # Check exit conditions
        exit_reason = None
        if current_price >= target:
            exit_reason = "target"
        elif current_price <= stop:
            exit_reason = "stop"
        elif trade["bars_held"] >= MAX_HOLD_BARS:
            exit_reason = "max_hold"

        if exit_reason:
            log.info(f"EXIT triggered: {exit_reason} | "
                     f"entry={cost:.0f} current={current_price:.0f} "
                     f"pnl={((current_price-cost)/cost*100):+.1f}%")
            order_id = place_sell(trade_client, qty, exit_reason)
            if order_id:
                pnl_pct = (current_price - cost) / cost * 100
                closed = {
                    "type": trade["type"],
                    "entry": cost,
                    "exit": current_price,
                    "stop_was": stop,
                    "target": target,
                    "qty": qty,
                    "pnl_pct": round(pnl_pct, 2),
                    "win": current_price > cost,
                    "reason": exit_reason,
                    "bars_held": trade["bars_held"],
                    "entry_time": trade.get("entry_time"),
                    "exit_time": str(datetime.now(timezone.utc)),
                }
                state["trade_log"].append(closed)
                state["open_trade"] = None
                state["high_water"] = 0.0
                log.info(f"Trade closed: pnl={pnl_pct:+.1f}% reason={exit_reason}")
        else:
            log.info(f"Holding: entry={cost:.0f} stop={stop:.0f} "
                     f"target={target:.0f} current={current_price:.0f} "
                     f"bars={trade['bars_held']} "
                     f"pnl={((current_price-cost)/cost*100):+.1f}%")
            # Save updated stop
            state["open_trade"] = trade

    # ── LOOK FOR ENTRY ────────────────────────────────────────────────────────
    elif alpaca_position is None:
        signal = check_signal(bars)
        if signal:
            qty = calc_qty(balance, signal["entry"], signal["stop"])
            if qty <= 0:
                log.warning(f"Qty too small ({qty}) — skipping entry")
            else:
                # Check we have enough buying power
                cost_est = qty * signal["entry"]
                if cost_est > balance * 0.95:
                    log.warning(f"Order cost ${cost_est:,.0f} exceeds 95% of balance — skip")
                else:
                    order_id = place_buy(trade_client, qty, signal, balance)
                    if order_id:
                        state["open_trade"] = {
                            "type":       signal["type"],
                            "entry":      signal["entry"],
                            "stop":       signal["stop"],
                            "target":     signal["target"],
                            "init_stop":  signal["stop"],
                            "qty":        qty,
                            "entry_time": str(datetime.now(timezone.utc)),
                            "bars_held":  0,
                            "order_id":   order_id,
                            "atr":        signal["atr"],
                        }
                        state["high_water"] = signal["entry"]
                        risk_dollars = balance * RISK_PCT
                        log.info(f"Trade opened: {signal['type']} | "
                                 f"qty={qty} risk=${risk_dollars:,.0f}")
        else:
            log.info("No signal — staying flat")

    # ── METRICS ───────────────────────────────────────────────────────────────
    tlog = state["trade_log"]
    if tlog:
        wins   = [t for t in tlog if t["win"]]
        losses = [t for t in tlog if not t["win"]]
        wr     = len(wins) / len(tlog) * 100
        avg_w  = sum(t["pnl_pct"] for t in wins)   / len(wins)   if wins   else 0
        avg_l  = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
        log.info(f"Stats: {len(tlog)} trades | WR={wr:.1f}% | "
                 f"AvgW={avg_w:+.1f}% AvgL={avg_l:+.1f}%")

    save_state(state)
    log.info("Strategy run complete")
    log.info("=" * 60)

    return {
        "status":     "ok",
        "balance":    balance,
        "btc_price":  current_price,
        "open_trade": state["open_trade"] is not None,
        "trade_type": state["open_trade"]["type"] if state["open_trade"] else None,
        "trades":     len(state["trade_log"]),
        "run":        state["run_count"],
    }


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def health():
    state = load_state()
    return jsonify({
        "status":     "running",
        "bot":        "BTC Paper Trading Bot",
        "strategy":   "v10/v11 body0.55 4%risk",
        "last_run":   state["last_run"],
        "run_count":  state["run_count"],
        "open_trade": state["open_trade"] is not None,
        "trades":     len(state["trade_log"]),
    })

@app.route("/run")
def run():
    """Called by cron every 4 hours."""
    try:
        result = run_strategy()
        return jsonify(result)
    except Exception as e:
        log.error(f"/run unhandled exception: {e}", exc_info=True)
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route("/status")
def status():
    """Full state and trade log."""
    state = load_state()
    tlog  = state["trade_log"]
    wins  = [t for t in tlog if t["win"]]
    losses= [t for t in tlog if not t["win"]]

    metrics = {}
    if tlog:
        wr     = len(wins) / len(tlog) * 100
        avg_w  = sum(t["pnl_pct"] for t in wins)   / len(wins)   if wins   else 0
        avg_l  = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
        rr     = abs(avg_w / avg_l) if avg_l else 0
        metrics = {
            "total_trades": len(tlog),
            "wins":         len(wins),
            "losses":       len(losses),
            "win_rate_pct": round(wr, 1),
            "avg_win_pct":  round(avg_w, 2),
            "avg_loss_pct": round(avg_l, 2),
            "rr":           round(rr, 2),
        }

    return jsonify({
        "open_trade":  state["open_trade"],
        "high_water":  state["high_water"],
        "metrics":     metrics,
        "trade_log":   tlog,
        "run_count":   state["run_count"],
        "last_run":    state["last_run"],
        "errors":      state["errors"],
    })

@app.route("/reset")
def reset():
    """
    Emergency reset — cancel all orders, clear open trade state.
    Does NOT close open positions — do that manually in Alpaca dashboard.
    """
    try:
        _, trade_client = get_clients()
        cancel_all_orders(trade_client)
    except Exception as e:
        log.error(f"Reset cancel failed: {e}")

    state = load_state()
    state["open_trade"] = None
    state["high_water"] = 0.0
    save_state(state)
    log.warning("RESET executed — open trade state cleared")
    return jsonify({
        "status": "reset",
        "msg": "State cleared. Check Alpaca dashboard for open positions."
    })


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    log.info(f"Starting BTC Paper Trading Bot on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
