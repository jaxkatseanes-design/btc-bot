"""
BTC Strategy Multi-Window Validator v3 — 6-MONTH WINDOWS
=========================================================
Tests Strategy B (v10/v11 winner: body 0.55, no time stop, 4% risk)
across every 6-month window from 2022 to present.

Windows:
  2022 H1 (Jan-Jun)   — BTC peak to crash
  2022 H2 (Jul-Dec)   — bottom, sideways
  2023 H1 (Jan-Jun)   — recovery begins
  2023 H2 (Jul-Dec)   — consolidation, then surge
  2024 H1 (Jan-Jun)   — bull run, new ATH
  2024 H2 (Jul-Dec)   — post-ATH, volatile
  2025 H1 (Jan-Jun)   — current period

Goal: find which 6-month regimes the strategy struggles in,
then target those specific conditions in v12.

Pass criteria: positive return in 5+ of 7 windows.
"""

import os
from datetime import datetime, timezone, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

SYMBOL       = "BTC/USD"
STARTING_BAL = 100000.0
FEE_PCT      = 0.0015
MAX_HOLD     = 84

API_KEY    = os.environ.get("ALPACA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

WINDOWS = [
    ("2022 H1 (crash)   ", datetime(2022, 1, 1, tzinfo=timezone.utc),  datetime(2022, 6, 30, tzinfo=timezone.utc)),
    ("2022 H2 (bottom)  ", datetime(2022, 7, 1, tzinfo=timezone.utc),  datetime(2022, 12, 31, tzinfo=timezone.utc)),
    ("2023 H1 (recovery)", datetime(2023, 1, 1, tzinfo=timezone.utc),  datetime(2023, 6, 30, tzinfo=timezone.utc)),
    ("2023 H2 (surge)   ", datetime(2023, 7, 1, tzinfo=timezone.utc),  datetime(2023, 12, 31, tzinfo=timezone.utc)),
    ("2024 H1 (bull ATH)", datetime(2024, 1, 1, tzinfo=timezone.utc),  datetime(2024, 6, 30, tzinfo=timezone.utc)),
    ("2024 H2 (volatile)", datetime(2024, 7, 1, tzinfo=timezone.utc),  datetime(2024, 12, 31, tzinfo=timezone.utc)),
    ("2025 H1 (current) ", datetime(2025, 1, 1, tzinfo=timezone.utc),  datetime(2025, 6, 30, tzinfo=timezone.utc)),
]


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
# STRATEGY B: body 0.55, no time stop, 4% risk
# ─────────────────────────────────────────────
def signal_fn(bars):
    if len(bars) < 80: return None
    current = bars[-1]["c"]
    a = atr(bars, 14)
    if a == 0: return None

    e20  = ema(bars, 20)
    e50  = ema(bars, 50)
    e200 = ema(bars, 200) if len(bars) >= 200 else e50
    r    = rsi(bars, 14)

    avg_a = atr_avg(bars, 14, 20)
    if a > 1.2 * avg_a: return None

    # MACD — body 0.55
    ml, ml_p, ms, ms_p = macd_vals(bars, 12, 26, 9)
    if (current >= e200 * 0.998 and r <= 55 and
            ml > ms and ml_p <= ms_p and
            is_green(bars[-1]) and strong(bars[-1], 0.55) and
            (ml - ms) > (ml_p - ms_p)):
        stop = current - 1.3 * a; target = current + 7.0 * a
        risk = current - stop
        if risk > 0 and (target - current) / risk >= 1.5:
            return ("long", current, stop, target, 0.04, 0.01, True, "macd", a)

    # EMA Pull
    if (current >= e50 and e20 >= e50 and
            e20 * 0.998 <= current <= e20 * 1.008 and r <= 60 and
            is_green(bars[-1]) and strong(bars[-1], 0.40) and is_green(bars[-2])):
        stop = current - 1.0 * a; target = current + 7.0 * a
        risk = current - stop
        if risk > 0 and (target - current) / risk >= 1.5:
            return ("long", current, stop, target, 0.04, 0.01, True, "ema_pull", a)

    # Sweep
    lows = swing_lows(bars[:-3], 3, w=4)
    if lows:
        support = sum(lows) / len(lows)
        swept = any(bars[-i]["l"] < support * 0.998 for i in range(2, 6))
        if (swept and current > support and r < 60 and
                is_green(bars[-1]) and strong(bars[-1], 0.40) and is_green(bars[-2])):
            stop = current - 1.0 * a; target = current + 7.0 * a
            risk = current - stop
            if risk > 0 and (target - current) / risk >= 1.5:
                return ("long", current, stop, target, 0.04, 0.01, True, "sweep", a)

    return None


# ─────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────
def backtest(all_bars):
    balance    = STARTING_BAL
    trades     = []
    open_trade = None
    high_water = 0.0
    LOOKBACK   = 210

    for i in range(LOOKBACK, len(all_bars)):
        window  = all_bars[i - LOOKBACK:i + 1]
        current = all_bars[i]["c"]

        if open_trade:
            held      = i - open_trade["idx"]
            cost      = open_trade["entry"]
            stop      = open_trade["stop"]
            target    = open_trade["target"]
            init_stop = open_trade["init_stop"]
            trail_p   = open_trade["trail_pct"]
            init_risk = abs(cost - init_stop)

            if current > high_water: high_water = current

            if init_risk > 0 and (current - cost) / init_risk >= 2.0:
                ts = high_water * (1 - trail_p)
                if ts > stop:
                    open_trade["stop"] = ts
                    stop = ts

            er = None; ep = current
            if current >= target: er = "target"
            elif current <= stop: er = "stop"; ep = stop
            if held >= MAX_HOLD:  er = "time"

            if er:
                qty   = (balance * 0.04) / init_risk if init_risk > 0 else 1
                gross = (ep - cost) * qty
                fee   = (cost * qty * FEE_PCT) + (ep * qty * FEE_PCT)
                net   = gross - fee
                balance += net
                trades.append({
                    "net": net, "win": net > 0, "reason": er,
                    "fee": fee, "hold": held,
                    "etype": open_trade.get("etype", "macd"),
                })
                open_trade = None; high_water = 0.0
            continue

        sig = signal_fn(window)
        if sig:
            _, entry, stop, target, r_pct, trail_p, cmp, etype, ea = sig
            if stop >= entry or target <= entry: continue
            open_trade = {
                "idx": i, "entry": entry, "stop": stop, "target": target,
                "init_stop": stop, "trail_pct": trail_p, "etype": etype,
            }
            high_water = entry

    return {"trades": trades, "final": balance}


# ─────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────
def fetch_window(label, start, end):
    warmup = start - timedelta(days=365)
    client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
    req = CryptoBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(4, TimeFrameUnit.Hour),
        start=warmup, end=end)
    df = client.get_crypto_bars(req).df
    if df.empty: return []
    if hasattr(df.index, "levels"):
        sym = SYMBOL.replace("/", "")
        if sym in df.index.get_level_values(0):   df = df.loc[sym]
        elif SYMBOL in df.index.get_level_values(0): df = df.loc[SYMBOL]
    bars = []
    for ts, row in df.iterrows():
        bt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        bars.append({"o": float(row["open"]), "h": float(row["high"]),
                     "l": float(row["low"]),  "c": float(row["close"]),
                     "v": float(row["volume"]), "t": bt})
    print(f"  {label.strip()}: {len(bars):,} bars loaded")
    return bars


# ─────────────────────────────────────────────
# SUMMARIZE + PRINT
# ─────────────────────────────────────────────
def summarize(label, result, btc_s, btc_e):
    trades = result["trades"]; final = result["final"]; n = len(trades)
    btc_chg = (btc_e - btc_s) / btc_s * 100 if btc_s else 0
    if n == 0:
        return {"label": label, "trades": 0, "win_rate": 0, "total_pct": 0,
                "avg_win": 0, "avg_loss": 0, "rr": 0, "fees": 0,
                "expect": 0, "btc_chg": btc_chg, "final": final,
                "macd_t": 0, "sweep_t": 0, "pull_t": 0, "stop_pct": 0,
                "target_pct": 0}
    winners = [t for t in trades if t["win"]]
    losers  = [t for t in trades if not t["win"]]
    stops   = [t for t in trades if t["reason"] == "stop"]
    targets = [t for t in trades if t["reason"] == "target"]
    wr   = len(winners) / n * 100
    aw   = sum(t["net"] for t in winners) / len(winners) if winners else 0
    al   = sum(t["net"] for t in losers)  / len(losers)  if losers  else 0
    rr   = abs(aw / al) if al != 0 else 0
    fees = sum(t["fee"] for t in trades)
    total = (final - STARTING_BAL) / STARTING_BAL * 100
    expect = (aw * (wr / 100)) + (al * ((100 - wr) / 100))
    return {
        "label": label, "trades": n, "win_rate": wr, "total_pct": total,
        "avg_win": aw, "avg_loss": al, "rr": rr, "fees": fees,
        "expect": expect, "btc_chg": btc_chg, "final": final,
        "stop_pct":   len(stops)   / n * 100,
        "target_pct": len(targets) / n * 100,
        "macd_t":  len([t for t in trades if t["etype"] == "macd"]),
        "sweep_t": len([t for t in trades if t["etype"] == "sweep"]),
        "pull_t":  len([t for t in trades if t["etype"] == "ema_pull"]),
    }

def print_report(summaries):
    sep = "═" * 108
    print(f"\n{sep}")
    print("  6-MONTH WINDOW VALIDATION — Strategy B (body 0.55, 4% risk)")
    print("  Goal: positive return in every 6-month window, identify weak regimes")
    print(sep)

    print(f"\n  {'WINDOW':<22} {'BTC%':>7} {'STRAT%':>9} {'TRADES':>7} "
          f"{'WIN%':>6} {'R/R':>5} {'EXPECT':>10} {'FEES':>9} {'AVG_L':>9} {'VERDICT':>8}")
    print("  " + "─" * 104)

    positive = 0
    fail_windows = []
    for s in summaries:
        passed = s["total_pct"] > 0
        if passed: positive += 1
        else: fail_windows.append(s["label"].strip())
        verdict  = "✓ PASS" if passed else "✗ FAIL"
        btc_str  = f"{s['btc_chg']:>+.0f}%"
        regime   = "📉" if s["btc_chg"] < -15 else ("📈" if s["btc_chg"] > 15 else "➡️ ")
        print(f"  {regime} {s['label']:<20} {btc_str:>7} {s['total_pct']:>+8.2f}% "
              f"{s['trades']:>7} {s['win_rate']:>5.1f}% {s['rr']:>5.2f}x "
              f"${s['expect']:>+8.2f} ${s['fees']:>7,.0f} ${abs(s['avg_loss']):>7,.0f} "
              f"{verdict:>8}")

    print(f"\n{sep}")
    print("  DETAILED BREAKDOWN")
    print(sep)
    for s in summaries:
        regime = "BEAR" if s["btc_chg"] < -15 else ("BULL" if s["btc_chg"] > 15 else "SIDEWAYS")
        print(f"\n  ▶ {s['label'].strip()}  [{regime}  BTC:{s['btc_chg']:>+.0f}%]")
        print(f"    Strategy return : {s['total_pct']:>+.2f}%  (${s['final'] - STARTING_BAL:>+,.2f})")
        print(f"    Trades          : {s['trades']}  "
              f"(MACD:{s['macd_t']} Pull:{s['pull_t']} Sweep:{s['sweep_t']})")
        print(f"    Win rate        : {s['win_rate']:.1f}%")
        print(f"    Avg winner      : ${s['avg_win']:>+,.2f}")
        print(f"    Avg loser       : ${s['avg_loss']:>+,.2f}")
        print(f"    R/R             : {s['rr']:.2f}x")
        print(f"    Expectancy      : ${s['expect']:>+,.2f} per trade")
        print(f"    Target hit      : {s['target_pct']:.1f}%")
        print(f"    Stop hit        : {s['stop_pct']:.1f}%")
        print(f"    Fees            : ${s['fees']:,.2f}")

    print(f"\n{sep}")
    print(f"  VERDICT: {positive}/{len(summaries)} windows positive")

    if positive == len(summaries):
        print("  ★★ PERFECT — Positive in every 6-month window across all regimes.")
        print("     Strategy is ready. Proceed to v12 for final tuning or paper trade.")
    elif positive >= 5:
        print(f"  ★  STRONG — {len(summaries)-positive} weak window(s): {', '.join(fail_windows)}")
        print("     Build v12 targeting those specific regime conditions.")
    else:
        print(f"  ~  MIXED — Fails in {len(summaries)-positive} windows.")
        print("     Significant regime sensitivity. v12 needs regime filter.")

    print(f"\n  REGIME SUMMARY:")
    bear_wins   = [s for s in summaries if s["btc_chg"] < -15 and s["total_pct"] > 0]
    bear_total  = [s for s in summaries if s["btc_chg"] < -15]
    bull_wins   = [s for s in summaries if s["btc_chg"] > 15  and s["total_pct"] > 0]
    bull_total  = [s for s in summaries if s["btc_chg"] > 15]
    side_wins   = [s for s in summaries if -15 <= s["btc_chg"] <= 15 and s["total_pct"] > 0]
    side_total  = [s for s in summaries if -15 <= s["btc_chg"] <= 15]

    print(f"    Bear markets : {len(bear_wins)}/{len(bear_total)} positive  "
          + (f"avg return: {sum(s['total_pct'] for s in bear_total)/len(bear_total):+.1f}%"
             if bear_total else "n/a"))
    print(f"    Bull markets : {len(bull_wins)}/{len(bull_total)} positive  "
          + (f"avg return: {sum(s['total_pct'] for s in bull_total)/len(bull_total):+.1f}%"
             if bull_total else "n/a"))
    print(f"    Sideways     : {len(side_wins)}/{len(side_total)} positive  "
          + (f"avg return: {sum(s['total_pct'] for s in side_total)/len(side_total):+.1f}%"
             if side_total else "n/a"))

    print(f"\n  WEAKEST WINDOWS (targets for v12):")
    sorted_s = sorted(summaries, key=lambda x: x["total_pct"])
    for s in sorted_s[:3]:
        regime = "BEAR" if s["btc_chg"] < -15 else ("BULL" if s["btc_chg"] > 15 else "SIDEWAYS")
        print(f"    {s['label'].strip():<22} {s['total_pct']:>+8.2f}%  "
              f"Regime:{regime}  BTC:{s['btc_chg']:>+.0f}%  "
              f"WR:{s['win_rate']:.0f}%  Trades:{s['trades']}")

    print(f"\n  v12 TARGETS (paste into next chat):")
    print(f"  ─────────────────────────────────────────────────────────────────")
    for s in summaries:
        regime = "BEAR" if s["btc_chg"] < -15 else ("BULL" if s["btc_chg"] > 15 else "SIDE")
        flag = " ← WEAK" if s["total_pct"] < 30 else ""
        print(f"  {s['label'].strip():<22} {s['total_pct']:>+8.2f}%  "
              f"{regime:<5}  WR:{s['win_rate']:.0f}%  "
              f"T:{s['trades']}  AvgL:${abs(s['avg_loss']):,.0f}{flag}")
    print(f"\n{sep}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not API_KEY or not SECRET_KEY:
        print("\nERROR: export ALPACA_API_KEY and ALPACA_SECRET_KEY\n"); exit(1)

    summaries = []
    for label, start, end in WINDOWS:
        print(f"\nFetching {label.strip()}...")
        bars = fetch_window(label, start, end)
        if not bars: continue

        window_bars = [b for b in bars if b["t"] >= start]
        btc_s = window_bars[0]["c"]  if window_bars else 0
        btc_e = window_bars[-1]["c"] if window_bars else 0

        print(f"  Running Strategy B...")
        result = backtest(bars)
        summaries.append(summarize(label, result, btc_s, btc_e))

    if summaries:
        print_report(summaries)
