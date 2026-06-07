"""
BTC 4-Hour MACD Optimizer v11 — LONGS ONLY
===========================================
v10 Analysis:

CONFIRMED REAL LEVERS:
  1. MACD body 0.55: +173.95%, 19 trades, 63.2% WR, $28,999 fees ★★
     - Filters 2 bad MACD entries, keeps all good ones
     - Single best change across all versions tested

  2. Time stop 18 bars: +168.31%, 21 trades, 57.1% WR, avg loser $5,303
     - Cuts dead trades before full stop-out
     - R/R jumped to 3.39x — losers are smaller, winners unchanged

DEAD LEVERS (removed from v11):
  - Cooldown after loss: zero effect, no re-entries within cooldown window
  - Min R/R 2.0/2.5: redundant, all trades already above 2.0 with body filter
  - Require 50 EMA all entries: too strict, kills 10 good trades
  - Full quality combine: over-filtering, removed too many good trades

v11 BETS:
  1. Combine body 0.55 + time stop 18 — are they additive?
  2. Combine body 0.55 + time stop 12 — tighter time stop with quality filter
  3. Time stop tuning: 15, 20, 24 bars — find the sweet spot
  4. Body + time stop + risk sizing: 3%, 4%, 5%
  5. Pull/sweep body filter: require 0.50 body on pull/sweep too (not just MACD)
  6. Time stop with tighter exit: move stop to 0.25R above entry (not just entry)
  7. Combine all confirmed winners at multiple risk levels

Baseline to beat: MACD body 0.55 — +173.95%, 19 trades, 63.2% WR, $28,999 fees
"""

import os
from datetime import datetime, timezone, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

SYMBOL       = "BTC/USD"
STARTING_BAL = 100000.0
FEE_PCT      = 0.0015
MONTHS_BACK  = 12
MAX_HOLD     = 84

API_KEY    = os.environ.get("ALPACA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")


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
def body(b):      return abs(b["c"] - b["o"])
def rng(b):       return max(b["h"] - b["l"], 0.0001)
def strong(b, p): return body(b) / rng(b) >= p


# ─────────────────────────────────────────────
# STRATEGY FACTORY v11
# ─────────────────────────────────────────────
def make_long_strategy(
    rsi_max          = 55,
    target_atr       = 7.0,
    risk_pct         = 0.040,
    trail_pct        = 0.010,
    vol_atr_mult     = 1.2,
    sweep_stop_atr   = 1.0,
    pull_stop_atr    = 1.0,
    macd_stop_atr    = 1.3,
    macd_body_min    = 0.55,   # v10 confirmed: 0.55 is the sweet spot
    pull_body_min    = 0.40,   # NEW: body filter for pull entries
    sweep_body_min   = 0.40,   # NEW: body filter for sweep entries
    add_sweep        = True,
    add_ema_pull     = True,
):
    def strategy(bars):
        if len(bars) < 80: return None
        current = bars[-1]["c"]
        a = atr(bars, 14)
        if a == 0: return None

        e20  = ema(bars, 20)
        e50  = ema(bars, 50)
        e200 = ema(bars, 200) if len(bars) >= 200 else e50
        r    = rsi(bars, 14)

        avg_a = atr_avg(bars, 14, 20)
        if a > vol_atr_mult * avg_a:
            return None

        # ── MACD CROSSOVER ──────────────────────────────────────────────────
        ml, ml_p, ms, ms_p = macd_vals(bars, 12, 26, 9)
        macd_ok = True
        if current < e200 * 0.998:              macd_ok = False
        if r > rsi_max:                         macd_ok = False
        if not (ml > ms and ml_p <= ms_p):      macd_ok = False
        if not is_green(bars[-1]):              macd_ok = False
        if not strong(bars[-1], macd_body_min): macd_ok = False
        if not ((ml - ms) > (ml_p - ms_p)):     macd_ok = False
        if macd_ok:
            stop   = current - macd_stop_atr * a
            target = current + target_atr * a
            risk = current - stop
            if risk > 0 and (target - current) / risk >= 1.5:
                return ("long", current, stop, target, risk_pct, trail_pct,
                        True, "macd", a)

        # ── EMA PULLBACK ────────────────────────────────────────────────────
        if add_ema_pull:
            pull_ok = True
            if current < e50:                               pull_ok = False
            if current < e20 * 0.998 or current > e20 * 1.008: pull_ok = False
            if r > 60:                                      pull_ok = False
            if not is_green(bars[-1]):                      pull_ok = False
            if not strong(bars[-1], pull_body_min):         pull_ok = False
            if not is_green(bars[-2]):                      pull_ok = False
            if e20 < e50:                                   pull_ok = False
            if pull_ok:
                stop   = current - pull_stop_atr * a
                target = current + target_atr * a
                risk = current - stop
                if risk > 0 and (target - current) / risk >= 1.5:
                    return ("long", current, stop, target, risk_pct, trail_pct,
                            True, "ema_pull", a)

        # ── LIQUIDITY SWEEP ─────────────────────────────────────────────────
        if add_sweep:
            lows = swing_lows(bars[:-3], 3, w=4)
            if lows:
                support = sum(lows) / len(lows)
                swept = any(bars[-i]["l"] < support * (1 - 0.002) for i in range(2, 6))
                if swept and current > support and r < 60:
                    if is_green(bars[-1]) and strong(bars[-1], sweep_body_min):
                        if is_green(bars[-2]):
                            stop   = current - sweep_stop_atr * a
                            target = current + target_atr * a
                            risk = current - stop
                            if risk > 0 and (target - current) / risk >= 1.5:
                                return ("long", current, stop, target, risk_pct,
                                        trail_pct, True, "sweep", a)
        return None
    return strategy, True


# ─────────────────────────────────────────────
# VARIANTS v11
# ─────────────────────────────────────────────
# Baseline: MACD body 0.55 — +173.95%, 19 trades, 63.2% WR, $28,999 fees
#
# Columns:
# Name, rsi_max, tgt_atr, risk, trail, vol_mult,
# sweep_stop, pull_stop, macd_stop,
# macd_body, pull_body, sweep_body,
# time_stop_bars, time_stop_r,   ← how flat before acting, how far to move stop
# add_sweep, add_pull

VARIANTS = [
    # ── BASELINE: v10 winner ────────────────────────────────────────────────
    ("v10 winner body 0.55 4%       ",
     55, 7.0, 0.040, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     0, 0.0,
     True, True),

    # ── LEVER 1: COMBINE BODY 0.55 + TIME STOP ─────────────────────────────
    ("Body 0.55 + time stop 18      ",
     55, 7.0, 0.040, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     18, 0.5,
     True, True),

    ("Body 0.55 + time stop 12      ",
     55, 7.0, 0.040, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     12, 0.5,
     True, True),

    ("Body 0.55 + time stop 24      ",
     55, 7.0, 0.040, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     24, 0.5,
     True, True),

    # ── LEVER 2: TIME STOP TUNING (on baseline body 0.55) ──────────────────
    # v10 showed 18 > 12. Where is the real peak?
    ("Body 0.55 + time stop 15      ",
     55, 7.0, 0.040, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     15, 0.5,
     True, True),

    ("Body 0.55 + time stop 20      ",
     55, 7.0, 0.040, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     20, 0.5,
     True, True),

    # ── LEVER 3: TIGHTER TIME STOP EXIT (move to 0.5R above entry) ─────────
    # Instead of moving stop to entry, move it to entry + 0.5R
    # Locks in a small profit on dead trades instead of just breakeven
    ("Body 0.55 + time stop 18 +0.5R",
     55, 7.0, 0.040, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     18, 1.0,   # time_stop_r=1.0 means move stop to entry + 0.5*init_risk
     True, True),

    # ── LEVER 4: PULL/SWEEP BODY FILTER ────────────────────────────────────
    # Require stronger candle on pull and sweep entries too
    ("Body 0.55 all entries         ",
     55, 7.0, 0.040, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.55, 0.55,
     0, 0.0,
     True, True),

    ("Body 0.50 pull+sweep          ",
     55, 7.0, 0.040, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.50, 0.50,
     0, 0.0,
     True, True),

    # ── LEVER 5: RISK SIZING ON WINNER ─────────────────────────────────────
    ("Body 0.55 3% risk             ",
     55, 7.0, 0.030, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     0, 0.0,
     True, True),

    ("Body 0.55 5% risk             ",
     55, 7.0, 0.050, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     0, 0.0,
     True, True),

    # ── LEVER 6: TIGHTER TRAIL ON WINNER ───────────────────────────────────
    # Trail activates at 2R — does tighter trail capture more?
    ("Body 0.55 trail 0.5%          ",
     55, 7.0, 0.040, 0.005, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     0, 0.0,
     True, True),

    ("Body 0.55 trail 0.7%          ",
     55, 7.0, 0.040, 0.007, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     0, 0.0,
     True, True),

    # ── FULL COMBINE: BODY 0.55 + TIME STOP 18 + BEST RISK ─────────────────
    ("FULL COMBINE body+ts18 3%     ",
     55, 7.0, 0.030, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     18, 0.5,
     True, True),

    ("FULL COMBINE body+ts18 4%     ",
     55, 7.0, 0.040, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     18, 0.5,
     True, True),

    ("FULL COMBINE body+ts18 5%     ",
     55, 7.0, 0.050, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.40, 0.40,
     18, 0.5,
     True, True),

    # ── FULL COMBINE + PULL/SWEEP BODY 0.50 ────────────────────────────────
    ("FULL body0.55+ts18+pull0.50 4%",
     55, 7.0, 0.040, 0.010, 1.2,
     1.0, 1.0, 1.3,
     0.55, 0.50, 0.50,
     18, 0.5,
     True, True),
]

STRATEGIES = []
for v in VARIANTS:
    name = v[0]
    fn, cmp = make_long_strategy(
        rsi_max=v[1], target_atr=v[2], risk_pct=v[3], trail_pct=v[4],
        vol_atr_mult=v[5], sweep_stop_atr=v[6], pull_stop_atr=v[7],
        macd_stop_atr=v[8], macd_body_min=v[9],
        pull_body_min=v[10], sweep_body_min=v[11],
        add_sweep=v[14], add_ema_pull=v[15],
    )
    STRATEGIES.append((name, fn, cmp, v[3], v[12], v[13]))


# ─────────────────────────────────────────────
# BACKTESTER v11
# time_stop_bars: if trade < 0.5R move after N candles, move stop up
# time_stop_r:    0.5 = move to entry, 1.0 = move to entry + 0.5*init_risk
# ─────────────────────────────────────────────
def backtest(all_bars, signal_fn, use_compound=True, risk_pct=0.04,
             time_stop_bars=0, time_stop_r=0.5):
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

            # Time stop: if trade hasn't moved enough, tighten stop
            if time_stop_bars > 0 and init_risk > 0 and held == time_stop_bars:
                move = current - cost
                if move < 0.5 * init_risk:
                    new_stop = cost + (time_stop_r - 0.5) * init_risk
                    if new_stop > stop:
                        open_trade["stop"] = new_stop
                        stop = new_stop

            # Trailing stop: activates after 2R
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
                sizing_bal = balance if use_compound else STARTING_BAL
                qty   = (sizing_bal * risk_pct) / init_risk if init_risk > 0 else 1
                gross = (ep - cost) * qty
                fee   = (cost * qty * FEE_PCT) + (ep * qty * FEE_PCT)
                net   = gross - fee
                balance += net
                trades.append({
                    "net": net, "win": net > 0, "reason": er,
                    "fee": fee, "hold": held,
                    "etype": open_trade.get("etype", "macd"),
                    "pnl_pct": (ep - cost) / cost * 100,
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
# RESULTS
# ─────────────────────────────────────────────
def summarize(name, result):
    trades = result["trades"]; final = result["final"]; n = len(trades)
    if n == 0:
        return {"name": name, "trades": 0, "win_rate": 0, "total_pct": 0,
                "avg_win": 0, "avg_loss": 0, "rr": 0, "fees": 0, "final": final,
                "target_pct": 0, "stop_pct": 0, "expect": 0, "avg_hold": 0,
                "macd_t": 0, "sweep_t": 0, "pull_t": 0}
    winners = [t for t in trades if t["win"]]
    losers  = [t for t in trades if not t["win"]]
    targets = [t for t in trades if t["reason"] == "target"]
    stops   = [t for t in trades if t["reason"] == "stop"]
    wr   = len(winners) / n * 100
    aw   = sum(t["net"] for t in winners) / len(winners) if winners else 0
    al   = sum(t["net"] for t in losers)  / len(losers)  if losers  else 0
    rr   = abs(aw / al) if al != 0 else 0
    fees = sum(t["fee"] for t in trades)
    total = (final - STARTING_BAL) / STARTING_BAL * 100
    expect = (aw * (wr / 100)) + (al * ((100 - wr) / 100))
    avg_hold = sum(t["hold"] for t in trades) / n * 4
    return {
        "name": name, "trades": n, "win_rate": wr, "total_pct": total,
        "avg_win": aw, "avg_loss": al, "rr": rr, "fees": fees, "final": final,
        "target_pct": len(targets) / n * 100,
        "stop_pct":   len(stops)   / n * 100,
        "expect": expect, "avg_hold": avg_hold,
        "macd_t":  len([t for t in trades if t["etype"] == "macd"]),
        "sweep_t": len([t for t in trades if t["etype"] == "sweep"]),
        "pull_t":  len([t for t in trades if t["etype"] == "ema_pull"]),
    }

def print_results(summaries):
    summaries.sort(key=lambda x: x["total_pct"], reverse=True)
    sep = "═" * 100
    BASELINE_RET    = 173.95
    BASELINE_TRADES = 19
    BASELINE_FEES   = 28999
    BASELINE_WR     = 63.2

    print(f"\n{sep}")
    print("  BTC 4H LONGS ONLY v11 — 12 MONTHS")
    print(f"  Baseline: v10 MACD body 0.55 — +{BASELINE_RET}%, "
          f"{BASELINE_TRADES} trades, ${BASELINE_FEES:,} fees, {BASELINE_WR}% WR")
    print(sep)
    print(f"\n  {'#':<4} {'VARIANT':<36} {'RETURN':>9} {'TRADES':>7} "
          f"{'WIN%':>6} {'R/R':>5} {'EXPECT':>9} {'FEES':>8}")
    print("  " + "─" * 96)
    for i, s in enumerate(summaries):
        beat_all = (s["total_pct"] > BASELINE_RET and
                    s["trades"] <= BASELINE_TRADES and
                    s["win_rate"] >= BASELINE_WR)
        beat_ret = s["total_pct"] > BASELINE_RET
        marker = " ◀" if i == 0 else (" ★★" if beat_all else (" ★" if beat_ret else ""))
        print(f"  {i+1:<4} {s['name']:<36} {s['total_pct']:>+8.2f}% "
              f"{s['trades']:>7} {s['win_rate']:>5.1f}% {s['rr']:>5.2f}x "
              f"${s['expect']:>+7.2f} ${s['fees']:>6,.0f}{marker}")

    print(f"\n{sep}")
    print("  DETAILED — TOP 3")
    print(sep)
    for s in summaries[:3]:
        print(f"\n  ▶ {s['name'].strip()}")
        print(f"    Return       : {s['total_pct']:>+.2f}%  (${s['final'] - STARTING_BAL:>+,.2f})")
        print(f"    Trades       : {s['trades']}  "
              f"(MACD:{s['macd_t']} Pull:{s['pull_t']} Sweep:{s['sweep_t']})")
        print(f"    Win rate     : {s['win_rate']:.1f}%")
        print(f"    Avg winner   : ${s['avg_win']:>+,.2f}")
        print(f"    Avg loser    : ${s['avg_loss']:>+,.2f}")
        print(f"    R/R          : {s['rr']:.2f}x")
        print(f"    Expectancy   : ${s['expect']:>+,.2f} per trade")
        print(f"    Avg hold     : {s['avg_hold']:.0f} hrs ({s['avg_hold']/24:.1f} days)")
        print(f"    Target hit   : {s['target_pct']:.1f}%")
        print(f"    Stop hit     : {s['stop_pct']:.1f}%")
        print(f"    Fees         : ${s['fees']:,.2f}")

    best = summaries[0]
    print(f"\n{sep}\n  VERDICT vs v10 baseline (+{BASELINE_RET}%, {BASELINE_TRADES} trades)")
    ret_d   = best["total_pct"] - BASELINE_RET
    trade_d = best["trades"] - BASELINE_TRADES
    fee_d   = best["fees"] - BASELINE_FEES
    print(f"  Return delta : {ret_d:>+.2f}%")
    print(f"  Trade delta  : {trade_d:>+d} ({'fewer' if trade_d < 0 else 'more'})")
    print(f"  Fee delta    : ${fee_d:>+,.0f} ({'saved' if fee_d < 0 else 'extra'})")
    print(f"  WR delta     : {best['win_rate'] - BASELINE_WR:>+.1f}%")

    if best["total_pct"] > BASELINE_RET and best["trades"] <= BASELINE_TRADES:
        print(f"\n  ★★ IDEAL — Better return, same or fewer trades.")
        print(f"     Run multi-window validation on this config.")
    elif best["total_pct"] > BASELINE_RET:
        print(f"\n  ★  Higher return but more trades. Worth it if fees justify.")
    else:
        print(f"\n  ~  No improvement. v10 winner (body 0.55) remains the champion.")
        print(f"     May have found the ceiling for this strategy on this window.")
        print(f"     Recommend: run multi-window validation on v10 winner now.")

    print(f"\n  TIME STOP ANALYSIS:")
    ts_vars = [(s["name"].strip(), s["total_pct"], s["trades"],
                s["win_rate"], abs(s["avg_loss"]))
               for s in summaries if "time stop" in s["name"] or "ts" in s["name"]]
    for t in sorted(ts_vars, key=lambda x: x[1], reverse=True):
        print(f"    {t[0]:<38} {t[1]:>+7.2f}%  T:{t[2]}  WR:{t[3]:.0f}%  AvgL:${t[4]:,.0f}")

    print(f"\n{sep}\n")
    print("  v11 HANDOFF NOTES:")
    print("  ─────────────────────────────────────────────────────────────────")
    for s in summaries[:5]:
        print(f"  [{s['total_pct']:>+8.2f}%] {s['name'].strip():<36} "
              f"WR:{s['win_rate']:.0f}% T:{s['trades']} "
              f"Fees:${s['fees']:,.0f} AvgL:${abs(s['avg_loss']):,.0f}")
    print(f"\n{sep}\n")


# ─────────────────────────────────────────────
# FETCH + MAIN
# ─────────────────────────────────────────────
def fetch_bars():
    print(f"\nFetching {MONTHS_BACK} months of BTC 4h candles...")
    client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=MONTHS_BACK * 30)
    req = CryptoBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(4, TimeFrameUnit.Hour),
        start=start, end=end)
    df = client.get_crypto_bars(req).df
    if df.empty: return []
    if hasattr(df.index, "levels"):
        sym = SYMBOL.replace("/", "")
        if sym in df.index.get_level_values(0):   df = df.loc[sym]
        elif SYMBOL in df.index.get_level_values(0): df = df.loc[SYMBOL]
    bars = []
    for _, row in df.iterrows():
        bars.append({"o": float(row["open"]),  "h": float(row["high"]),
                     "l": float(row["low"]),   "c": float(row["close"]),
                     "v": float(row["volume"])})
    print(f"Loaded {len(bars):,} candles — {len(bars)*4/24:.0f} days\n")
    return bars

if __name__ == "__main__":
    if not API_KEY or not SECRET_KEY:
        print("\nERROR: export ALPACA_API_KEY and ALPACA_SECRET_KEY\n"); exit(1)
    all_bars = fetch_bars()
    if not all_bars: exit(1)
    summaries = []
    for name, fn, compound, risk_pct, ts_bars, ts_r in STRATEGIES:
        print(f"  Testing {name.strip()}...")
        result = backtest(all_bars, fn, compound, risk_pct,
                          time_stop_bars=ts_bars, time_stop_r=ts_r)
        summaries.append(summarize(name, result))
    print_results(summaries)
