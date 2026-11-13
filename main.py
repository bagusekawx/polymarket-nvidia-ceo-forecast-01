"""Polymarket "Next CEO of NVIDIA?" — Yes-price quantiles, 4 hours ahead.
Version 3.0 — asymmetric IQR (-0.75/+0.15), recency-blended volatility
(65% x 720-min + 35% x 60-min), downside-only tail decay (1 + 2/sqrt(h)).
Version 3.0: asymmetric interquartile range, recency-blended volatility,
and a downside-only horizon-decaying tail multiplier.

Method (v3 = v2 + three changes, all validated on the six practice cases
against real Polymarket actuals; weighted pinball loss 0.251062 vs v2's
0.261309 and the house baseline's 0.363589):

1. Asymmetric IQR. Prices move in steps, and the pooled standardized
   residuals are lopsided: the 25% quantile sits near the normal value
   while the 75% quantile hugs the median (the next move up is usually
   nothing at all). v3 uses z = -0.75 below the median and +0.15 above
   it, instead of v2's symmetric +-0.50. The optimum is a shallow
   plateau (-0.70 to -0.80 below, +0.10 to +0.20 above), not a spike.
2. Recency-blended volatility. Volatility clusters: the last hour's
   moves predict the next hours' spread better than the 12-hour
   average alone. v3 blends 65% of the 720-minute jump-robust sigma
   with 35% of the same estimator on the last 60 minutes.
3. One-sided tail decay. The downside tail is fat at short horizons
   and thins out; the upside tail is thin early and widening it costs
   at every quiet minute. v3 keeps the decaying multiplier (1 + 2/sqrt(h))
   on the 5% quantile only and leaves the 95% quantile flat.

Unchanged from v2: persistence median (last price) with the sum-to-one
median correction (the 24 outcomes are mutually exclusive, so medians
drift proportionally toward a unit sum, ramping linearly over h/240);
per-market jump-robust sigma (1.2533 * mean |move|, winsorized at the
99th percentile, floored 0.0005, capped 0.01, scaled 0.7); quantiles
median + z * sigma * sqrt(h), clamped to [0, 1], rounded to 6 decimals.
No randomness: two runs on the same case are byte-identical.
"""
import json, math, os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Asymmetric interquartile range (see docstring): wide below, tight above.
Z_Q05 = -1.644854
Z_Q25 = -0.75
Z_Q75 = 0.15
Z_Q95 = 1.644854
# Downside tail multiplier 1 + TAIL_LO_DECAY / sqrt(h); upside stays flat.
TAIL_LO_DECAY = 2.0
TAIL_HI_DECAY = 0.0

LONG_WINDOW = 720      # minutes for the base volatility estimate
SHORT_WINDOW = 60      # minutes for the recency estimate
SHORT_WEIGHT = 0.35    # weight on the recency estimate
WINSOR_Q = 0.99
VOL_FLOOR = 0.0005
VOL_CAP = 0.01
VOL_SCALE = 0.7
MEAN_ABS_TO_SIGMA = 1.253314  # E|X| = sigma * sqrt(2/pi) for a normal


def clamp6(x):
    """Clamp to [0, 1], round to 6 decimals, never -0.0."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def window_vol(prices, minutes):
    """Jump-robust per-minute volatility from the last `minutes` of moves."""
    tail = prices[-(minutes + 1):]
    moves = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
    if len(moves) == 0:
        return VOL_FLOOR * VOL_SCALE
    mags = sorted(abs(d) for d in moves)
    idx = int(WINSOR_Q * len(mags))
    cap = mags[idx if idx < len(mags) else len(mags) - 1]
    clipped = [max(-cap, min(cap, d)) for d in moves]
    vol = MEAN_ABS_TO_SIGMA * sum(abs(d) for d in clipped) / len(clipped)
    return min(VOL_CAP, max(VOL_FLOOR, vol)) * VOL_SCALE


def market_vol(prices):
    """Blend long-window and recent volatility; volatility clusters."""
    long_vol = window_vol(prices, LONG_WINDOW)
    short_vol = window_vol(prices, SHORT_WINDOW)
    return (1.0 - SHORT_WEIGHT) * long_vol + SHORT_WEIGHT * short_vol


def forecast_market(px, vol, pull, horizon):
    """The 240 rows of five quantiles for one market."""
    rows = []
    for h in range(1, horizon + 1):
        centre = px + pull * (h / 240.0)
        width = vol * math.sqrt(h)
        down = 1.0 + TAIL_LO_DECAY / math.sqrt(h)
        up = 1.0 + TAIL_HI_DECAY / math.sqrt(h)
        rows.append([
            clamp6(centre + Z_Q05 * down * width),
            clamp6(centre + Z_Q25 * width),
            clamp6(centre),
            clamp6(centre + Z_Q75 * width),
            clamp6(centre + Z_Q95 * up * width),
        ])
    return rows


def forecast_case(case):
    horizon = int(case["horizon_minutes"])
    levels = []
    vols = []
    for market in case["markets"]:
        series = market["prices"]
        if len(series) == 0:
            raise ValueError("no prices for market %s" % market["id"])
        levels.append(series[-1])
        vols.append(market_vol(series))
    total = sum(levels)
    forecasts = []
    for i, market in enumerate(case["markets"]):
        # Proportional pull toward a unit sum across the mutually exclusive
        # outcomes; zero when the checkpoint already sums to 1.
        pull = (1.0 / total - 1.0) * levels[i] if total > 0 else 0.0
        forecasts.append({"id": market["id"],
                          "q": forecast_market(levels[i], vols[i], pull, horizon)})
    return {"forecasts": forecasts}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in names:
        case_id = name[:-len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.loads(fh.read())
        if list(case["quantiles"]) != list(QUANTILE_LEVELS):
            raise ValueError("unexpected quantiles in %s: %r" % (name, case["quantiles"]))
        result = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
