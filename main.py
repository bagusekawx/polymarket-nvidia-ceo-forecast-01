#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 4 hours ahead.
Version 2.0 — persistence + jump-robust per-market sigma + sum-to-one median,
with a narrowed interquartile range and a horizon-decaying tail multiplier.
v2.0: IQR z +-0.50, tail multiplier 1 + 1/sqrt(h) on the outer quantiles.

Method (v2): v1's persistence median with per-market jump-robust volatility
and the sum-to-one median correction, plus two shape changes motivated by
the practice diagnostics (30,720 scored items):

1. Narrowed IQR: Polymarket prices move in steps — most minutes they do
   not move at all — so the true 25%/75% quantiles sit much closer to the
   median than a normal's +-0.674 sigma. The pooled standardized residuals
   put them near +-0.35/+-0.17; v2 uses a conservative symmetric +-0.50.
2. Horizon-decaying tail: jump risk is concentrated at short horizons
   (the h1-30 band's 5% tail sits near -2.8 sigma, the h121-240 band's
   near -1.5). The 5%/95% z-values get a multiplier 1 + 1/sqrt(h), wide
   at h=1 and ~1.06 at h=240, instead of a flat widening that would pay
   at every quiet long-horizon minute.

Per market, from the last 720 one-minute moves:
  sigma = 1.2533 * mean(|move|) after winsorizing |moves| at the 99th
  percentile (jumps inflate a raw std, so they are clipped first),
  floored at 0.0005, capped at 0.01, then scaled by 0.7.
Quantiles are median + z_tau * sigma * sqrt(h), clamped to [0, 1],
rounded to 6 decimals. Median = last price + the sum correction ramping
linearly in h/240. No randomness: two runs on the same case are identical.

Local practice (6 cases, real Polymarket actuals): weighted pinball loss
0.261309 vs v1's 0.262984 and the house baseline's 0.363589.
"""
import json
import math
import os

IN_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Narrowed interquartile range: +-0.50 instead of the normal's +-0.674490.
Z = [-1.644854, -0.50, 0.0, 0.50, 1.644854]
# Tail multiplier 1 + TAIL_DECAY / sqrt(h) on the 5%/95% z-values only.
TAIL_DECAY = 1.0

HISTORY_MINUTES = 720
WINSOR_Q = 0.99
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.01
SIGMA_SCALE = 0.7
MEAN_ABS_TO_SIGMA = 1.253314  # E|X| = sigma * sqrt(2/pi) for a normal


def clip6(value):
    """Clamp to [0, 1], round to 6 decimals, never -0.0."""
    rounded = round(min(1.0, max(0.0, value)) * 1e6) / 1e6
    return rounded + 0.0


def robust_sigma(prices):
    """Jump-robust per-minute volatility from the recent moves."""
    tail = prices[-(HISTORY_MINUTES + 1):]
    steps = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
    if not steps:
        return SIGMA_FLOOR
    mags = sorted(abs(s) for s in steps)
    cap = mags[min(len(mags) - 1, int(WINSOR_Q * len(mags)))]
    clipped = [max(-cap, min(cap, s)) for s in steps]
    sigma = MEAN_ABS_TO_SIGMA * sum(abs(s) for s in clipped) / len(clipped)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma)) * SIGMA_SCALE


def build_rows(spot, sigma, correction, horizon):
    """240 rows of five quantiles for one market."""
    rows = []
    for h in range(1, horizon + 1):
        centre = spot + correction * (h / 240.0)
        width = sigma * math.sqrt(h)
        # Jump risk is front-loaded: the outer quantiles widen most at
        # short horizons and the multiplier decays toward 1 as h grows.
        outer = 1.0 + TAIL_DECAY / math.sqrt(h)
        rows.append([
            clip6(centre + Z[0] * outer * width),
            clip6(centre + Z[1] * width),
            clip6(centre),
            clip6(centre + Z[3] * width),
            clip6(centre + Z[4] * outer * width),
        ])
    return rows


def forecast_case(case):
    horizon = case["horizon_minutes"]
    markets = case["markets"]
    spots = []
    sigmas = []
    for market in markets:
        series = market["prices"]
        if not series:
            raise ValueError("market %s has no prices" % market["id"])
        spots.append(series[-1])
        sigmas.append(robust_sigma(series))
    total = sum(spots)
    forecasts = []
    for i, market in enumerate(markets):
        # Proportional pull toward a unit sum across the mutually
        # exclusive outcomes; zero when the checkpoint already sums to 1.
        correction = (1.0 / total - 1.0) * spots[i] if total > 0 else 0.0
        forecasts.append({
            "id": market["id"],
            "q": build_rows(spots[i], sigmas[i], correction, horizon),
        })
    return {"forecasts": forecasts}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(IN_DIR) if f.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(IN_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        if list(case["quantiles"]) != LEVELS:
            raise ValueError("%s asks for quantiles %r" % (name, case["quantiles"]))
        out = forecast_case(case)
        with open(os.path.join(OUT_DIR, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
