#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 4 hours ahead.
Version 1.0 — persistence + jump-robust per-market sigma + sum-to-one median.

Method (v1): persistence with per-market jump-robust volatility plus a
sum-to-one median correction. The 24 outcomes are mutually exclusive, so
their true probabilities sum to 1; when the checkpoint sum differs from 1
the medians drift proportionally back toward a unit sum over the horizon.

Per market, from the last 720 one-minute moves:
  sigma = 1.2533 * mean(|move|) after winsorizing |moves| at the 99th
  percentile (jumps inflate a raw std, so they are clipped first),
  floored at 0.0005, capped at 0.01, then scaled by 0.7.
Quantiles are normal: median + z_tau * sigma * sqrt(h), clamped to [0, 1],
rounded to 6 decimals. Median = last price + the sum correction ramping
linearly in h/240. No randomness: two runs on the same case are identical.

Local practice (6 cases, real Polymarket actuals): weighted pinball loss
0.262984 vs the house baseline's 0.363589; every case beats the baseline.
"""
import json
import math
import os

IN_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

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
        rows.append([
            clip6(centre + Z[0] * width),
            clip6(centre + Z[1] * width),
            clip6(centre),
            clip6(centre + Z[3] * width),
            clip6(centre + Z[4] * width),
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
