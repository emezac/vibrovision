#!/usr/bin/env python3
"""Compare benchmark JSON files and compute cost per million ROI-frames.

Prices are an *input* (USD/hour, on-demand, your region, checked on the day you
run it) - they are not hard-coded so the table cannot silently go stale:

    python bench/compare.py results/bench_*.json \
        --price c7i-pip=0.0893 --price c7g-pip=0.0725 --price c7g-cool=0.0725+0.0XX

For COOL, add the Marketplace software fee to the instance price.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--price", action="append", default=[], help="label=USD_per_hour (expressions like 0.07+0.01 ok)")
    a = ap.parse_args()
    prices = {}
    for p in a.price:
        k, v = p.split("=", 1)
        prices[k] = float(sum(float(x) for x in v.split("+")))
    rows = [json.load(open(f)) for f in a.files]
    base = rows[0]
    prims = list(base["primitives"])
    print(f"{'label':14s} {'instance':12s} {'cv2':10s} {'ROI-frames/s':>13s} {'x base':>7s} {'USD/1M ROI-fr':>14s}")
    for r in rows:
        fps = r["end_to_end"]["roi_frames_per_s"]
        price = prices.get(r["label"])
        cost = f"{price / (fps * 3600) * 1e6:.4f}" if price else "-"
        print(f"{r['label']:14s} {r['instance_type'] or r['machine']:12s} {r['opencv_version']:10s} "
              f"{fps:13.0f} {fps / base['end_to_end']['roi_frames_per_s']:7.2f} {cost:>14s}")
    print("\nmedian microseconds per call")
    print(f"{'primitive':30s} " + " ".join(f"{r['label']:>12s}" for r in rows))
    for p in prims:
        print(f"{p:30s} " + " ".join(f"{r['primitives'].get(p, {}).get('median_us', float('nan')):12.1f}" for r in rows))


if __name__ == "__main__":
    main()
