"""Measure realised per-cell wall clock per (method, suite) against c2_slot_plan's assumptions."""

import glob
import json
import statistics as st
from collections import defaultdict

ROOT = "/mnt/home/users/tic_163_uma/mpascual/fscratch/results/isalsr/c2_3arm"

ASSUMED = {
    ("udfs", "nguyen"): 11.5,
    ("udfs", "feynman"): 0.9,
    ("udfs", "hard"): 12.0,
    ("udfs", "cherrypicked"): 12.0,
    ("udfs", "roundoff"): 12.0,
    ("udfs", "feynman_remainder"): 6.0,
    ("udfs", "strogatz"): 11.0,
    ("bingo", "nguyen"): 0.15,
    ("bingo", "feynman"): 0.20,
    ("bingo", "hard"): 5.3,
    ("bingo", "cherrypicked"): 7.5,
    ("bingo", "roundoff"): 5.0,
    ("bingo", "feynman_remainder"): 2.0,
    ("bingo", "strogatz"): 2.0,
}

hours = defaultdict(list)
for path in glob.glob(ROOT + "/*/*/*/*/*/run_log.json"):
    parts = path.split("/")
    key = (parts[-6], parts[-5])
    try:
        blob = json.load(open(path))
    except Exception:
        continue
    res = blob.get("results") or {}
    tim = res.get("time") if isinstance(res.get("time"), dict) else {}
    wall = tim.get("wall_clock_total_s")
    if wall:
        hours[key].append(float(wall) / 3600.0)

print("%-26s %6s %9s %8s %9s %7s" % ("method:suite", "n", "median_h", "p90_h", "assumed", "ratio"))
for key in sorted(hours):
    vals = sorted(hours[key])
    med = st.median(vals)
    p90 = vals[max(0, int(0.9 * len(vals)) - 1)]
    exp = ASSUMED.get(key, 0.0)
    ratio = (med / exp) if exp else float("nan")
    print("%-26s %6d %9.2f %8.2f %9.2f %6.1fx" % (":".join(key), len(vals), med, p90, exp, ratio))
