"""結果JSONのplan_js/plan_poseをシーンごとに要約する(開発中の確認用)。使い方: scene_stats.py results/xxx.json [plan_js|plan_pose]"""
import collections
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
key = sys.argv[2] if len(sys.argv) > 2 else "plan_js"
by = collections.defaultdict(list)
fail = collections.Counter()
for r in d[key]:
    if r.get("success"):
        by[r["scene"]].append(r["time_ms"])
    else:
        fail[r["scene"]] += 1
allt = []
print(f"{'scene':20s} {'n':>4s} {'fail':>4s} {'med':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s}")
for k, v in by.items():
    allt += v
    print(f"{k:20s} {len(v):4d} {fail[k]:4d} {np.median(v):7.2f} {np.percentile(v, 95):7.2f} {np.percentile(v, 99):7.2f} {max(v):7.2f}")
print(f"{'ALL':20s} {len(allt):4d} {sum(fail.values()):4d} {np.median(allt):7.2f} {np.percentile(allt, 95):7.2f} {np.percentile(allt, 99):7.2f} {max(allt):7.2f}")
