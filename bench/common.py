"""3つのrunner(vamp / curobo / vkmotion)で共通のI/O。結果JSONのスキーマはここで固定する。"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"


def load_problems(limit=None, scenes=None):
    d = json.loads((DATA / "problems.json").read_text())
    probs = d["problems"]
    if scenes:
        probs = [p for p in probs if p["scene"] in scenes]
    if limit:
        # シーンごとに先頭limit件(シーン間のバランスを保つ)
        per, out = {}, []
        for p in probs:
            per[p["scene"]] = per.get(p["scene"], 0) + 1
            if per[p["scene"]] <= limit:
                out.append(p)
        probs = out
    return probs


def load_samples():
    return json.loads((DATA / "samples.json").read_text())


def path_length(path):
    p = np.asarray(path, dtype=float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def timer():
    return time.perf_counter()


def save(lib, data):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{lib}.json").write_text(json.dumps(data))
    print(f"saved {RESULTS / (lib + '.json')}")
