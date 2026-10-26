#!/usr/bin/env python3
"""VAMP (CPU SIMD) の FK / 衝突判定 / 関節空間RRT-Connect+simplify を計測する。"""
import argparse
import sys

import numpy as np
import vamp

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import load_problems, load_samples, path_length, save, timer  # noqa: E402
from prepare import make_vamp_env  # noqa: E402


def bench_fk_collision(samples):
    q = samples["q"]
    out = {}
    t0 = timer()
    fk = [np.asarray(vamp.panda.eefk(qi)) for qi in q]
    out["fk_us_per_call"] = (timer() - t0) / len(q) * 1e6
    out["fk"] = [f[:3, 3].tolist() for f in fk]
    out["collision"] = {}
    for name, sc in samples["scenes"].items():
        env = make_vamp_env(vamp, sc["objects"])
        t0 = timer()
        valid = [int(vamp.panda.validate(qi, env)) for qi in q]
        dt = timer() - t0
        out["collision"][name] = {"us_per_check": dt / len(q) * 1e6, "valid": valid}
    return out


def bench_plan(problems, repeats):
    mod, planner, ps, ss = vamp.configure_robot_and_planner_with_kwargs("panda", "rrtc")
    sampler = mod.halton()
    rows = []
    for p in problems:
        env = make_vamp_env(vamp, p["objects"])
        row = {"scene": p["scene"], "index": p["index"], "valid": bool(mod.validate(p["start"], env) and mod.validate(p["goal"], env))}
        if row["valid"]:
            sampler.reset()
            times, succ, plens, nwp, paths = [], 0, [], [], []
            for _ in range(repeats):
                t0 = timer()
                r = planner(p["start"], [p["goal"]], env, ps, sampler)
                if r.solved:
                    s = mod.simplify(r.path, env, ss, sampler)
                    times.append((timer() - t0) * 1e3)
                    path = np.asarray(s.path.numpy() if hasattr(s.path, "numpy") else s.path)
                    plens.append(path_length(path))
                    nwp.append(len(path))
                    paths.append(path.tolist())
                    succ += 1
                else:
                    times.append((timer() - t0) * 1e3)
            row.update(success=succ == repeats, success_rate=succ / repeats, time_ms=float(np.median(times)), path_len=float(np.median(plens)) if plens else None,
                       n_waypoints=int(np.median(nwp)) if nwp else None, path=paths[0] if paths else None)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="シーンごとの問題数上限(0=全部)")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    problems = load_problems(a.limit or None)
    res = {"lib": "vamp", "device": "CPU (SIMD, 1 thread, python bindings)", **bench_fk_collision(load_samples()), "plan_js": bench_plan(problems, a.repeats)}
    save("vamp", res)


if __name__ == "__main__":
    main()
