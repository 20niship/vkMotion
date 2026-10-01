#!/usr/bin/env python3
"""Convert run_stress_test JSON output (+ URDFs) into a compact Cartesian dataset for the results Artifact."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize import UrdfChain  # noqa: E402


def trajectory_to_cartesian(chain, trajectory):
    return [chain.link_positions(q).tolist() for q in trajectory]


def convert_robot(name, urdf_path, json_path):
    chain = UrdfChain(urdf_path)
    data = json.load(open(json_path))
    trials_out = []
    for t in data["trials"]:
        if t.get("setup_failed"):
            trials_out.append({"trial_index": t["trial_index"], "setup_failed": True})
            continue
        entry = {
            "trial_index": t["trial_index"],
            "seed": t["seed"],
            "obstacles": t["obstacles"],
            "start_links": chain.link_positions(t["start_joint_state"]).tolist(),
            "goal_links": chain.link_positions(t["goal_joint_state"]).tolist(),
            "cpu": {
                "success": t["cpu"]["success"],
                "elapsed_ms": t["cpu"]["elapsed_ms"],
                "links": trajectory_to_cartesian(chain, t["cpu"]["trajectory"]) if t["cpu"]["success"] else [],
            },
            "gpu": {
                "success": t["gpu"]["success"],
                "elapsed_ms": t["gpu"]["elapsed_ms"],
                "links": trajectory_to_cartesian(chain, t["gpu"]["trajectory"]) if t["gpu"]["success"] else [],
            },
        }
        trials_out.append(entry)
    return {"robot": name, "urdf": str(urdf_path), "dof": len(chain.active), "trials": trials_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", action="append", nargs=3, metavar=("NAME", "URDF", "JSON"), required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    result = [convert_robot(name, urdf, j) for name, urdf, j in args.robot]
    with open(args.output, "w") as f:
        json.dump(result, f)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
