#!/usr/bin/env python3
"""cuRobo v2 (GPU, CUDA/warp) の FK / 衝突判定 / IK / 関節空間・姿勢ゴール軌道生成を計測する。"""
import argparse
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import DATA, load_problems, load_samples, path_length, save, timer  # noqa: E402

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose  # noqa: E402
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg  # noqa: E402
from curobo.kinematics import Kinematics, KinematicsCfg  # noqa: E402

import curobo.runtime as _rt  # noqa: E402

_rt.cuda_graph_reset = True  # 問題の種類(関節ゴール/姿勢ゴール)でgoal bufferの形が変わるためCUDA graphの再構築を許可
DEV = "cuda"
MAX_CUBOIDS = 64


def sync():
    torch.cuda.synchronize()


def scene_dict(objs):
    d = {"cuboid": {}}
    for o in objs:
        assert o["type"] == "cuboid"
        d["cuboid"][o["name"]] = {"dims": o["dims"], "pose": o["pose"]}  # pose = xyz + wxyz
    return d


def robot_dict():
    return yaml.safe_load((DATA / "panda_curobo.yml").read_text())


def bench_fk_collision(samples):
    out = {}
    cfg = KinematicsCfg.from_robot_yaml_file(str(DATA / "panda_curobo.yml")) if hasattr(KinematicsCfg, "from_robot_yaml_file") else None
    kin = Kinematics(cfg)
    q = torch.tensor(samples["q"], device=DEV, dtype=torch.float32)
    js = JointState.from_position(q, joint_names=kin.joint_names)
    for _ in range(3):
        st = kin.compute_kinematics(js)
    sync()
    t0 = timer()
    n = 10
    for _ in range(n):
        st = kin.compute_kinematics(js)
    sync()
    out["fk_us_per_call"] = (timer() - t0) / n / len(q) * 1e6
    tp = st.tool_poses.to_dict()["panda_grasptarget"]
    out["fk"] = tp.position.reshape(len(q), 3).cpu().tolist()

    out["collision"] = {}
    big = torch.rand(100000, 7, device=DEV) * (torch.tensor(samples["upper"], device=DEV) - torch.tensor(samples["lower"], device=DEV)) + torch.tensor(samples["lower"], device=DEV)
    for name, sc in samples["scenes"].items():
        ccfg = RobotCollisionCheckerCfg.load_from_config(robot_dict(), scene_dict(sc["objects"]), n_cuboids=MAX_CUBOIDS,
                                                         collision_activation_distance=0.0, self_collision_activation_distance=0.0)
        chk = RobotCollisionChecker(ccfg)

        def validate(x):  # RobotCollisionChecker.validate はcost APIの変更に追従していないので同等処理を直接呼ぶ
            b = x.shape[0]
            st = chk.get_kinematics(x.unsqueeze(1))
            ns = st.robot_spheres.shape[-2]
            for c in (chk.collision_cost, chk.collision_constraint, chk.self_collision_cost):
                if hasattr(c, "update_num_spheres"):
                    c.update_num_spheres(ns, b, 1)
                else:
                    c.setup_batch_tensors(b, 1)
            d_world = chk.collision_constraint.forward(st).reshape(b, -1)
            d_self = chk.self_collision_cost.forward(st.robot_spheres).reshape(b, -1)
            return ~((d_world > 0).any(-1) | (d_self > 0).any(-1))

        valid = validate(q).int().cpu().tolist()
        for _ in range(2):
            validate(big)
        sync()
        t0 = timer()
        validate(big)
        sync()
        out["collision"][name] = {"us_per_check": (timer() - t0) / len(big) * 1e6, "valid": valid}
    return out


def make_planner(max_goalset=1):
    cfg = MotionPlannerCfg.create(robot=robot_dict()["robot_cfg"] if False else str(DATA / "panda_curobo.yml"),
                                  scene_model=scene_dict([{"type": "cuboid", "name": "dummy", "dims": [0.01] * 3, "pose": [5, 5, 5, 1, 0, 0, 0]}]),
                                  collision_cache={"obb": MAX_CUBOIDS}, num_trajopt_seeds=4, num_ik_seeds=32)
    mp = MotionPlanner(cfg)
    mp.warmup(enable_graph=True, num_warmup_iterations=5)
    return mp


def bench_plan(problems, mode, planner):
    rows = []
    for k, p in enumerate(problems):
        planner.update_world(scene_to_cfg(p["objects"]))
        planner.reset_seed()
        q0 = JointState.from_position(torch.tensor([p["start"]], device=DEV, dtype=torch.float32), joint_names=planner.joint_names)
        row = {"scene": p["scene"], "index": p["index"], "valid": True}
        sync()
        t0 = timer()
        if mode == "js":
            goal = JointState.from_position(torch.tensor([p["goal"]], device=DEV, dtype=torch.float32), joint_names=planner.joint_names)
            res = planner.plan_cspace(goal, q0, max_attempts=10, enable_graph_attempt=2)
        else:
            gp = p["goal_pose"]
            goal = GoalToolPose(tool_frames=planner.tool_frames,
                                position=torch.tensor([[[[[gp[0], gp[1], gp[2]]]]]], device=DEV, dtype=torch.float32),
                                quaternion=torch.tensor([[[[[gp[3], gp[4], gp[5], gp[6]]]]]], device=DEV, dtype=torch.float32))
            res = planner.plan_pose(goal, q0, max_attempts=10, enable_graph_attempt=2)
        sync()
        dt = (timer() - t0) * 1e3
        ok = res is not None and bool(res.success.any())
        row.update(success=ok, time_ms=dt)
        if ok:
            traj = res.get_interpolated_plan()
            pos = traj.position.reshape(-1, traj.position.shape[-1]).cpu().numpy()
            dt = float(traj.dt.item())
            row.update(path_len=path_length(pos), n_waypoints=len(pos), path=pos.tolist(), dt=dt, motion_time_s=dt * (len(pos) - 1))
            if mode == "pose":
                row.update(pos_err_mm=float(res.position_error.min().item()) * 1e3, rot_err_deg=float(res.rotation_error.min().item()) * 180 / np.pi)
        rows.append(row)
        if k % 20 == 0:
            print(f"[{mode}] {k}/{len(problems)}", flush=True)
    return rows


def scene_to_cfg(objs):
    from curobo.scene import Scene as SceneCfg
    return SceneCfg.create(scene_dict(objs))


def bench_ik(problems, planner):
    rows = []
    for p in problems:
        planner.update_world(scene_to_cfg(p["objects"]))
        gp = p["goal_pose"]
        goal = GoalToolPose(tool_frames=planner.tool_frames,
                            position=torch.tensor([[[[[gp[0], gp[1], gp[2]]]]]], device=DEV, dtype=torch.float32),
                            quaternion=torch.tensor([[[[[gp[3], gp[4], gp[5], gp[6]]]]]], device=DEV, dtype=torch.float32))
        sync()
        t0 = timer()
        r = planner.ik_solver.solve_pose(goal, return_seeds=1)
        sync()
        dt = (timer() - t0) * 1e3
        ok = bool(r.success.any())
        row = {"scene": p["scene"], "index": p["index"], "valid": True, "success": ok, "time_ms": dt}
        if ok:
            row.update(pos_err_mm=float(r.position_error.min().item()) * 1e3, rot_err_deg=float(r.rotation_error.min().item()) * 180 / np.pi)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[], help="fkcol / js / pose / ik")
    a = ap.parse_args()
    problems = load_problems(a.limit or None)
    res = {"lib": "curobo", "device": torch.cuda.get_device_name(0) + " (cuda, warp, CUDA graph)"}
    if "fkcol" not in a.skip:
        res.update(bench_fk_collision(load_samples()))
    planner = make_planner()
    if "js" not in a.skip:
        res["plan_js"] = bench_plan(problems, "js", planner)
    if "pose" not in a.skip:
        res["plan_pose"] = bench_plan(problems, "pose", planner)
    if "ik" not in a.skip:
        res["ik"] = bench_ik(problems, planner)
    save("curobo", res)


if __name__ == "__main__":
    main()
