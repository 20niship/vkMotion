#!/usr/bin/env python3
"""vamp / cuRobo / vkMotion 共通のベンチ入力を生成する。

- VAMP同梱のMotionBenchMaker(panda)問題を中立JSONへ変換 (bench/data/problems.json)
- VAMPの球モデル(panda_spherized.urdf)からメッシュ無しURDFを作り(bench/data/panda.urdf)、
  3ライブラリが同一の衝突球・同一のACMで動くようにする
- cuRobo用ロボットYAML(bench/data/panda_curobo.yml)も同じ球から生成する
"""
import argparse
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
VAMP = HERE / ".deps" / "vamp" / "resources" / "panda"
OUT = HERE / "data"


def quat_to_wxyz(q_xyzw):
    return [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]


def convert_pose(tf):
    return tf["position"], tf["orientation"]  # orientation: xyzw (MoveIt)


def quat_mul(a, b):  # xyzw
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2]


def quat_rot(q, v):  # xyzw
    x, y, z, w = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return r @ np.asarray(v)


def load_scene(data, prob):
    objs = []
    for co in data["world"]["collision_objects"]:
        bp, bq = (([0, 0, 0], [0, 0, 0, 1]) if "pose" not in co else convert_pose(co["pose"]))
        pp, pq = convert_pose(co["primitive_poses"][0])
        pos = (np.asarray(bp) + quat_rot(bq, pp)).tolist()
        quat = quat_mul(bq, pq)
        prim = co["primitives"][0]
        t, d = prim["type"], prim["dimensions"]
        o = {"name": co["id"], "pose": pos + quat_to_wxyz(quat)}
        if t == "box":
            o.update(type="cuboid", dims=list(d))
        elif t == "sphere":
            o.update(type="sphere", radius=d[0])
        elif t == "cylinder":
            # cuRoboのシーン衝突はOBB/mesh/voxelのみ(円柱はOBBに変換される)。3ライブラリで同一形状にするため、円柱は外接直方体として正規化する。
            length, r = d[0], d[1]
            o.update(type="cuboid", dims=[2 * r, 2 * r, length])
        else:
            raise RuntimeError(t)
        objs.append(o)
    return objs


def load_request(data, joints):
    js = data["start_state"]["joint_state"]
    start = [js["position"][js["name"].index(j)] for j in joints]
    jc = {e["joint_name"]: e["position"] for e in data["goal_constraints"][0]["joint_constraints"]}
    return start, [jc[j] for j in joints]


def build_problems(max_per_scene):
    import vamp
    from yaml import CSafeLoader as Loader

    joints = vamp.panda.joint_names()
    scenes, reqs = defaultdict(dict), defaultdict(dict)
    tar_path = VAMP / "problems.tar.bz2"
    with tarfile.open(tar_path, "r:bz2") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            _, prob, fname = m.name.split("/")
            prob = prob.replace("_panda", "")
            idx = int(re.findall(r"\d+", fname)[0])
            data = yaml.load(tar.extractfile(m).read(), Loader=Loader)
            if "scene" in fname:
                scenes[prob][idx] = load_scene(data, prob)
            else:
                reqs[prob][idx] = load_request(data, joints)

    problems = []
    for prob in sorted(scenes):
        n = 0
        for idx in sorted(scenes[prob]):
            start, goal = reqs[prob][idx]
            env = make_vamp_env(vamp, scenes[prob][idx])
            # VAMPで start/goal 両方が有効なものだけ採用(後段で各ライブラリが自モデルでも再検証する)
            if not (vamp.panda.validate(start, env) and vamp.panda.validate(goal, env)):
                continue
            pos, quat = ee_pose(vamp, goal)
            problems.append({"scene": prob, "index": idx, "objects": scenes[prob][idx], "start": start, "goal": goal,
                             "goal_pose": pos + quat})
            n += 1
            if n >= max_per_scene:
                break
    return joints, problems


def make_vamp_env(vamp, objs):
    env = vamp.Environment()
    for o in objs:
        x, y, z, qw, qx, qy, qz = o["pose"]
        e = euler_xyz([qx, qy, qz, qw])
        if o["type"] == "cuboid":
            c = vamp.Cuboid([x, y, z], e, [v / 2 for v in o["dims"]])
            env.add_cuboid(c)
        elif o["type"] == "sphere":
            env.add_sphere(vamp.Sphere([x, y, z], o["radius"]))
        else:  # capsule(base/tip = local z)
            length = o["tip"][2] - o["base"][2]
            env.add_capsule(vamp.Cylinder([x, y, z], e, o["radius"], length))
    return env


def euler_xyz(q):  # xyzw -> 静的xyz(roll,pitch,yaw) ; vamp.transformations.euler_from_matrix(sxyz) と同じ
    import vamp.transformations as T
    m = T.quaternion_matrix(q)  # vamp.transformations は xyzw
    return list(T.euler_from_matrix(m))


def ee_pose(vamp, q):
    """VAMP FKのend-effector(panda_grasptarget)姿勢 (xyz, wxyz)。"""
    import vamp.transformations as T
    m = np.asarray(vamp.panda.eefk(q), dtype=float)
    p = m[:3, 3].tolist()
    qwxyz = quat_to_wxyz(T.quaternion_from_matrix(m).tolist())  # vamp.transformations は xyzw を返す
    return p, qwxyz


def write_urdf_and_cfg():
    """panda_spherized.urdf からメッシュ(visual)とinertialを除いたURDFを書き、cuRobo YAMLを生成。"""
    tree = ET.parse(VAMP / "panda_spherized.urdf")
    root = tree.getroot()
    spheres = defaultdict(list)
    for link in root.findall("link"):
        for tag in ("visual", "inertial"):
            for e in link.findall(tag):
                link.remove(e)
        for col in link.findall("collision"):
            sp = col.find("geometry/sphere")
            o = col.find("origin")
            xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
            spheres[link.get("name")].append({"center": xyz, "radius": float(sp.get("radius"))})
    OUT.mkdir(exist_ok=True)
    tree.write(OUT / "panda.urdf", xml_declaration=True, encoding="utf-8")

    # SRDF: ACM
    acm = [(e.get("link1"), e.get("link2")) for e in ET.parse(VAMP / "panda.srdf").getroot().findall("disable_collisions")]
    (OUT / "acm.json").write_text(json.dumps(acm))

    links = [l for l in spheres if spheres[l]]
    ignore = defaultdict(list)
    for a, b in acm:
        if a in links and b in links:
            ignore[a].append(b)
            ignore[b].append(a)
    joints = [f"panda_joint{i}" for i in range(1, 8)]
    cfg = {"robot_cfg": {"kinematics": {
        "format_version": 2.0,
        "urdf_path": str((OUT / "panda.urdf").resolve()),
        "asset_root_path": str(OUT.resolve()),
        "base_link": "panda_link0",
        "tool_frames": ["panda_grasptarget"],
        "collision_link_names": links,
        "collision_sphere_buffer": 0.0,
        "collision_spheres": dict(spheres),
        "self_collision_buffer": {l: 0.0 for l in links},
        "self_collision_ignore": {l: sorted(set(ignore[l])) for l in links},
        "cspace": {"joint_names": joints, "default_joint_position": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
                   "cspace_distance_weight": [1.0] * 7, "null_space_weight": [1.0] * 7,
                   "max_acceleration": 15.0, "max_jerk": 500.0},
        "mesh_link_names": [],
        "lock_joints": None,
    }}}
    (OUT / "panda_curobo.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def write_samples(problems, n=2000, seed=0):
    """FK/衝突判定の精度比較用の共通サンプル。参照値はVAMP(FK・衝突)。"""
    import vamp
    rng = np.random.default_rng(seed)
    lo, hi = np.array(vamp.panda.lower_bounds()), np.array(vamp.panda.upper_bounds())
    q = rng.uniform(lo, hi, size=(n, 7))
    fk = [sum(ee_pose(vamp, qi.tolist()), []) for qi in q]
    scenes = {}
    for p in problems:
        if p["scene"] in scenes:
            continue
        env = make_vamp_env(vamp, p["objects"])
        scenes[p["scene"]] = {"objects": p["objects"], "valid": [int(vamp.panda.validate(qi.tolist(), env)) for qi in q]}
    (OUT / "samples.json").write_text(json.dumps({"lower": lo.tolist(), "upper": hi.tolist(), "q": q.tolist(), "fk": fk, "scenes": scenes}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", type=int, default=100)
    args = ap.parse_args()
    write_urdf_and_cfg()
    joints, problems = build_problems(args.per_scene)
    (OUT / "problems.json").write_text(json.dumps({"robot": "panda", "joints": joints, "ee_link": "panda_grasptarget",
                                                     "problems": problems}))
    write_samples(problems)
    print(f"wrote {len(problems)} problems to {OUT}")


if __name__ == "__main__":
    main()
