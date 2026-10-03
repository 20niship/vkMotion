#!/usr/bin/env python3
"""Animate a run_stress_test JSON trial: robot, obstacles, CPU/GPU trajectories."""

import argparse
import json
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.eye(3)
    axis = axis / n
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def parse_vec(text, default=(0.0, 0.0, 0.0)):
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


class UrdfChain:
    """Minimal URDF loader: just enough to FK a serial chain of revolute/continuous/prismatic/fixed joints."""

    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = []
        for j in root.findall("joint"):
            origin = j.find("origin")
            axis_el = j.find("axis")
            self.joints.append(
                {
                    "name": j.get("name"),
                    "type": j.get("type"),
                    "parent": j.find("parent").get("link"),
                    "child": j.find("child").get("link"),
                    "xyz": parse_vec(origin.get("xyz") if origin is not None else None),
                    "rpy": parse_vec(origin.get("rpy") if origin is not None else None),
                    "axis": parse_vec(axis_el.get("xyz") if axis_el is not None else None, (1, 0, 0)),
                }
            )
        # order joints root->tip by following parent/child chains from the first link that is never a child
        children = {j["child"] for j in self.joints}
        parents = {j["parent"] for j in self.joints}
        root_link_candidates = parents - children
        root_link = next(iter(root_link_candidates)) if root_link_candidates else self.joints[0]["parent"]
        ordered = []
        current = root_link
        remaining = list(self.joints)
        while remaining:
            nxt = next((j for j in remaining if j["parent"] == current), None)
            if nxt is None:
                break
            ordered.append(nxt)
            remaining.remove(nxt)
            current = nxt["child"]
        self.joints = ordered if ordered else self.joints
        self.active = [j for j in self.joints if j["type"] != "fixed"]

    def link_positions(self, q):
        """Return world-frame positions of every joint frame along the chain for active-joint values q."""
        t = np.eye(4)
        pts = [t[:3, 3].copy()]
        qi = 0
        for j in self.joints:
            local = np.eye(4)
            local[:3, :3] = rpy_to_matrix(j["rpy"])
            local[:3, 3] = j["xyz"]
            t = t @ local
            if j["type"] != "fixed":
                angle = q[qi] if qi < len(q) else 0.0
                qi += 1
                if j["type"] == "prismatic":
                    t = t @ np.vstack([np.hstack([np.eye(3), (j["axis"] * angle).reshape(3, 1)]), [0, 0, 0, 1]])
                else:
                    rot = np.eye(4)
                    rot[:3, :3] = axis_angle_matrix(j["axis"], angle)
                    t = t @ rot
            pts.append(t[:3, 3].copy())
        return np.array(pts)


def quat_to_matrix(qw, qx, qy, qz):
    n = np.linalg.norm([qw, qx, qy, qz])
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def sphere_points(center, radius, n=10):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2))
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    return x, y, z


def draw_obstacle(ax, obs):
    pose = obs["pose"]
    pos = np.array(pose[0:3])
    rot = quat_to_matrix(*pose[3:7])
    kind = obs["type"]
    if kind == "sphere":
        x, y, z = sphere_points(pos, obs["radius"])
        ax.plot_wireframe(x, y, z, color="tab:red", linewidth=0.4, alpha=0.5)
    elif kind == "cuboid":
        dx, dy, dz = (d / 2 for d in obs["dims"])
        corners = np.array([[sx * dx, sy * dy, sz * dz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        world = (rot @ corners.T).T + pos
        ax.scatter(world[:, 0], world[:, 1], world[:, 2], color="tab:red", s=4, alpha=0.6)
        for i in range(8):
            for j in range(i + 1, 8):
                if bin(i ^ j).count("1") == 1:
                    ax.plot(*zip(world[i], world[j]), color="tab:red", linewidth=0.5, alpha=0.4)
    elif kind in ("cylinder", "capsule"):
        radius = obs["radius"]
        if kind == "cylinder":
            half_h = obs["height"] / 2
            base_local, tip_local = np.array([0, 0, -half_h]), np.array([0, 0, half_h])
        else:
            base_local, tip_local = np.array(obs["base"]), np.array(obs["tip"])
        base = rot @ base_local + pos
        tip = rot @ tip_local + pos
        theta = np.linspace(0, 2 * np.pi, 12)
        axis_dir = tip - base
        length = np.linalg.norm(axis_dir)
        axis_dir = axis_dir / length if length > 1e-9 else np.array([0, 0, 1])
        arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp1 = np.cross(axis_dir, arbitrary)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(axis_dir, perp1)
        for frac in (0.0, 1.0):
            center = base + axis_dir * length * frac
            ring = np.array([center + radius * (np.cos(t) * perp1 + np.sin(t) * perp2) for t in theta])
            ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="tab:red", linewidth=0.5, alpha=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--robot-urdf", required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--output", default=None, help="out.gif or out.mp4; omitted = plt.show()")
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)
    trial = data["trials"][args.trial]
    if trial.get("setup_failed"):
        raise SystemExit(f"trial {args.trial} had setup_failed=true, nothing to animate")

    chain = UrdfChain(args.robot_urdf)

    cpu_traj = [np.array(q) for q in trial["cpu"]["trajectory"]]
    gpu_traj = [np.array(q) for q in trial["gpu"]["trajectory"]]
    n_frames = max(len(cpu_traj), len(gpu_traj), 1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for obs in trial["obstacles"]:
        draw_obstacle(ax, obs)

    cpu_ee = [chain.link_positions(q)[-1] for q in cpu_traj] if cpu_traj else []
    gpu_ee = [chain.link_positions(q)[-1] for q in gpu_traj] if gpu_traj else []
    if cpu_ee:
        cpu_ee_arr = np.array(cpu_ee)
        ax.plot(cpu_ee_arr[:, 0], cpu_ee_arr[:, 1], cpu_ee_arr[:, 2], color="tab:blue", linewidth=1.5, label="CPU end-effector path")
    if gpu_ee:
        gpu_ee_arr = np.array(gpu_ee)
        ax.plot(gpu_ee_arr[:, 0], gpu_ee_arr[:, 1], gpu_ee_arr[:, 2], color="tab:green", linestyle="--", linewidth=1.5, label="GPU end-effector path")

    (robot_line,) = ax.plot([], [], [], color="black", marker="o", markersize=3, linewidth=2, label="robot (CPU traj)")

    all_pts = np.array(cpu_ee + gpu_ee) if (cpu_ee or gpu_ee) else np.zeros((1, 3))
    span = max(0.3, float(np.max(np.abs(all_pts))) * 1.3)
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(-span, span)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    cpu_ms, gpu_ms = trial["cpu"]["elapsed_ms"], trial["gpu"]["elapsed_ms"]
    cpu_ok, gpu_ok = trial["cpu"]["success"], trial["gpu"]["success"]
    title = ax.set_title(
        f"trial {trial['trial_index']}  obstacles={len(trial['obstacles'])}\n"
        f"CPU: {'ok' if cpu_ok else 'fail'} {cpu_ms:.2f}ms   GPU: {'ok' if gpu_ok else 'fail'} {gpu_ms:.2f}ms"
    )
    ax.legend(loc="upper left", fontsize=7)

    anim_traj = cpu_traj if cpu_traj else gpu_traj

    def update(frame):
        q = anim_traj[min(frame, len(anim_traj) - 1)] if anim_traj else np.zeros(0)
        pts = chain.link_positions(q)
        robot_line.set_data(pts[:, 0], pts[:, 1])
        robot_line.set_3d_properties(pts[:, 2])
        return robot_line, title

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=120, blit=False)

    if args.output:
        if args.output.endswith(".gif"):
            anim.save(args.output, writer=animation.PillowWriter(fps=8))
        else:
            anim.save(args.output, writer=animation.FFMpegWriter(fps=8))
        print(f"wrote {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
