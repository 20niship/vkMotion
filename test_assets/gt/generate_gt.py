#!/usr/bin/env python3
# Independent FK reimplementation (no vkMotion/cuRobo code) for tests/test_collision_gt.cpp GT values.
import numpy as np


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy(roll, pitch, yaw):
    # URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    return rz(yaw) @ ry(pitch) @ rx(roll)


def axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])


def T(xyz, R):
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = xyz
    return m


def fk_chain(joints, q):
    """joints: list of (xyz, rpy, axis). q: joint values. Returns final 4x4 world transform."""
    world = np.eye(4)
    for (xyz, rpy_, axis), value in zip(joints, q):
        origin = T(xyz, rpy(*rpy_))
        motion = T((0, 0, 0), axis_angle(axis, value))
        world = world @ origin @ motion
    return world


# ---- OpenArm (right arm, 7-DoF) ------------------------------------------------
openarm_joints = [
    ((0, 0, 0.0625), (0, 0, 0), (0, 0, 1)),
    ((-0.0301, 0.0, 0.06), (0, 0, 0), (-1, 0, 0)),
    ((0.0301, 0.0, 0.06625), (0, 0, 0), (0, 0, 1)),
    ((0.0, 0.0315, 0.15375), (0, 0, 0), (0, 1, 0)),
    ((0.0, -0.0315, 0.0955), (0, 0, 0), (0, 0, 1)),
    ((0.0375, 0.0, 0.1205), (0, 0, 0), (1, 0, 0)),
    ((-0.0375, 0.0, 0.0), (0, 0, 0), (0, 1, 0)),
]
openarm_tip_offset = np.array([0.000001, 0.0205, 0.0])

# ---- SO-101 (6-DoF) --------------------------------------------------------------
so101_joints = [
    ((0.0388353, -0.00000000897657, 0.0624), (3.14159, 0, -3.14159), (0, 0, 1)),
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0), (0, 0, 1)),
    ((-0.11257, -0.028, 0), (0, 0, 1.5708), (0, 0, 1)),
    ((-0.1349, 0.0052, 0), (0, 0, -1.5708), (0, 0, 1)),
    ((0, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159), (0, 0, 1)),
    ((0.0202, 0.0188, -0.0234), (1.5708, 0, 0), (0, 0, 1)),
]

# ---- UR5e (6-DoF) -----------------------------------------------------------------
ur5e_joints = [
    ((0, 0, 0.1625), (0, 0, 0), (0, 0, 1)),
    ((0, 0, 0), (1.570796327, 0, 0), (0, 0, 1)),
    ((-0.425, 0, 0), (0, 0, 0), (0, 0, 1)),
    ((-0.3922, 0, 0.1333), (0, 0, 0), (0, 0, 1)),
    ((0, -0.0997, 0), (1.570796327, 0, 0), (0, 0, 1)),
    ((0, 0.0996, 0), (1.570796327, 0, 3.141592654), (0, 0, 1)),
]

if __name__ == "__main__":
    zero7 = [0.0] * 7
    zero6 = [0.0] * 6

    m = fk_chain(openarm_joints, zero7)
    tip = (m @ np.array([*openarm_tip_offset, 1.0]))[:3]
    print("openarm zero-config link_tip world position:", tip)

    q = [0.5, -0.3, 0.2, 0.0, 0.4, -0.6, 0.1]
    m = fk_chain(openarm_joints, q)
    tip = (m @ np.array([*openarm_tip_offset, 1.0]))[:3]
    print("openarm q=", q, "-> link_tip world position:", tip)

    m = fk_chain(so101_joints, zero6)
    print("so101 zero-config moving_jaw world position:", m[:3, 3])

    q = [0.4, -0.5, 0.6, -0.2, 0.3, 0.0]
    m = fk_chain(so101_joints, q)
    print("so101 q=", q, "-> moving_jaw world position:", m[:3, 3])

    m = fk_chain(ur5e_joints, zero6)
    print("ur5e zero-config wrist_3_link world position:", m[:3, 3])

    q = [0.1, -1.0, 1.2, -0.5, 0.7, 0.0]
    m = fk_chain(ur5e_joints, q)
    print("ur5e q=", q, "-> wrist_3_link world position:", m[:3, 3])
