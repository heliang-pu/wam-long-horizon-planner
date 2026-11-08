"""End-effector pose of both PiPER arms from their joint angles, as recorded next to the joint-space data.

Per arm 9 numbers: the flange position x, y, z (m) in that arm's base frame, then rot6d = the first two columns of
the flange rotation matrix taken row by row, [R00, R01, R10, R11, R20, R21] -- the layout of the organizers' real
datasets. Left arm first, then right: 18 numbers.

The pose comes from pyAgxArm's modified-DH forward kinematics for the PiPER, which is what the arm controller
reports as its flange pose. The rotation is R = Rz(yaw) Ry(pitch) Rx(roll) of pyAgxArm's Euler angles. Checked
against cobotmagic_Real_sample_loading: for their joint angles this gives their ee_pos and ee_rot6d to about 1e-5
(read as two consecutive 3-vectors, their rot6d looks invalid; it is this row-by-row layout).
"""

from __future__ import annotations

import numpy as np

NAMES = [
    f"{side}_ee_{part}"
    for side in ("left", "right")
    for part in ("pos_x", "pos_y", "pos_z", *(f"rot6d_{i}" for i in range(6)))
]
LEFT_ARM = slice(0, 6)
RIGHT_ARM = slice(7, 13)

_MDH = None


def _mdh():
    global _MDH
    if _MDH is None:
        from pyAgxArm.utiles.mdh_kinematics import get_mdh  # noqa: PLC0415

        _MDH = get_mdh("piper")
    return _MDH


def rot_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz @ ry @ rx


def arm_pose9(real_joints: np.ndarray) -> np.ndarray:
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh  # noqa: PLC0415

    x, y, z, roll, pitch, yaw = fk_from_mdh(_mdh(), [float(v) for v in real_joints])
    r = rot_xyz(roll, pitch, yaw)
    return np.array([x, y, z, *r[:, :2].reshape(6)], dtype=np.float64)  # [R00, R01, R10, R11, R20, R21]


def ee_pose18(vec14: np.ndarray, calibrations) -> np.ndarray:
    """18-dim pose of both arms for a 14-dim state/action in sim units (joints converted with each arm's calibration)."""
    left = calibrations[0].sim_to_real(np.asarray(vec14[LEFT_ARM], dtype=np.float64))
    right = calibrations[1].sim_to_real(np.asarray(vec14[RIGHT_ARM], dtype=np.float64))
    return np.concatenate([arm_pose9(left), arm_pose9(right)]).astype(np.float32)
