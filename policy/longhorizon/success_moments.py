"""The frame each success episode reaches its task's official success condition (finals FAQ 6), from its joints alone.

Written as a CSV in the format of the 10-29 audit (build_done_features.load_success_frames reads either); the labels of
the planner and the done detector start "finished" there instead of at the end of the recording, which in the organizers'
demonstrations goes on for up to tens of seconds. Rules (forward kinematics of the recorded joints, opening fractions):

- released: the last time a gripper starts to open while its arm is out over the table, the later of the two arms
  (items_handover, sample_loading, handle_basket, table_rearrangement, water_pouring, drawer_open_place);
- click_bell: the right flange first reaches its lowest height (a press leaves nothing else in the joints);
- mixer_operating: the same for the left flange (the button on the mixer's left side);
- manipulate_pipette: the frame the left flange's press is released, as the condition is press and release;
- item_assembly: the two flanges stop moving relative to each other, at the closest they come.

An episode with no such event has no row (the labels then stay on its first part and last 1.5 s).

    policy/fmc3_no1/.venv/bin/python policy/longhorizon/success_moments.py --out <csv> [--check <old audit csv>]
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sim2real"))

from policy.longhorizon import build_done_features as bdf  # noqa: E402

AUDIT_20261029 = (
    REPO_ROOT / "audit_reports" / "success_time_check_20261029" / "成功时刻.csv"
)  # by eye, 320 long episodes
RELEASE_TASKS = {
    "items_handover",
    "sample_loading",
    "handle_basket",
    "table_rearrangement",
    "water_pouring",
    "drawer_open_place",
}
OUT_OVER_TABLE_M = 0.12  # a gripper opening nearer the arm base than this is the arm going home, not a release
MIN_OPENING_RISE = 0.05  # of the full opening: 5 mm
PRESS_DEPTH_M = 0.003  # "reached the lowest point" within this
PIPETTE_LIFTS_M = 0.015  # a press is followed by a rise of at least this
PIPETTE_RELEASED_M = 0.01  # the left flange back up this much after the press
ASSEMBLY_HOLD_M = 0.008  # the tubes count as still relative to each other within this (10-29 audit)


def flange_positions(state: np.ndarray) -> np.ndarray:
    """(n, 14) -> (n, 2, 3): both flanges, each in its own arm's base frame (the bases are parallel)."""
    import ee_pose

    return np.stack([[ee_pose.arm_pose9(q[:6])[:3], ee_pose.arm_pose9(q[7:13])[:3]] for q in state])


def last_release(opening: np.ndarray, reach: np.ndarray) -> int | None:
    """The first frame of the last rise of ``opening`` by MIN_OPENING_RISE with the arm out over the table (a gap of up
    to 3 frames does not end a rise), or None."""
    rising = np.diff(opening, prepend=opening[0]) > 0.001
    last, start, gap = None, None, 0
    for t, up in enumerate(rising):
        if up:
            start, gap = t if start is None else start, 0
        elif start is not None:
            gap += 1
        if start is not None and (gap > 3 or t == len(rising) - 1):
            if opening[start : t + 1].max() - opening[start] >= MIN_OPENING_RISE and reach[start] >= OUT_OVER_TABLE_M:
                last = max(start - 1, 0)
            start, gap = None, 0
    return last


def first_at_lowest(height: np.ndarray) -> int:
    return int(np.argmax(height <= height.min() + PRESS_DEPTH_M))


def pipette_released(left: np.ndarray) -> int:
    """left: the left flange, (n, 3). Once the arm is out at the pipette, the press is its lowest point that it still
    rises from (near the end of a demonstration it can sink lower, going home); released PIPETTE_RELEASED_M above it.
    An episode stopped during the press: its lowest point."""
    reach, height = left[:, 0], left[:, 2]
    out = int(np.argmax(reach >= 0.75 * reach.max()))
    rises_later = np.maximum.accumulate(height[::-1])[::-1] >= height + PIPETTE_LIFTS_M
    candidates = np.flatnonzero(rises_later[out:]) + out
    press = int(candidates[np.argmin(height[candidates])]) if len(candidates) else out + int(np.argmin(height[out:]))
    after = np.flatnonzero(height[press:] >= height[press] + PIPETTE_RELEASED_M)
    return press + int(after[0]) if len(after) else press


def assembly_docked(flanges: np.ndarray) -> int | None:
    """The first frame the flanges' relative position stays within ASSEMBLY_HOLD_M of the closest they come sideways (a
    left flange's y falls and a right one's rises towards the middle, so their difference is lowest when docked)."""
    relative = flanges[:, 0] - flanges[:, 1]
    docked = int(np.argmin(relative[:, 1]))
    far = np.flatnonzero(np.linalg.norm(relative[: docked + 1] - relative[docked], axis=1) > ASSEMBLY_HOLD_M)
    return int(far[-1]) + 1 if len(far) else None


def success_moment(task: str, state: np.ndarray) -> tuple[int | None, str]:
    """(frame, the rule that gave it) for one episode's (n, 14) state."""
    flanges = flange_positions(state)
    if task in RELEASE_TASKS:
        found = [last_release(state[:, 6], flanges[:, 0, 0]), last_release(state[:, 13], flanges[:, 1, 0])]
        arms = [(frame, arm) for frame, arm in zip(found, "LR", strict=True) if frame is not None]
        if not arms:
            return None, "no release"
        frame, arm = max(arms)
        return frame, f"release starts ({arm})"
    if task == "click_bell":
        return first_at_lowest(flanges[:, 1, 2]), "right flange lowest"
    if task == "mixer_operating":
        return first_at_lowest(flanges[:, 0, 2]), "left flange lowest"
    if task == "manipulate_pipette":
        return pipette_released(flanges[:, 0]), "press released"
    if task == "item_assembly":
        frame = assembly_docked(flanges)
        return frame, "tubes still relative to each other" if frame is not None else "no docking"
    raise ValueError(task)


def moment_of(row: dict) -> dict:
    import pandas as pd

    state = np.stack(pd.read_parquet(row["parquet"], columns=["observation.state"])["observation.state"].to_numpy())
    if state.shape[1] != 14:
        state = state[:, bdf.OFFICIAL_STATE]
    frame, rule = success_moment(row["task"], state.astype(np.float64))
    return {**row, "frame": frame, "rule": rule}


def source_of(row: dict) -> str:
    """As in the audit: "官方 Real/cobotmagic_Real_<task>" or real_rollouts/<task>/<recording>; the last word is the key."""
    path = Path(row["dataset"])
    return f"官方 Real/{path.name}" if row["source"] == "official" else f"real_rollouts/{path.parent.name}/{path.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--check", default=str(AUDIT_20261029), help="an audit CSV to compare with")
    parser.add_argument("--tasks", nargs="*", default=list(bdf.TASKS))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    datasets = bdf.official_datasets(REPO_ROOT / "Real", args.tasks) + bdf.our_datasets(
        REPO_ROOT / "real_rollouts", args.tasks
    )
    rows = [row for dataset in datasets for row in bdf.episodes_of(dataset, {}) if row["success"]]
    print(f"{len(rows)} success episodes", flush=True)
    with cf.ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(moment_of, rows, chunksize=16))

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["任务", "来源", "原始数据", "原始组号", "成功帧(原视频,10fps)", "总帧数", "怎么判断的"])
        for r in results:
            if r["frame"] is not None:
                source = "官方" if r["source"] == "official" else r["recording"]
                writer.writerow(
                    [r["task"], source, source_of(r), r["episode_index"], r["frame"], r["length"], r["rule"]]
                )

    audited = bdf.load_success_frames(Path(args.check)) if args.check else {}
    by_task = collections.defaultdict(list)
    for r in results:
        by_task[r["task"], r["source"]].append(r)
    print("task source | episodes found | end - success s: median p5 p95 | vs audit (n, median |diff| s, within 2 s)")
    for (task, source), group in sorted(by_task.items()):
        found = [r for r in group if r["frame"] is not None]
        gaps = np.array([(r["length"] - 1 - r["frame"]) / r["fps"] for r in found]) if found else np.zeros(1)
        diffs = [
            (r["frame"] - audited[key]) / r["fps"]
            for r in found
            if (key := (bdf.dataset_key(Path(r["dataset"])), r["episode_index"])) in audited
        ]
        vs = f"{len(diffs)} {np.median(np.abs(diffs)):.1f} {np.mean(np.abs(diffs) <= 2):.0%}" if diffs else "-"
        print(
            f"{task:21s} {source:8s} | {len(group):4d} {len(found):4d} | "
            f"{np.median(gaps):5.1f} {np.percentile(gaps, 5):5.1f} {np.percentile(gaps, 95):5.1f} | {vs}",
            flush=True,
        )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
