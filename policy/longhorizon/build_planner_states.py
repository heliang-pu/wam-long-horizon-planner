"""The hands of every episode of a build_planner_data.py directory, for a planner shown them (train_planner.py --state):
<data>/states.json, episode dir -> per kept frame left x, y, z (m, flange in the arm's base frame), opening, right the
same. From the recorded joints; no images are decoded. Run under the policy venv (pyAgxArm's forward kinematics).
Also <data>/takeovers.json: success episodes a model started and the operator finished, episode dir -> the first kept
frame at the takeover (before it the model was failing, for eval_planner.py --sweep).

    policy/fmc3_no1/.venv/bin/python policy/longhorizon/build_planner_states.py --data <planner data>
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sim2real"))

from policy.longhorizon import build_done_features as bdf  # noqa: E402
from policy.longhorizon import planner_prompt as pp  # noqa: E402
from policy.longhorizon.build_planner_data import episode_dir  # noqa: E402
from policy.longhorizon.success_moments import flange_positions  # noqa: E402


def hands_of(item: tuple[dict, list[int]]) -> tuple[str, list[list[float]] | None]:
    """One episode's rows; None if the recording no longer has the frames it was built from (a purge renumbered it)."""
    import pandas as pd

    row, keep = item
    state = np.stack(pd.read_parquet(row["parquet"], columns=["observation.state"])["observation.state"].to_numpy())
    if state.shape[1] != bdf.STATE_DIM:
        state = state[:, bdf.OFFICIAL_STATE]
    if keep[-1] >= len(state):
        return episode_dir(row), None
    state = state[keep].astype(np.float64)
    flanges = flange_positions(state)
    rows = np.concatenate([flanges[:, 0], state[:, [6]], flanges[:, 1], state[:, [13]]], axis=1)
    return episode_dir(row), np.round(rows, 3).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    data = Path(args.data).expanduser()
    built = {}
    for line in (data / "episodes.jsonl").read_text().splitlines():
        episode = json.loads(line)
        built[episode["dir"]] = episode["frame_index"]
    datasets = bdf.official_datasets(REPO_ROOT / "Real", pp.SKILLS) + bdf.our_datasets(
        REPO_ROOT / "real_rollouts", pp.SKILLS
    )
    rows = [row for dataset in datasets for row in bdf.episodes_of(dataset, {}) if episode_dir(row) in built]
    with cf.ProcessPoolExecutor(args.workers) as pool:
        found = dict(pool.map(hands_of, [(row, built[episode_dir(row)]) for row in rows], chunksize=16))
    states = {name: value for name, value in found.items() if value is not None}
    (data / "states.json").write_text(json.dumps(states))
    print(f"{len(states)} of {len(built)} episodes -> {data / 'states.json'}", flush=True)
    takeovers = {}
    for row in rows:
        name, start = episode_dir(row), row["intervention_start"]
        if row["success"] and start and not row["teleop"]:
            takeovers[name] = sum(frame < start for frame in built[name])
    (data / "takeovers.json").write_text(json.dumps(takeovers))
    print(f"{len(takeovers)} takeover episodes -> {data / 'takeovers.json'}", flush=True)


if __name__ == "__main__":
    main()
