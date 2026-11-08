"""More of the frames where a model was failing, for the planner's training: <data>/samples_hard.jsonl = samples.jsonl plus,
for the training episodes that failed or that the operator took over (takeovers.json, build_planner_states.py), up to
--frames frames of the last --window seconds before the end or the takeover, each with --contexts random goals. They all
answer "keep the running skill". These look most like a finished task, and random frames of an episode rarely hit them.

Also samples of a robot that stands still: for each training episode one finished and one not-finished frame whose last 1-3
s of top-camera history are the frame itself. In samples.jsonl the history only stands still at the start of an episode,
always "not finished", and a planner learns that no motion means just started: it takes back a switch once the arms stop.

A build_planner_data.py directory stays as it is otherwise (its held-out episodes too), so results can be compared.

    ../venvs/planner/bin/python policy/longhorizon/add_hard_negatives.py --data <planner data>
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.longhorizon import build_done_features as bdf  # noqa: E402
from policy.longhorizon import planner_prompt as pp  # noqa: E402
from policy.longhorizon.build_planner_data import episode_dir  # noqa: E402
from policy.longhorizon.build_planner_data import sample_of  # noqa: E402


def hard_frames(episode: dict, end: int, rng: random.Random, n: int, window: int) -> list[int]:
    """Up to n not-finished frames among the last ``window`` before frame ``end``."""
    pool = [k for k in range(max(0, end - window), end) if episode["labels"][k] == 0]
    return sorted(rng.sample(pool, min(n, len(pool))))


def standing_still(sample: dict, rng: random.Random) -> dict:
    """The sample with the robot still for its last 1-3 s: images are the three cameras, then the top camera PAST_S ago."""
    images = list(sample["images"])
    held = rng.randint(1, len(pp.PAST_S))
    for j, seconds in enumerate(pp.PAST_S):
        if seconds <= held:
            images[len(pp.CAMERAS) + j] = images[0]
    return {**sample, "images": images, "still": held}


def write_frames(item: tuple[dict, str, str, list[int]]) -> None:
    """The JPEGs those frames need that are not there yet."""
    from PIL import Image

    row, out, name, frames = item
    views, _state, _keep = bdf.decode_frames(row, max(1, round(row["fps"])))
    root = Path(out) / "frames" / name
    needed = {(k, cam) for k in frames for cam in pp.CAMERAS}
    needed |= {(max(0, k - s), "cam_high") for k in frames for s in pp.PAST_S}
    for k, cam in sorted(needed):
        path = root / f"{k:04d}_{cam}.jpg"
        if not path.exists():
            partial = path.with_suffix(".tmp")
            Image.fromarray(views[cam][k]).resize(pp.IMAGE_SIZE, Image.BILINEAR).save(partial, "JPEG", quality=90)
            partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--window", type=int, default=6, help="seconds")
    parser.add_argument("--contexts", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = Path(args.data).expanduser()
    episodes = {}
    for line in (data / "episodes.jsonl").read_text().splitlines():
        episode = json.loads(line)
        episodes[episode["dir"]] = episode
    takeovers = json.loads((data / "takeovers.json").read_text())
    datasets = bdf.official_datasets(REPO_ROOT / "Real", pp.SKILLS) + bdf.our_datasets(
        REPO_ROOT / "real_rollouts", pp.SKILLS
    )
    rows = {episode_dir(row): row for dataset in datasets for row in bdf.episodes_of(dataset, {})}
    rng = random.Random(args.seed)
    extra, todo = [], []
    for name, episode in sorted(episodes.items()):
        if episode["split"] != "train" or name not in rows:
            continue
        if len(range(0, rows[name]["length"], max(1, round(rows[name]["fps"])))) != len(episode["frame_index"]):
            continue  # a purge renumbered it since the build
        if not episode["success"]:
            end = len(episode["frame_index"])
        elif takeovers.get(name):
            end = takeovers[name]
        else:
            continue
        frames = hard_frames(episode, end, rng, args.frames, args.window)
        if not frames:
            continue
        todo.append((rows[name], str(data), name, frames))
        for k in frames:
            extra += [sample_of(episode, k, rng) for _ in range(args.contexts)]
    still = []
    for episode in episodes.values():
        if episode["split"] != "train":
            continue
        for wanted in (1, 0):
            frames = [k for k in episode["frames"] if episode["labels"][k] == wanted]
            if frames:
                still.append(standing_still(sample_of(episode, rng.choice(frames), rng), rng))
    print(f"{len(todo)} episodes, {len(extra)} more hard samples, {len(still)} standing still", flush=True)
    extra += still
    with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for n, _ in enumerate(pool.map(write_frames, todo, chunksize=4), 1):
            if n % 200 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    for sample in extra:
        missing = [path for path in sample["images"] if not (data / path).is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
    samples = [json.loads(line) for line in (data / "samples.jsonl").read_text().splitlines()] + extra
    random.Random(args.seed + 1).shuffle(samples)
    partial = data / "samples_hard.jsonl.tmp"
    partial.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples))
    partial.replace(data / "samples_hard.jsonl")
    print(f"{len(samples)} samples -> {data / 'samples_hard.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
