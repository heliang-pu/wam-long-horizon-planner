"""Training and replay data of the high-level planner (planner_prompt.py), from the single-task real recordings.

Every recorded episode is one skill of a chain: a random goal of 1-4 skills puts the episode's task at a random place,
the memory says the skills before it are finished, and the episode's own labels (train_done_detector.frame_labels:
success flags, audited success frames) say whether its skill is finished at a frame:
- not finished (and every frame of a failed episode): keep the running skill, the memory stays;
- finished: the next skill of the goal (or done), the memory adds this one;
- the band in between is not trained on.
The images of a frame are the three cameras now and the top camera 3, 2 and 1 s before (1 frame per second, 320x240).

Held out (the same hash as the done detector's): the episodes eval_planner.py replays second by second; all their
frames are kept. Training episodes keep their finished frames and a few not-finished ones.

    ../venvs/planner/bin/python policy/longhorizon/build_planner_data.py --out ~/workspace/models/longhorizon/planner_data_<date>

Writes frames/<episode>/<frame>_<camera>.jpg, episodes.jsonl (per episode: its frames and labels) and samples.jsonl
(per training sample: images, goal, running skill, memory, target).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.longhorizon import build_done_features as bdf  # noqa: E402
from policy.longhorizon import planner_prompt as pp  # noqa: E402
from policy.longhorizon.train_done_detector import frame_labels  # noqa: E402
from policy.longhorizon.train_done_detector import held_out  # noqa: E402

FPS = 1.0  # planner frames per second (a chunk is 1-2 s)


def episode_dir(row: dict) -> str:
    return f"{row['task']}/{row['source']}/{row['recording']}/episode_{row['episode_index']:06d}"


def pick_frames(labels: np.ndarray, rng: random.Random, n_negative: int, n_finished: int = 3) -> list[int]:
    """A training episode's frames: the first finished one (the switch) and up to n_finished in all, and n_negative
    not-finished ones (the last one always)."""
    finished = [k for k, y in enumerate(labels) if y == 1]
    not_finished = [k for k, y in enumerate(labels) if y == 0]
    chosen = set(finished[:1]) | set(rng.sample(finished[1:], max(0, min(n_finished, len(finished)) - 1)))
    if not_finished:
        chosen.add(not_finished[-1])  # the end of a failed episode, or the last frame before the band
        chosen.update(rng.sample(not_finished, min(n_negative, len(not_finished))))
    return sorted(chosen)


def extract(row: dict, out: str, n_negative: int, n_finished: int, seed: int, *, val: bool) -> dict:
    """Worker: decode one episode at 1 fps and write the JPEGs it needs; its episodes.jsonl line."""
    from PIL import Image

    stride = max(1, round(row["fps"] / FPS))
    views, _state, keep = bdf.decode_frames(row, stride)
    labels = frame_labels(row, keep, pos_tail_s=1.5)
    rng = random.Random(f"{seed}/{episode_dir(row)}")
    frames = list(range(len(keep))) if val else pick_frames(labels, rng, n_negative, n_finished)
    needed = {(k, cam) for k in frames for cam in pp.CAMERAS}
    needed |= {(max(0, k - s), "cam_high") for k in frames for s in pp.PAST_S}
    root = Path(out) / "frames" / episode_dir(row)
    root.mkdir(parents=True, exist_ok=True)
    for k, cam in sorted(needed):
        path = root / f"{k:04d}_{cam}.jpg"
        if not path.exists():
            partial = path.with_suffix(".tmp")  # a killed run leaves no half image that a rerun would take as done
            Image.fromarray(views[cam][k]).resize(pp.IMAGE_SIZE, Image.BILINEAR).save(partial, "JPEG", quality=90)
            partial.replace(path)
    return {
        "dir": episode_dir(row),
        "task": row["task"],
        "split": "val" if val else "train",
        "success": bool(row["success"]),
        "source": row["source"],
        "frames": frames,
        "frame_index": [int(v) for v in keep],
        "labels": [int(v) for v in labels],
    }


def images_of(episode: dict, k: int, past=pp.PAST_S) -> list[str]:
    """The image paths of frame k (relative to the data dir): three cameras now, the top camera ``past`` s before."""
    base = f"frames/{episode['dir']}"
    now = [f"{base}/{k:04d}_{cam}.jpg" for cam in pp.CAMERAS]
    return now + [f"{base}/{max(0, k - s):04d}_cam_high.jpg" for s in past]


def chain_with(task: str, rng: random.Random, max_len: int = 4) -> tuple[list[str], int]:
    """A random goal of 1..max_len distinct skills with ``task`` in it; (chain, its index)."""
    others = [t for t in pp.SKILLS if t != task]
    length = rng.randint(1, max_len)
    chain = rng.sample(others, length - 1)
    index = rng.randint(0, length - 1)
    chain.insert(index, task)
    return chain, index


def sample_of(episode: dict, k: int, rng: random.Random) -> dict:
    """One training sample: the episode's frame k inside a random chain."""
    chain, index = chain_with(episode["task"], rng)
    finished = episode["labels"][k] == 1
    running = episode["task"]
    if index == 0 and k == 0 and rng.random() < 0.5:
        running = None  # the start of the whole chain: nothing runs yet
    return {
        "episode": episode["dir"],
        "frame": k,
        "images": images_of(episode, k),
        "goal": pp.goal_text(chain, rng),
        "chain": chain,
        "index": index,
        "running": running,
        "memory": pp.memory_text(chain[:index]),
        "target_skill": pp.next_skill(chain, index) if finished else episode["task"],
        "target_memory": pp.memory_text(chain[: index + 1] if finished else chain[:index]),
        "finished": finished,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", nargs="*", default=list(pp.SKILLS))
    parser.add_argument("--val-percent", type=float, default=15.0)
    parser.add_argument("--negatives", type=int, default=4, help="not-finished frames per training episode")
    parser.add_argument("--finished", type=int, default=3, help="finished frames per training episode (the first one)")
    parser.add_argument("--finished-contexts", type=int, default=2, help="random chains per finished frame")
    parser.add_argument("--contexts", type=int, default=2, help="random chains per not-finished frame")
    parser.add_argument("--limit", type=int, default=0, help="at most this many episodes per dataset (smoke tests)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    datasets = bdf.official_datasets(REPO_ROOT / "Real", args.tasks) + bdf.our_datasets(
        REPO_ROOT / "real_rollouts", args.tasks
    )
    success_frames = bdf.load_success_frames(bdf.DEFAULT_SUCCESS_FRAMES)
    rows = []
    for dataset in datasets:
        episodes = bdf.episodes_of(dataset, success_frames)
        rows += episodes[: args.limit] if args.limit else episodes
    print(f"{len(datasets)} datasets, {len(rows)} episodes -> {out}", flush=True)

    episodes = []
    with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [
            pool.submit(
                extract, row, str(out), args.negatives, args.finished, args.seed, val=held_out(row, args.val_percent)
            )
            for row in rows
        ]
        for n, future in enumerate(cf.as_completed(futures), 1):
            try:
                episodes.append(future.result())
            except Exception as exc:  # a broken video: reported, the rest goes on
                print(f"FAILED: {exc!r}", flush=True)
            if n % 100 == 0 or n == len(futures):
                print(f"{n}/{len(futures)} episodes", flush=True)
    episodes.sort(key=lambda e: e["dir"])
    (out / "episodes.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes))

    rng = random.Random(args.seed)
    samples = []
    for episode in episodes:
        if episode["split"] != "train":
            continue
        for k in episode["frames"]:
            if episode["labels"][k] == -1:
                continue
            repeats = args.finished_contexts if episode["labels"][k] == 1 else args.contexts
            samples += [sample_of(episode, k, rng) for _ in range(repeats)]
    rng.shuffle(samples)
    (out / "samples.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples))
    finished = sum(s["finished"] for s in samples)
    n_val = sum(e["split"] == "val" for e in episodes)
    print(f"{len(samples)} training samples ({finished} finished), {n_val} held-out episodes -> {out}", flush=True)
    (out / "meta.json").write_text(
        json.dumps({"args": vars(args), "samples": len(samples), "finished": finished, "held_out": n_val}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
