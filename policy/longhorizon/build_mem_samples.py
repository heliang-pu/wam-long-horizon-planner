"""Training samples of a memory planner (train_planner.py --progress) from build_planner_data.py / add_hard_negatives.py
samples: the same frames and images, each put into a new random goal of 1..10 skills that may repeat a skill
(planner_prompt.any_chain), and the memory keeping how far the running skill is.

The progress of a frame is its share of the time to the frame its skill is finished (planner_prompt.progress_of; a
failure: the task's median time to there, never finished). The memory the planner is asked with is the one of 1-4 s
before (the robot takes a stage once two of three replies agree, so it lags), sometimes a stage off; the reply is the
next skill and the finished list with it once the frame is finished, else the running skill and its progress now. Some
finished frames are asked with nothing running and the skill already in the memory (after the operator's n).

G2 (--past 20,10,3,2,1 --lag 2,3,4,5,6,7,8 --long-still 0.4 --out samples_mem2.jsonl): the top camera also 10 and 20 s
before (frames extracted from the recordings where missing), the memory lagging as much as it does when asked every 2-3 s
(two of three replies take a stage), and some standing-still samples still for 10 or 20 s, not 1-3.

    python policy/longhorizon/build_mem_samples.py --data <planner data> [--samples samples_hard.jsonl]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import random
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.longhorizon import planner_prompt as pp  # noqa: E402

LAG_S = (1, 2, 3, 4)  # the memory asked with is from this many planner frames (seconds) before
OFF_BY_ONE = 0.15  # share of samples asked with a progress one stage off
NOTHING_RUNNING = 0.1  # share of finished frames asked with nothing running (after the operator's n)


def finished_at(episode: dict) -> int | None:
    """The first finished planner frame of an episode, None if none."""
    return next((k for k, y in enumerate(episode["labels"]) if y == 1), None)


def median_finish(episodes: list[dict]) -> dict[str, int]:
    """Per task: the median first finished planner frame of its success episodes (a failure's time scale)."""
    by_task = collections.defaultdict(list)
    for episode in episodes:
        frame = finished_at(episode)
        if episode["success"] and frame:
            by_task[episode["task"]].append(frame)
    return {task: int(np.median(frames)) for task, frames in by_task.items()}


def stage_at(episode: dict, k: int, medians: dict[str, int]) -> int:
    """The progress stage at planner frame k (< 0: the skill has not started)."""
    if k < 0:
        return 0
    frame = finished_at(episode)
    if frame is None:  # a failure (or a success with no finished frame): on the task's usual clock, never finished
        return min(pp.STAGES - 1, pp.progress_of(k, medians.get(episode["task"])))
    return pp.progress_of(k, frame)


def history_images(episode: dict, k: int, past, still: int | None) -> list[str]:
    """images_of with the history of a robot still for ``still`` s: those images are the current top camera."""
    from policy.longhorizon.build_planner_data import images_of

    images = images_of(episode, k, past)
    if still:
        for j, seconds in enumerate(past):
            if seconds <= still:
                images[len(pp.CAMERAS) + j] = images[0]
    return images


def mem_sample(
    sample: dict,
    episode: dict,
    medians: dict[str, int],
    rng: random.Random,
    *,
    lag=LAG_S,
    past=None,
    long_still: float = 0.0,
) -> dict:
    """One sample of a memory planner from a build_planner_data.py one (its frame and images; with ``past`` the
    images are rebuilt with that history, standing-still ones still 10 or 20 s with probability ``long_still``)."""
    k, task, finished = sample["frame"], episode["task"], bool(sample["finished"])
    chain, index = pp.any_chain(task, rng)
    before = stage_at(episode, k - rng.choice(lag), medians)
    if before < pp.STAGES and rng.random() < OFF_BY_ONE:
        before = min(pp.STAGES - 1, max(0, before + rng.choice((-1, 1))))
    running = task
    memory = pp.memory_text(chain[:index], before)
    if index == 0 and k == 0 and rng.random() < 0.5:
        running, memory = None, pp.memory_text([], 0)  # the start of the whole goal
    if finished:
        target_skill, target_memory = pp.next_skill(chain, index), pp.memory_text(chain[: index + 1], 0)
        if rng.random() < NOTHING_RUNNING:
            running, memory = None, target_memory
    else:
        target_skill = task
        target_memory = pp.memory_text(chain[:index], min(pp.STAGES - 1, stage_at(episode, k, medians)))
    out = {key: sample[key] for key in ("episode", "frame", "images") if key in sample}
    still = sample.get("still")
    if past is not None:
        if still and rng.random() < long_still:
            still = rng.choice((10, 20))
        out["images"] = history_images(episode, k, past, still)
    out.update(
        goal=pp.any_goal_text(chain, rng),
        chain=chain,
        index=index,
        running=running,
        memory=memory,
        target_skill=target_skill,
        target_memory=target_memory,
        finished=finished,
        **({"still": still} if still else {}),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--samples", default="samples_hard.jsonl", help="under --data")
    parser.add_argument("--out", default="samples_mem.jsonl", help="under --data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--past", default=None, help="rebuild the images with this top-camera history, e.g. 20,10,3,2,1"
    )
    parser.add_argument("--lag", default=",".join(str(v) for v in LAG_S), help="seconds the memory asked with lags")
    parser.add_argument("--long-still", type=float, default=0.0, help="share of standing-still samples still 10-20 s")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    data = Path(args.data).expanduser()
    episodes = [json.loads(line) for line in (data / "episodes.jsonl").read_text().splitlines() if line.strip()]
    by_dir = {e["dir"]: e for e in episodes}
    medians = median_finish([e for e in episodes if e["split"] == "train"])
    rng = random.Random(args.seed)
    samples = [json.loads(line) for line in (data / args.samples).read_text().splitlines() if line.strip()]
    past = tuple(int(v) for v in args.past.split(",")) if args.past else None
    lag = tuple(int(v) for v in args.lag.split(","))
    out = [mem_sample(s, by_dir[s["episode"]], medians, rng, lag=lag, past=past, long_still=args.long_still)
           for s in samples]  # fmt: skip
    if past is not None:  # the history frames the recordings were not cut into yet
        from policy.longhorizon import upstream_planner as up

        needed: dict[str, set[int]] = {}
        for s in out:
            for path in s["images"]:
                if not (data / path).is_file():
                    needed.setdefault(s["episode"], set()).add(int(Path(path).name[:4]))
        if needed:
            up.extract_needed(data, {name: sorted(f) for name, f in needed.items()}, ("cam_high",), args.workers)
        missing = [path for s in out for path in s["images"] if not (data / path).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images still missing, e.g. {missing[0]}")
    partial = data / (args.out + ".tmp")
    partial.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in out))
    partial.replace(data / args.out)

    lengths = collections.Counter(len(s["chain"]) for s in out)
    repeats = sum(len(set(s["chain"])) < len(s["chain"]) for s in out)
    stages = collections.Counter(pp.parse_memory(s["memory"])[1] for s in out)
    print(f"{len(out)} samples -> {data / args.out}")
    print(f"goal lengths {dict(sorted(lengths.items()))}, {repeats} with a repeated skill")
    print(f"progress asked with {dict(sorted(stages.items()))}; median finish frame per task {medians}")


if __name__ == "__main__":
    main()
