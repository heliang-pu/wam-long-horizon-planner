"""Encode the real recordings for the done detector (image source): every episode's frames at 2 per second, three
cameras each. The phoenix source goes through the same episode list and files (wam_features.py extract).

Sources (LeRobot v2.1, 10 fps), all ten tasks by default:
- the organizers' demonstrations, Real/cobotmagic_Real_<task> (all successes; 32-D state -> [0:7] + [16:23]);
- our recordings, real_rollouts/<task>/<recording>/ with episode_success.json (successes, failures, takeovers,
  直接采集); episodes marked discarded are skipped. Recordings being written (.<name>.lock) are skipped.

Per episode one npz: features (n, C, D) float16 (image: C = the cameras in done_detector.CAMERAS order; phoenix: layers x
cameras), state (n, 14), frame_index (n,); one line in <out>/index.jsonl with what the labels are made of
(train_done_detector.py makes them, so they can change without encoding again); <out>/features.json says what the
features are. A rerun skips episodes already encoded and refreshes the labels of the others.

    policy/fmc3_no1/.venv/bin/python policy/longhorizon/build_done_features.py \
        --encoder ~/workspace/models/longhorizon/encoder_dinov3s224 --out ~/workspace/models/longhorizon/features_20261030

Decoding and resizing run in --workers processes (CPU), the encoder on --device (cuda: about 1 GB).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.longhorizon.done_detector import CAMERAS  # noqa: E402
from policy.longhorizon.done_detector import STATE_DIM  # noqa: E402
from policy.longhorizon.done_detector import TASKS  # noqa: E402

OFFICIAL_STATE = [*range(7), *range(16, 23)]  # the organizers' 32-D layout -> our 14-D
DEFAULT_SUCCESS_FRAMES = REPO_ROOT / "audit_reports" / "success_moments_20261106" / "成功时刻.csv"  # success_moments.py
# Run with one task's prompt on another task's scene (11-04), labelled by what the scene's task did: never training data.
WRONG_SCENE_UIDS = {
    "186fa8cb96614e8ba7034ed69bd8c80a",  # item_assembly, basket scene
    "22a3ddf8f36848f788ba7fb6fa5994ca",  # drawer_open_place, basket scene
    "2375418ca7ef40d298476e76fccef7b5",  # manipulate_pipette, rack scene
}


def official_datasets(root: Path, tasks) -> list[dict]:
    out = []
    for task in tasks:
        path = root / f"cobotmagic_Real_{task}"
        if (path / "meta" / "info.json").is_file():
            out.append(
                {
                    "task": task,
                    "source": "official",
                    "recording": path.name,
                    "path": path,
                }
            )
    return out


def being_recorded(path: Path) -> bool:
    """A run_real recorder holds .<name>.lock (rollout_v21._lock_dataset); the file itself stays after it closes."""
    lock = path.parent / f".{path.name}.lock"
    if not lock.exists():
        return False
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
        except BlockingIOError:
            return True
    return False


def our_datasets(root: Path, tasks) -> list[dict]:
    out = []
    for task in tasks:
        for path in sorted((root / task).glob("*")):
            if path.name.startswith(".") or not (path / "episode_success.json").is_file():
                continue
            if being_recorded(path):
                print(f"skip {path}: being recorded", flush=True)
                continue
            out.append({"task": task, "source": "ours", "recording": path.name, "path": path})
    return out


def dataset_key(path: Path) -> str:
    """real_rollouts/<task>/<recording> -> <task>/<recording>; Real/cobotmagic_Real_<task> -> Real/cobotmagic_Real_<task>."""
    return f"{path.parent.name}/{path.name}"


def load_success_frames(csv_path: Path | None) -> dict[tuple[str, int], int]:
    """(dataset_key, episode index) -> first success frame, from a success-moment audit CSV (成功时刻.csv; its episode
    numbers are those of the day it was made: no recording it names was purged after 10-29 05:46)."""
    if csv_path is None or not csv_path.is_file():
        return {}
    frames = {}
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            frame = row.get("成功帧(原视频,10fps)") or ""
            if not frame.strip():
                continue
            frames[
                (
                    dataset_key(Path(str(row["原始数据"]).split()[-1])),
                    int(row["原始组号"]),
                )
            ] = int(float(frame))
    return frames


def episodes_of(dataset: dict, success_frames: dict) -> list[dict]:
    """One row per episode to encode: where it is and what its labels are made of."""
    path = dataset["path"]
    info = json.loads((path / "meta" / "info.json").read_text())
    lengths = {}
    with (path / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            lengths[int(row["episode_index"])] = int(row["length"])
    labels = {}
    if dataset["source"] == "ours":
        for row in json.loads((path / "episode_success.json").read_text())["episodes"]:
            labels[int(row["episode_index"])] = row
    rows = []
    chunk_size = int(info.get("chunks_size", 1000))
    for index, length in sorted(lengths.items()):
        label = labels.get(index, {})
        if dataset["source"] == "ours" and (not label or label.get("discarded")):
            continue
        if label.get("uid") in WRONG_SCENE_UIDS:
            continue
        success = True if dataset["source"] == "official" else bool(label.get("success"))
        success_frame = success_frames.get((dataset_key(path), index)) if success else None
        chunk = f"chunk-{index // chunk_size:03d}"
        rows.append(
            {
                "task": dataset["task"],
                "source": dataset["source"],
                "recording": dataset["recording"],
                "dataset": str(path),
                "episode_index": index,
                "length": length,
                "fps": float(info["fps"]),
                "parquet": str(path / "data" / chunk / f"episode_{index:06d}.parquet"),
                "videos": {
                    cam: str(path / "videos" / chunk / f"observation.images.{cam}" / f"episode_{index:06d}.mp4")
                    for cam in CAMERAS
                },
                "uid": label.get("uid"),
                "success": success,
                "stopped_by_operator": bool(label.get("stopped_by_operator")),
                "truncated": bool(label.get("truncated")),
                "human_steps": int(label.get("human_steps") or 0),
                "intervention_start": label.get("intervention_start"),
                "teleop": bool(label.get("teleop")),
                "success_frame": success_frame if success_frame is not None and success_frame < length else None,
            }
        )
    return rows


def refresh_index(index_path: Path, rows: list[dict], scanned: set[str]) -> list[dict]:
    """Rewrite index.jsonl with today's labels for the episodes encoded before; one of a scanned dataset that is gone,
    discarded, or another episode now (its length or uid changed: a purge renumbered it) is dropped, to be encoded
    again. Rows of datasets not scanned this time stay as they are."""
    if not index_path.is_file():
        return []
    fresh = {row["file"]: row for row in rows}
    kept = []
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        old = json.loads(line)
        if old["dataset"] not in scanned:
            kept.append(old)
            continue
        new = fresh.get(old["file"])
        if new is not None and new["length"] == old["length"] and new.get("uid") == old.get("uid"):
            kept.append({**new, "stride": old["stride"], "n": old["n"]})
    tmp = index_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
    tmp.replace(index_path)
    return kept


def decode_frames(row: dict, stride: int):
    """The episode's every stride-th frame: ({camera: [HWC uint8 RGB, ...]}, state (n, 14), frame indices)."""
    import av
    import pandas as pd

    table = pd.read_parquet(row["parquet"], columns=["observation.state", "frame_index"])
    state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
    if state.shape[1] != STATE_DIM:
        state = state[:, OFFICIAL_STATE]
    n = min(len(state), int(row["length"]))
    keep = list(range(0, n, stride))
    views = {}
    for cam in CAMERAS:
        frames = []
        with av.open(row["videos"][cam]) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for i, frame in enumerate(container.decode(stream)):
                if i >= n:
                    break
                if i % stride == 0:
                    frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) < len(keep):  # a video a few frames short of the parquet: the last frame stands in
            frames += [frames[-1]] * (len(keep) - len(frames))
        views[cam] = frames[: len(keep)]
    return views, state[keep], np.asarray(keep, dtype=np.int32)


def decode_episode(row: dict, stride: int, size: int, mean, std):
    """Worker of the image source: decode_frames, preprocessed for the encoder (float16)."""
    import torch

    from policy.longhorizon.done_detector import preprocess

    torch.set_num_threads(2)
    views, state, keep = decode_frames(row, stride)
    pixels = [preprocess(torch.from_numpy(np.stack(views[cam])), size, mean, std).half() for cam in CAMERAS]
    return row, torch.stack(pixels, dim=1).numpy(), state, keep  # (n_keep, cameras, 3, S, S)


def write_meta(out: Path, meta: dict) -> None:
    """features.json: what the features are. A directory holds one source (and checkpoint and layers) only."""
    path = out / "features.json"
    if path.is_file():
        old = json.loads(path.read_text())
        keys = ("source", "checkpoint", "layers", "model", "image_size", "fps")
        if any(old.get(key) != meta.get(key) for key in keys):
            raise ValueError(f"{out} holds other features ({ {k: old.get(k) for k in keys} }): use another --out")
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def collect_rows(
    out: Path,
    tasks,
    *,
    official: bool,
    ours: bool,
    limit: int,
    official_root=None,
    rollouts_root=None,
    success_frames_csv=DEFAULT_SUCCESS_FRAMES,
) -> tuple[list[dict], list[dict]]:
    """(episodes encoded before, with today's labels; episodes to encode) for --out."""
    datasets = []
    if official:
        datasets += official_datasets(Path(official_root or REPO_ROOT / "Real"), tasks)
    if ours:
        datasets += our_datasets(Path(rollouts_root or REPO_ROOT / "real_rollouts"), tasks)
    success_frames = load_success_frames(Path(success_frames_csv) if success_frames_csv else None)
    rows, scanned = [], set()
    for dataset in datasets:
        episodes = episodes_of(dataset, success_frames)
        scanned.add(str(dataset["path"]))
        for row in episodes:
            row["file"] = f"{row['task']}/{row['source']}/{row['recording']}/episode_{row['episode_index']:06d}.npz"
        rows += episodes
    kept = refresh_index(out / "index.jsonl", rows, scanned)
    done = {row["file"] for row in kept}
    todo = [row for row in rows if row["file"] not in done]
    if limit:  # smoke tests: this run encodes the first --limit episodes of each dataset
        counts: dict[str, int] = {}
        limited = []
        for row in todo:
            counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
            if counts[row["dataset"]] <= limit:
                limited.append(row)
        todo = limited
    print(f"{len(datasets)} datasets, {len(rows)} episodes, {len(kept)} encoded before, {len(todo)} to encode -> {out}",
          flush=True)  # fmt: skip
    return kept, todo


def write_episode(out: Path, index, row: dict, features: np.ndarray, state: np.ndarray, keep: np.ndarray) -> int:
    """One encoded episode: its npz and its index line; returns the frames."""
    path = out / row["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features.astype(np.float16), state=state, frame_index=keep)
    row = {**row, "stride": int(keep[1] - keep[0]) if len(keep) > 1 else 1, "n": len(keep)}
    index.write(json.dumps(row, ensure_ascii=False) + "\n")
    index.flush()
    return len(keep)


def run_extraction(
    out, meta: dict, encode, *, tasks=None, fps: float, limit: int = 0, official=True, ours=True
) -> None:
    """The loop of a feature source that encodes in this process (wam_features.py extract): encode(row, stride) ->
    (features (n, C, D), state (n, 14), frame indices (n,))."""
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_meta(out, meta)
    _kept, todo = collect_rows(out, list(tasks or TASKS), official=official, ours=ours, limit=limit)
    started, frames_done = time.monotonic(), 0
    with (out / "index.jsonl").open("a") as index:
        for k, row in enumerate(todo, 1):
            try:
                features, state, keep = encode(row, max(1, round(row["fps"] / fps)))
            except Exception as exc:  # a broken video: reported, the rest goes on
                print(f"FAILED {row['file']}: {exc!r}", flush=True)
                continue
            frames_done += write_episode(out, index, row, features, state, keep)
            if k % 20 == 0 or k == len(todo):
                rate = frames_done / max(1e-6, time.monotonic() - started)
                print(f"{k}/{len(todo)} episodes, {frames_done} frames ({rate:.1f} frames/s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", required=True, help="export_encoder.py output dir (encoder.pt, encoder.json)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--official-root", default=str(REPO_ROOT / "Real"))
    parser.add_argument("--rollouts-root", default=str(REPO_ROOT / "real_rollouts"))
    parser.add_argument("--no-official", action="store_true")
    parser.add_argument("--no-ours", action="store_true")
    parser.add_argument(
        "--success-frames",
        default=str(DEFAULT_SUCCESS_FRAMES),
        help="success-moment audit CSV: those episodes get their positives from the success frame on",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="frames per second kept (the detector's rate)")
    parser.add_argument("--limit", type=int, default=0, help="encode at most this many episodes per dataset this run")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    from policy.longhorizon.done_detector import resolve_device

    encoder_dir = Path(args.encoder).expanduser()
    meta = json.loads((encoder_dir / "encoder.json").read_text())
    device = resolve_device(args.device)
    encoder = torch.jit.load(str(encoder_dir / "encoder.pt"), map_location=device).eval()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_meta(out, {"source": "image", **meta, "encoder_dir": str(encoder_dir), "fps": args.fps})
    (out / "encoder.json").write_text(json.dumps({**meta, "encoder_dir": str(encoder_dir)}, indent=2) + "\n")
    _kept, todo = collect_rows(
        out, args.tasks, official=not args.no_official, ours=not args.no_ours, limit=args.limit,
        official_root=args.official_root, rollouts_root=args.rollouts_root, success_frames_csv=args.success_frames,
    )  # fmt: skip

    started, frames_done, k = time.monotonic(), 0, 0
    context = mp.get_context("spawn")
    queue, running = list(reversed(todo)), set()
    with cf.ProcessPoolExecutor(args.workers, mp_context=context) as pool, (out / "index.jsonl").open("a") as index:
        while queue or running:
            while queue and len(running) < 2 * args.workers:  # decoded episodes waiting for the encoder stay few
                row = queue.pop()
                stride = max(1, round(row["fps"] / args.fps))
                running.add(pool.submit(decode_episode, row, stride, meta["image_size"], meta["mean"], meta["std"]))
            finished, running = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for future in finished:
                k += 1
                try:
                    row, pixels, state, keep = future.result()
                except Exception as exc:  # a broken video: reported, the rest goes on
                    print(f"FAILED: {exc!r}", flush=True)
                    continue
                flat = torch.from_numpy(pixels).reshape(-1, *pixels.shape[2:])
                with torch.no_grad():
                    chunks = [encoder(flat[i : i + 192].to(device).float()).cpu() for i in range(0, len(flat), 192)]
                features = torch.cat(chunks).reshape(pixels.shape[0], pixels.shape[1], -1).numpy()
                frames_done += write_episode(out, index, row, features, state, keep)
                if k % 50 == 0 or k == len(todo):
                    rate = frames_done / max(1e-6, time.monotonic() - started)
                    print(f"{k}/{len(todo)} episodes, {frames_done} frames ({rate:.0f} frames/s)", flush=True)


if __name__ == "__main__":
    main()
