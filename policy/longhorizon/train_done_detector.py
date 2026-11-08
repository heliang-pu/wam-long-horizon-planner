"""Train the done detector's head (done_detector.DoneHead) on encoded recordings (build_done_features.py for the image
source, wam_features.py extract for Phoenix's own features), pick each task's threshold on held-out episodes, and
write a detector directory long_horizon.py loads.

Labels per encoded frame (frame_labels; 1 done, 0 not done, -1 not trained on):
- an episode that ended in success (the organizers' demonstrations, ours marked success, takeovers the operator
  finished): done from its success frame (success_moments.py) on, not done up to 1 s before it; where the rules find
  no success frame, done in its last pos_tail_s seconds and not done in its first neg_frac (a recording can go on long
  after the task is finished, so nothing later says "not done"); in between not trained on;
- a failed episode: not done anywhere, its end included (the hard negatives: the arms stopped, the task not finished).

Held out: --val-percent of the episodes of each recording (by a hash of the episode, so reruns hold out the same).
On them, the runtime rule (threshold, --confirm frames in a row, not before --min-steps) is replayed per episode:
a success episode is caught (fires where it is labelled done or not trained on), early (fires where it is labelled
not done) or missed; a failure episode is false if it fires at all. Each task's threshold is the lowest whose early +
false episodes stay within --max-false of its held-out episodes (the most caught), else the one with the fewest.

Rate: the head sees the seconds since the frame before. Phoenix features are encoded at 1 per second, while on the
robot one comes per chunk (10 or 20 steps at 10 Hz: 1 or 2 s); --subsample trains on every frame and every other one
(random offset), and the thresholds hold at every rate in --subsample.

    policy/fmc3_no1/.venv/bin/python policy/longhorizon/train_done_detector.py \
        --features ~/workspace/models/longhorizon/phoenix_features_<run>_<step> --out ~/workspace/models/longhorizon/done_<...>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.longhorizon.done_detector import STATE_DIM  # noqa: E402
from policy.longhorizon.done_detector import TASKS  # noqa: E402

THRESHOLDS = np.round(np.arange(0.05, 0.991, 0.01), 2)


def frame_labels(row: dict, frame_index: np.ndarray, pos_tail_s: float, neg_frac: float = 0.4) -> np.ndarray:
    """1 done / 0 not done / -1 not trained on, for an episode's encoded frames (raw frame indices)."""
    t = np.asarray(frame_index)
    if not row["success"]:
        return np.zeros(len(t), np.int8)
    fps, end = float(row["fps"]), int(row["length"]) - 1
    pos_from, neg_until = end - pos_tail_s * fps, neg_frac * end
    if row.get("success_frame") is not None:
        pos_from = min(pos_from, float(row["success_frame"]))
        neg_until = float(row["success_frame"]) - fps
    labels = np.full(len(t), -1, np.int8)
    labels[t < neg_until] = 0
    labels[t >= pos_from] = 1
    return labels


def held_out(row: dict, percent: float) -> bool:
    key = f"{row['dataset']}/{row['episode_index']}".encode()
    return int(hashlib.sha1(key).hexdigest()[:8], 16) % 1000 < percent * 10


def fire_frame(
    probs: np.ndarray,
    frame_index: np.ndarray,
    threshold: float,
    confirm: int,
    min_steps: int,
) -> int:
    """The runtime rule replayed: the first frame where the last ``confirm`` probabilities reach the threshold and
    at least ``min_steps`` steps have run (raw frames = control steps at 10 Hz); -1 if never."""
    run = 0
    for j, (p, t) in enumerate(zip(probs, frame_index, strict=True)):
        run = run + 1 if p >= threshold else 0
        if run >= confirm and t >= min_steps:
            return j
    return -1


def episode_outcome(probs, frame_index, labels, *, success: bool, threshold, confirm, min_steps) -> tuple[str, int]:
    j = fire_frame(probs, frame_index, threshold, confirm, min_steps)
    if not success:
        return ("false" if j >= 0 else "quiet"), j
    if j < 0:
        return "missed", j
    return ("early" if labels[j] == 0 else "caught"), j


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def features_meta(features_dir: Path) -> dict:
    """features.json (an image-source directory from before it existed: encoder.json, 2 frames per second)."""
    path = features_dir / "features.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"source": "image", "fps": 2.0, **json.loads((features_dir / "encoder.json").read_text())}


def load_episodes(features_dir: Path, tasks, pos_tail_s, neg_frac=0.4) -> list[dict]:
    episodes = []
    for line in (features_dir / "index.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["task"] not in tasks:
            continue
        with np.load(features_dir / row["file"]) as data:
            features, state, frame_index = data["features"], data["state"], data["frame_index"]
        if len(frame_index) == 0:
            continue
        episodes.append(
            {
                "row": row,
                "task": tasks.index(row["task"]),
                "features": features,  # float16 as stored: Phoenix's are 9 x 3072 per frame
                "state": state.astype(np.float32),
                "frame_index": frame_index,
                "labels": frame_labels(row, frame_index, pos_tail_s, neg_frac),
            }
        )
    return episodes


def view_of(episode: dict, factor: int = 1, offset: int = 0) -> dict:
    """Every factor-th encoded frame from offset (as a slower chunk rate sees the episode), with dt (seconds since the
    frame before; 0 for the first)."""
    keep = slice(offset, None, factor)
    frame_index = episode["frame_index"][keep]
    dt = np.diff(frame_index, prepend=frame_index[:1]).astype(np.float32) / float(episode["row"]["fps"])
    return {
        "task": episode["task"],
        "features": episode["features"][keep],
        "state": episode["state"][keep],
        "frame_index": frame_index,
        "labels": episode["labels"][keep],
        "dt": dt,
    }


def batch_tensors(batch: list[dict], device):
    import torch

    n = max(len(e["frame_index"]) for e in batch)
    features = np.zeros((len(batch), n, *batch[0]["features"].shape[1:]), np.float32)
    state = np.zeros((len(batch), n, STATE_DIM), np.float32)
    dt = np.zeros((len(batch), n), np.float32)
    labels = np.full((len(batch), n), -1, np.int64)
    for b, e in enumerate(batch):
        k = len(e["frame_index"])
        features[b, :k], state[b, :k], dt[b, :k], labels[b, :k] = e["features"], e["state"], e["dt"], e["labels"]
    task = np.array([e["task"] for e in batch])
    to = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
    return to(features), to(state), to(dt), to(labels), to(task)


def predict(head, views: list[dict], device, batch: int = 32) -> list[np.ndarray]:
    import torch

    head.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(views), batch):
            chunk = views[start : start + batch]
            features, state, dt, _labels, task = batch_tensors(chunk, device)
            logits, _ = head(features, state, dt)
            probs = torch.sigmoid(logits[torch.arange(len(chunk), device=device), :, task]).cpu().numpy()
            out += [probs[b, : len(e["frame_index"])] for b, e in enumerate(chunk)]
    return out


def normalization(train: list[dict]) -> tuple[np.ndarray, ...]:
    """Per-dimension mean and std of the features and of qpos over the training frames (accumulated per episode)."""
    total, total_sq, count = 0.0, 0.0, 0
    for e in train:
        x = e["features"].astype(np.float64)
        total, total_sq, count = total + x.sum(0), total_sq + (x * x).sum(0), count + len(x)
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 0.0)) + 1e-6
    states = np.concatenate([e["state"] for e in train])
    return mean.astype(np.float32), std.astype(np.float32), states.mean(0), states.std(0) + 1e-3


def choose_threshold(rows: list[tuple], confirm: int, min_steps: int, max_false: float) -> dict:
    """rows: (probs, frame_index, labels, success, fps) of one task's held-out episodes."""
    table = []
    for threshold in THRESHOLDS:
        counts = {"caught": 0, "early": 0, "missed": 0, "false": 0, "quiet": 0}
        delays = []
        for probs, frame_index, labels, success, fps in rows:
            outcome, j = episode_outcome(
                probs, frame_index, labels, success=success, threshold=threshold, confirm=confirm, min_steps=min_steps
            )
            counts[outcome] += 1
            if outcome == "caught":
                pos = np.flatnonzero(labels == 1)
                if len(pos):
                    delays.append(float(frame_index[j] - frame_index[pos[0]]) / fps)
        n_success = counts["caught"] + counts["early"] + counts["missed"]
        n_fail = counts["false"] + counts["quiet"]
        table.append(
            {
                "threshold": float(threshold),
                **counts,
                "recall": counts["caught"] / n_success if n_success else None,
                "false_rate": (counts["early"] + counts["false"]) / max(1, n_success + n_fail),
                "median_delay_s": float(np.median(delays)) if delays else None,
            }
        )
    ok = [t for t in table if t["false_rate"] <= max_false]
    best = ok[0] if ok else min(table, key=lambda t: (t["false_rate"], -(t["recall"] or 0)))
    return {**best, "met_max_false": bool(ok)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, help="build_done_features.py / wam_features.py extract output")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--pos-tail-s", type=float, default=1.5)
    parser.add_argument(
        "--neg-frac", type=float, default=0.4, help="not done in this first share, no success frame known"
    )
    parser.add_argument("--val-percent", type=float, default=15.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--camera-drop", type=float, default=0.1, help="per episode and view, its features dropped")
    parser.add_argument("--feature-noise", type=float, default=0.1, help="Gaussian noise, in feature std units")
    parser.add_argument(
        "--subsample",
        type=int,
        nargs="+",
        default=None,
        help="frame-rate factors trained and evaluated (default: 1 2 for phoenix, a chunk of 10 or 20 steps; 1 for image)",
    )
    parser.add_argument("--confirm", type=int, default=2)
    parser.add_argument("--min-steps", type=int, default=30)
    parser.add_argument("--max-false", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    from policy.longhorizon.done_detector import DoneHead
    from policy.longhorizon.done_detector import resolve_device

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    features_dir = Path(args.features).expanduser()
    meta = features_meta(features_dir)
    source = meta["source"]
    factors = args.subsample or ([1, 2] if source == "phoenix" else [1])
    tasks = [t for t in TASKS if t in args.tasks]
    episodes = load_episodes(features_dir, tasks, args.pos_tail_s, args.neg_frac)
    train = [e for e in episodes if not held_out(e["row"], args.val_percent)]
    val = [e for e in episodes if held_out(e["row"], args.val_percent)]
    if not train or not val:
        raise SystemExit(f"{len(train)} training and {len(val)} held-out episodes: need both")
    n_views, feature_dim = (int(v) for v in train[0]["features"].shape[1:])
    print(f"{source} features {n_views} x {feature_dim}: {len(episodes)} episodes ({len(train)} train, "
          f"{len(val)} held out), {len(tasks)} tasks, rate factors {factors}", flush=True)  # fmt: skip

    head = DoneHead(feature_dim=feature_dim, n_cameras=n_views, hidden=args.hidden, n_tasks=len(tasks),
                    dropout=args.dropout)  # fmt: skip
    head.set_normalization(*normalization(train))
    head.to(device)
    pos = np.zeros(len(tasks))
    neg = np.zeros(len(tasks))
    for e in train:
        pos[e["task"]] += (e["labels"] == 1).sum()
        neg[e["task"]] += (e["labels"] == 0).sum()
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 30), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total_steps = args.epochs * steps_per_epoch
    warmup = max(1, total_steps // 10)  # linear warmup over the first 10 %, then cosine to 0

    def lr_factor(step: int) -> float:
        return min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    def loss_of(batch, *, train_mode: bool):
        features, state, dt, labels, task = batch_tensors(batch, device)
        if train_mode:
            drop = torch.rand(features.shape[0], 1, features.shape[2], 1, device=device) < args.camera_drop
            features = torch.where(drop, head.feature_mean.expand_as(features), features)
            features = features + torch.randn_like(features) * head.feature_std * args.feature_noise
        logits, _ = head(features, state, dt)
        logits = logits[torch.arange(len(batch), device=device), :, task]  # (B, T): each episode's own task
        mask = labels >= 0
        weight = torch.where(labels == 1, pos_weight[task][:, None], torch.ones_like(logits))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[mask], labels[mask].float(), weight=weight[mask], reduction="sum"
        )
        return loss / weight[mask].sum().clamp(min=1)

    val_views = [view_of(e, factor) for factor in factors for e in val]
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = rng.permutation(len(train))
        total = 0.0
        for start in range(0, len(order), args.batch):
            batch = []
            for i in order[start : start + args.batch]:
                factor = int(rng.choice(factors))
                batch.append(view_of(train[i], factor, int(rng.integers(factor))))
            loss = loss_of(batch, train_mode=True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss)
        if epoch % 5 == 0 or epoch == args.epochs:
            head.eval()
            with torch.no_grad():
                val_loss = np.mean(
                    [float(loss_of(val_views[i : i + 32], train_mode=False)) for i in range(0, len(val_views), 32)]
                )
            print(f"epoch {epoch}: train loss {total / steps_per_epoch:.4f}, held-out loss {val_loss:.4f}", flush=True)

    probs = predict(head, val_views, device)
    report = {}
    for k, task in enumerate(tasks):
        idx = [i for i, v in enumerate(val_views) if v["task"] == k]
        if not idx:
            continue
        success = {i: bool(val[i % len(val)]["row"]["success"]) for i in idx}
        fps = {i: float(val[i % len(val)]["row"]["fps"]) for i in idx}
        rows = [(probs[i], val_views[i]["frame_index"], val_views[i]["labels"], success[i], fps[i]) for i in idx]
        scores = np.concatenate([probs[i] for i in idx])
        labels = np.concatenate([val_views[i]["labels"] for i in idx])
        n_episodes = len(idx) // len(factors)
        report[task] = {
            "held_out_success": sum(r[3] for r in rows) // len(factors),
            "held_out_failure": (len(rows) - sum(r[3] for r in rows)) // len(factors),
            "frame_auroc": auroc(scores, labels),
            **choose_threshold(rows, args.confirm, args.min_steps, args.max_false),
            "replays": n_episodes * len(factors),
        }

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if source == "image":
        encoder_dir = Path(json.loads((features_dir / "encoder.json").read_text())["encoder_dir"])
        for name in ("encoder.pt", "encoder.json"):
            shutil.copy2(encoder_dir / name, out / name)
    else:
        (out / "features.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    torch.save({k: v.cpu() for k, v in head.state_dict().items()}, out / "head.pt")
    config = {
        "source": source,
        "tasks": tasks,
        "head": {
            "feature_dim": feature_dim,
            "n_cameras": n_views,
            "state_dim": STATE_DIM,
            "hidden": args.hidden,
            "n_tasks": len(tasks),
            "dropout": args.dropout,
        },
        "fps": float(meta["fps"]),
        "subsample": factors,
        "thresholds": {task: r["threshold"] for task, r in report.items()},
        "default_threshold": 0.5,
        "confirm": args.confirm,
        "min_steps": args.min_steps,
        "labels": {"pos_tail_s": args.pos_tail_s, "neg_frac": args.neg_frac},
        "features": str(features_dir),
        "episodes": {"train": len(train), "held_out": len(val)},
        "args": vars(args),
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    rates = "、".join(f"{float(meta['fps']) / f:g}" for f in factors)
    lines = [
        f"# 完成判别器 {out.name}（{source} 特征）",  # noqa: RUF001
        "",
        f"训练 {len(train)} 组、留出 {len(val)} 组；判定规则：概率 ≥ 阈值连续 {args.confirm} 帧，"  # noqa: RUF001
        f"子任务开始 {args.min_steps} 步之后。留出组按每秒 {rates} 帧各重放一遍，"  # noqa: RUF001
        f"阈值取误判（成功组提前 + 失败组误触发）≤ {args.max_false:.0%} 里最低的。",  # noqa: RUF001
        "",
        "| 任务 | 留出成功 | 留出失败 | 帧 AUROC | 阈值 | 抓到 | 提前 | 漏掉 | 失败误触发 | 误判率 | 延迟中位数(秒) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for task, r in report.items():
        auc = f"{r['frame_auroc']:.3f}" if r["frame_auroc"] is not None else "—"
        delay = f"{r['median_delay_s']:+.1f}" if r["median_delay_s"] is not None else "—"
        flag = "" if r["met_max_false"] else "（未达标）"  # noqa: RUF001
        lines.append(
            f"| {task} | {r['held_out_success']} | {r['held_out_failure']} | {auc} | {r['threshold']:.2f}{flag} | "
            f"{r['caught']} | {r['early']} | {r['missed']} | {r['false']} | {r['false_rate']:.0%} | {delay} |"
        )
    lines += [
        "",
        "抓到 / 提前 / 漏掉 / 失败误触发按重放次数计（每组每个速率一次）。"  # noqa: RUF001
        "延迟：触发帧减第一个标为完成的帧，负数表示在成功组末尾那段之前、不训练的区间里就触发了。",  # noqa: RUF001
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
