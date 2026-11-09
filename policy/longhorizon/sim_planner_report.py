"""Scores sim_planner_eval.py runs offline: the planner's first switch under any threshold and replies in a row,
recomputed from the logged asks (each with the running skill's probability and the greedy reply), against the
simulator's own success. The planner does not move the robot before it switches, so a run where it never switches
(--long_horizon_planner_confirm 999) scores every setting exactly.

    python policy/longhorizon/sim_planner_report.py <dir with <task>/episodes.jsonl> [--out report.md]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

THRESHOLDS = (None, 0.5, 0.7, 0.8, 0.9)
HZ = 10.0  # simulator control steps per second (t70kit10)


def switch_step(asks: list[dict], task: str, threshold: float | None, confirm: int) -> int | None:
    """The step of the confirm-th reply in a row voting to switch, or None."""
    run = 0
    for ask in asks:
        if threshold is None:
            vote = (ask.get("greedy") or ask.get("skill")) not in (None, task)
        else:
            vote = ask.get("p_run") is not None and ask["p_run"] < threshold
        run = run + 1 if vote else 0
        if run >= confirm:
            return ask["step"]
    return None


def outcome(first_success: int | None, switch: int | None) -> str:
    if first_success is None:
        return "false" if switch is not None else "quiet"
    if switch is None:
        return "missed"
    return "early" if switch < first_success else "caught"


def load(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/episodes.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if not record.get("steps"):  # an empty record a run wrote at exit (fixed in sim_planner_eval.py)
                    continue
                record["asks"] = [p for p in record.get("plans") or [] if p.get("running") == record["task"]]
                rows.append(record)
    return rows


def report(rows: list[dict]) -> str:
    tasks = sorted({r["task"] for r in rows})
    success = sum(r["first_success"] is not None for r in rows)
    lines = [f"仿真闭环：{len(rows)} 个回合，{len(tasks)} 个任务；世界模型在 {success} 个回合里做成（仿真判定）", "",  # noqa: RUF001
             "| 阈值 | 连续几次 | 做成的回合 抓到 | 提前切 | 漏掉 | 没做成的回合 误切 | 延迟中位数(秒) |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    best = {}
    for threshold in THRESHOLDS:
        for confirm in (1, 2, 3):
            c, delays = collections.Counter(), []
            for r in rows:
                s = switch_step(r["asks"], r["task"], threshold, confirm)
                o = outcome(r["first_success"], s)
                c[o] += 1
                if o == "caught":
                    delays.append((s - r["first_success"]) / HZ)
            n_ok, n_bad = c["caught"] + c["early"] + c["missed"], c["false"] + c["quiet"]
            name = "贪心" if threshold is None else threshold
            delay = sorted(delays)[len(delays) // 2] if delays else None
            lines.append(f"| {name} | {confirm} | {c['caught']}/{n_ok} | {c['early']} | {c['missed']} | "
                         f"{c['false']}/{n_bad} | {delay} |")  # fmt: skip
            best[(threshold, confirm)] = c
    lines += [
        "",
        "| 任务 | 回合 | 做成 | 阈值 0.8、连续 2 次 抓到 / 提前 / 漏掉 / 误切 | 抓到时的延迟(秒) |",
        "|---|---|---|---|---|",
    ]
    for task in tasks:
        group = [r for r in rows if r["task"] == task]
        c, delays = collections.Counter(), []
        for r in group:
            s = switch_step(r["asks"], task, 0.8, 2)
            o = outcome(r["first_success"], s)
            c[o] += 1
            if o == "caught":
                delays.append(round((s - r["first_success"]) / HZ, 1))
        done = sum(r["first_success"] is not None for r in group)
        lines.append(f"| {task} | {len(group)} | {done} | {c['caught']} / {c['early']} / {c['missed']} / {c['false']} "
                     f"| {delays} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    text = report(load(Path(args.root)))
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
