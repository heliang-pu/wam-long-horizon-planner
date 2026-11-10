"""Replay held-out episodes through the high-level planner, as on the robot, and score its subtask switches.

Each held-out episode (build_planner_data.py, all its frames kept) is one skill of a random goal (seeded by the
episode, so every planner sees the same goals). The planner is asked every --every seconds with that frame's six images,
the skill running and the memory before it; a switch is --confirm replies in a row naming another skill (or done),
as long_horizon.py does. Outcomes as the done detector's (train_done_detector.episode_outcome):
- a success episode: caught (switched where the skill is finished or not trained on), early (switched where it is
  not finished) or missed; a caught switch is also checked for going to the right next skill and writing the memory;
- a failure episode: false if it switched at all, else quiet.

    ../venvs/planner/bin/python policy/longhorizon/eval_planner.py --data <planner data> --adapter <train_planner out>/adapter
    ../venvs/planner/bin/python policy/longhorizon/eval_planner.py --data <planner data> --adapter base   # zero-shot

Several GPUs: one process per --shard k/n (CUDA_VISIBLE_DEVICES=k), which writes its own replays, then once with --merge n
for the report.

--sweep asks every --ask-every seconds (a chunk on the robot) without stopping at a switch, and goes on for HOLD_S seconds
after the recording ends as if the arms stood still there (after that all six images are the last frame, so the answer
stays). <out>/sweep.md then scores multiples of that rate and any number of replies in a row from those answers, and counts
a switch before a takeover as a false one: the model was failing there. Recordings that stop soon after the success can be
caught this way too.
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
from policy.longhorizon.build_planner_data import chain_with  # noqa: E402
from policy.longhorizon.build_planner_data import images_of  # noqa: E402


def replay(planner, data: Path, episode: dict, *, every: int, confirm: int, seed: int, states=None) -> dict:
    """One held-out episode through the planner; states: <data>/states.json, for a planner shown the hands."""
    from PIL import Image

    rng = random.Random(f"{seed}/{episode['dir']}")
    chain, index = chain_with(episode["task"], rng)
    goal, running = pp.goal_text(chain, rng), episode["task"]
    memory = pp.memory_text(chain[:index])
    run, switch, replies, seconds = 0, None, [], []
    for k in range(0, len(episode["frame_index"]), every):
        images = [Image.open(data / path).convert("RGB") for path in images_of(episode, k)]
        rows = states.get(episode["dir"]) if states else None
        state = pp.hands_text(rows, k) if rows else None
        skill, new_memory, reply, took = planner.plan(images, goal, running, memory, state=state)
        replies.append({"frame": k, "reply": reply})
        seconds.append(took)
        run = run + 1 if skill is not None and skill != running else 0
        if run >= confirm:
            switch = {"frame": k, "skill": skill, "memory": new_memory}
            break
    labels = episode["labels"]
    if not episode["success"]:
        outcome = "false" if switch else "quiet"
    elif switch is None:
        outcome = "missed"
    else:
        outcome = "early" if labels[switch["frame"]] == 0 else "caught"
    result = {
        "episode": episode["dir"],
        "task": episode["task"],
        "chain": chain,
        "index": index,
        "goal": goal,
        "outcome": outcome,
        "switch": switch,
        "replies": replies,
        "mean_s": float(np.mean(seconds)) if seconds else None,
        "invalid": sum(pp.parse_reply(r["reply"])[0] is None for r in replies),
    }
    if outcome == "caught":
        result["next_ok"] = switch["skill"] == pp.next_skill(chain, index)
        result["memory_ok"] = switch["memory"] == pp.memory_text(chain[: index + 1])
        first = next((j for j, y in enumerate(labels) if y == 1), None)
        if first is not None:
            fi = episode["frame_index"]
            result["delay_s"] = (fi[switch["frame"]] - fi[first]) / 10.0
    return result


HOLD_S = 3  # the seconds until the images of a robot standing still no longer change (planner_prompt.PAST_S)


def held_images(episode: dict, k: int, past=pp.PAST_S) -> list[str]:
    """images_of, with frames after the end of the recording standing still at its last frame."""
    last = len(episode["frame_index"]) - 1
    base = f"frames/{episode['dir']}"
    now = [f"{base}/{min(k, last):04d}_{cam}.jpg" for cam in pp.CAMERAS]
    return now + [f"{base}/{min(max(0, k - s), last):04d}_cam_high.jpg" for s in past]


MAX_STILL_ASKS = 10  # asks with every image frozen at the end of a recording, at most


def scored(planner, images, goal: str, running: str | None, memory: str, state=None) -> tuple:
    """(skill, memory, p of the running skill, the likeliest other skill); the last two None for a planner without
    plan_scored or a reply that does not start with "skill:"."""
    if not hasattr(planner, "plan_scored"):
        skill, new_memory, _reply, _took = planner.plan(images, goal, running, memory, state=state)
        return skill, new_memory, None, None
    result = planner.plan_scored(images, goal, running, memory, state=state)
    probs = result["probs"]
    if not probs or running not in probs:
        return result["skill"], result["memory"], None, None
    other = max((name for name in probs if name != running), key=probs.get)
    return result["skill"], result["memory"], probs[running], other


def decide(skill, p_run, other, running, threshold: float | None):
    """The skill an ask names: the greedy reply's, or with a threshold the running one while its p is at least that."""
    if threshold is None or p_run is None:
        return skill
    return running if p_run >= threshold else other


def answers_of(planner, data: Path, episode: dict, *, seed: int, ask_every: int, states=None) -> dict:
    """The planner's skill every ``ask_every`` frames of a held-out episode, seeded as replay(), then on with every
    image frozen at its end (a robot standing still) until the input stops changing: from there the answer holds."""
    from PIL import Image

    rng = random.Random(f"{seed}/{episode['dir']}")
    chain, index = chain_with(episode["task"], rng)
    goal, running = pp.goal_text(chain, rng), episode["task"]
    memory = pp.MemoryTracker(progress=getattr(planner, "uses_progress", False), finished=chain[:index])
    n = len(episode["frame_index"])
    past = getattr(planner, "past", pp.PAST_S)
    rows = states.get(episode["dir"]) if states else None
    skills, p_runs, others, k, still, last = [], [], [], 0, 0, None
    while True:
        frozen = k >= n + max(past)  # all images are the last frame from here
        text = memory.text()
        if frozen and (text == last or still >= MAX_STILL_ASKS):
            break
        images = [Image.open(data / path).convert("RGB") for path in held_images(episode, k, past)]
        state = pp.hands_text(rows + [rows[-1]] * (k + 1), k) if rows else None
        skill, new_memory, p_run, other = scored(planner, images, goal, running, text, state)
        memory.observe(new_memory)
        skills.append(skill)
        p_runs.append(p_run)
        others.append(other)
        if frozen:
            still, last = still + 1, text
        k += ask_every
    return {
        "episode": episode["dir"],
        "task": episode["task"],
        "next": pp.next_skill(chain, index),
        "ask_every": ask_every,
        "skills": skills,
        "p_run": p_runs,
        "other": others,
    }


THRESHOLDS = (None, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)


def threshold_report(records: list[dict], episodes: dict, takeovers: dict, rate: int) -> str:
    """sweep_report at the rate asked, for each threshold on the running skill's p (None: the greedy replies)."""
    lines = [
        "| 阈值 | 连续几次 | 成功回合抓到 | 提前切 | 模型没做成时误切 | 延迟中位数(秒) |",
        "|---|---|---|---|---|---|",
    ]
    n_bad = sum(not episodes[r["episode"]]["success"] for r in records) + sum(
        r["episode"] in takeovers for r in records
    )
    for threshold in THRESHOLDS:
        decided = [{**r, "skills": [decide(sk, p, o, r["task"], threshold) for sk, p, o in
                                    zip(r["skills"], r.get("p_run") or [None] * len(r["skills"]),
                                        r.get("other") or [None] * len(r["skills"]), strict=True)]}
                   for r in records]  # fmt: skip
        table, _text = sweep_report(decided, episodes, takeovers, rates=(rate,))
        for row in table[:3]:
            c = collections.Counter(row["counts"])
            ok = c["caught"] + c["early"] + c["missed"] + c["pretakeover"]
            bad = c["false"] + c["pretakeover"]
            name = "贪心" if threshold is None else threshold
            lines.append(f"| {name} | {row['confirm']} | {c['caught']}/{ok} ({c['caught'] / max(1, ok):.0%}) | "
                         f"{c['early']} ({c['early'] / max(1, ok):.0%}) | {bad}/{n_bad} ({bad / max(1, n_bad):.0%}) "
                         f"| {row['delay_s']} |")  # fmt: skip
    return "\n".join(lines) + "\n"


CHAIN_LENGTHS = (2, 3, 4, 6, 10)


def chain_specs(held: list[dict], seed: int, per_length: int, lengths=CHAIN_LENGTHS, goal_style: str = "named",
                listing: bool = False, hints: bool = False) -> list[dict]:
    """Random goals of each length (any skills, repeats allowed, not twice in a row), each skill one held-out success
    episode of it; the same for every planner with the same seed."""
    rng = random.Random(f"{seed}/chains")
    pool = {t: [e["dir"] for e in held if e["task"] == t and e["success"] and 1 in e["labels"]] for t in pp.SKILLS}
    skills = [t for t in pp.SKILLS if pool[t]]
    specs = []
    if goal_style == "count":  # G5: repeats in a row (按三次铃), goals with counts in held-out wordings; own stream
        crng = random.Random(f"{seed}/count")
        for length in lengths:
            for _ in range(per_length):
                chain = [crng.choice(skills)]
                while len(chain) < length:
                    chain.append(chain[-1] if crng.random() < 0.4 else crng.choice([t for t in skills if t != chain[-1]]))
                specs.append({"chain": chain, "episodes": [crng.choice(pool[t]) for t in chain],
                              "goal": pp.step_goal_text(chain, crng, split="eval", counted=0.85)})  # fmt: skip
        return specs
    for length in lengths:
        for _ in range(per_length):
            chain = [rng.choice(skills)]
            while len(chain) < length:
                chain.append(rng.choice([t for t in skills if t != chain[-1]]))
            specs.append({"chain": chain, "episodes": [rng.choice(pool[t]) for t in chain],
                          "goal": (pp.abstract_goal_text(chain, rng, split="eval", named_share=0.0)
                                   if goal_style == "abstract" else pp.any_goal_text(chain, rng))})  # fmt: skip
            if listing:  # G4: every skill may be in the list (held-out ones too), canonical descriptions; its own
                # random stream, so that the chains are the same as without lists
                lrng = random.Random(f"{seed}/listing/{len(specs)}")
                specs[-1]["listing"] = pp.Listing.build(chain, list(pp.SKILLS), lrng, vary=False, hints=hints).to_json()
    return specs


def run_chain(
    planner, data: Path, spec: dict, by_dir: dict, *, every: int, confirm: int, states=None, threshold=None
) -> dict:
    """One goal closed-loop, as long_horizon.py runs it: its held-out episodes one after the other, the planner asked
    every ``every`` seconds with the memory it built (MemoryTracker), a skill ends after ``confirm`` replies in a row
    naming another one (or HOLD_S s after its recording ends, as a robot standing still: missed). Each switch is scored
    like the sweep (caught / early / wrong next skill / missed); the goal goes on with the right skill after a mistake,
    so that every switch is scored, and counts as completed only without any."""
    from PIL import Image

    past = getattr(planner, "past", pp.PAST_S)

    def ask(episode, k, running):
        images = [Image.open(data / path).convert("RGB") for path in held_images(episode, k, past)]
        rows = states.get(episode["dir"]) if states else None
        state = pp.hands_text(rows + [rows[-1]] * (k + 1), k) if rows else None  # past the end: standing still
        return scored(planner, images, spec["goal"], running, memory.text(), state)

    listing = pp.Listing(spec["listing"]) if getattr(planner, "uses_list", False) else None
    planner.listing = listing  # G4: the goal's skill list (letters in its prompt, memory and replies)
    chain = spec["chain"]
    memory = pp.MemoryTracker(progress=getattr(planner, "uses_progress", False),
                              namer=listing.letter if listing else None)  # fmt: skip
    first = ask(by_dir[spec["episodes"][0]], 0, None)
    steps, asks = [], 1
    for i, (task, name) in enumerate(zip(chain, spec["episodes"], strict=True)):
        episode = by_dir[name]
        labels, run, switch, trace = episode["labels"], 0, None, []
        k, still, last = 0, 0, None
        while True:  # on with every image frozen at the end until the input stops changing (and no switch pending)
            frozen = k >= len(labels) + max(past)
            text = memory.text()
            if frozen and ((text == last and run == 0) or still >= MAX_STILL_ASKS):
                break
            skill, new_memory, p_run, other = ask(episode, k, task)
            skill = decide(skill, p_run, other, task, threshold)
            asks += 1
            memory.observe(new_memory)
            trace.append([k, skill, memory.progress, None if p_run is None else round(p_run, 3)])
            run = run + 1 if skill is not None and skill != task else 0
            if run >= confirm:
                switch = {"frame": k, "skill": skill}
                break
            if frozen:
                still, last = still + 1, text
            k += every
        if switch is None:
            outcome = "missed"
        elif labels[min(switch["frame"], len(labels) - 1)] == 0:
            outcome = "early"
        elif switch["skill"] != pp.next_skill(chain, i):
            outcome = "wrong"
        else:
            outcome = "caught"
        finish = labels.index(1)
        steps.append({"task": task, "episode": name, "outcome": outcome, "switch": switch, "finished_at": finish,
                      "delay_s": switch["frame"] - finish if switch else None, "trace": trace})  # fmt: skip
        memory.finish(task)
    start = decide(first[0], None, None, None, None)
    return {"chain": chain, "goal": spec["goal"], "start_ok": start == chain[0], "start": start,
            "completed": start == chain[0] and all(s["outcome"] == "caught" for s in steps), "asks": asks,
            "steps": steps, "threshold": threshold}  # fmt: skip


def run_chain_steps(planner, data: Path, spec: dict, by_dir: dict, *, every: int, confirm: int, states=None) -> dict:
    """run_chain for G5: the goal split once (decompose, scored against the chain), then each step replayed as in
    run_chain with the right plan (so that the steps are scored apart from the split); a step ends after ``confirm``
    replies in a row naming another step or done. Completed: the split right and every switch caught."""
    from PIL import Image

    past = getattr(planner, "past", pp.PAST_S)
    chain = spec["chain"]
    split = planner.decompose(spec["goal"])
    memory = pp.StepTracker(progress=True)
    steps, asks = [], 1
    for i, (task, name) in enumerate(zip(chain, spec["episodes"], strict=True)):
        episode = by_dir[name]
        labels, run, switch, trace = episode["labels"], 0, None, []
        k, still, last = 0, 0, None
        while True:
            frozen = k >= len(labels) + max(past)
            text = memory.text()
            if frozen and ((text == last and run == 0) or still >= MAX_STILL_ASKS):
                break
            images = [Image.open(data / path).convert("RGB") for path in held_images(episode, k, past)]
            rows = states.get(episode["dir"]) if states else None
            state = pp.hands_text(rows + [rows[-1]] * (k + 1), k) if rows else None
            result = planner.step(images, chain, i + 1, text, state)
            asks += 1
            memory.observe(result["memory"])
            step = result["step"]
            trace.append([k, step, memory.progress])
            run = run + 1 if step is not None and step != i + 1 else 0
            if run >= confirm:
                switch = {"frame": k, "skill": step}
                break
            if frozen:
                still, last = still + 1, text
            k += every
        right = i + 2 if i + 1 < len(chain) else pp.DONE
        if switch is None:
            outcome = "missed"
        elif labels[min(switch["frame"], len(labels) - 1)] == 0:
            outcome = "early"
        elif switch["skill"] != right:
            outcome = "wrong"
        else:
            outcome = "caught"
        finish = labels.index(1)
        steps.append({"task": task, "episode": name, "outcome": outcome, "switch": switch, "finished_at": finish,
                      "delay_s": switch["frame"] - finish if switch else None, "trace": trace})  # fmt: skip
        memory.finish(task)
    plan_ok = split["plan"] == chain
    monitor = all(s["outcome"] == "caught" for s in steps)
    return {"chain": chain, "goal": spec["goal"], "plan": split["plan"], "plan_reply": split["reply"],
            "plan_ok": plan_ok, "start_ok": bool(split["plan"]) and split["plan"][0] == chain[0],
            "start": split["plan"][0] if split["plan"] else None, "completed": plan_ok and monitor,
            "monitor_completed": monitor, "asks": asks, "steps": steps, "threshold": None}  # fmt: skip


def decompose_report(planner, n: int, seed: int) -> str:
    """G5's split alone, on goals never seen: random plans (repeats in a row too) in held-out wordings ("eval" purpose
    words and count templates), and in G2's named wordings; exact plans per kind of goal."""
    rng = random.Random(f"{seed}/decompose")
    rows = collections.defaultdict(lambda: [0, 0])
    misses = []
    for i in range(n):
        chain = pp.repeat_chain(rng, repeat=0.35)
        kind = "eval" if i % 2 else "named"
        goal = pp.step_goal_text(chain, rng, split="eval") if kind == "eval" else pp.any_goal_text(chain, rng)
        if kind == "named" and len(pp.runs_of(chain)) != len(chain):
            goal = pp.step_goal_text(chain, rng, split="train", purpose=0.0)  # named skills with counts
            kind = "named+count"
        got = planner.decompose(goal)["plan"]
        key = (kind, "有连续重复" if len(pp.runs_of(chain)) != len(chain) else "无连续重复")
        rows[key][0] += got == chain
        rows[key][1] += 1
        if got != chain and len(misses) < 12:
            misses.append(f"- {goal} → {got}（应为 {chain}）")  # noqa: RUF001
    lines = ["# G5 拆任务（只看文字，没见过的说法）", "", "| 说法 | 连续重复 | 拆对 |", "|---|---|---|"]
    for (kind, rep), (ok, total) in sorted(rows.items()):
        lines.append(f"| {kind} | {rep} | {ok}/{total} |")
    return "\n".join(lines + ["", "拆错的例子：", *misses]) + "\n"


def chain_report(records: list[dict], every: int, confirm: int, name: str) -> str:
    """Goals completed per length, then every switch by outcome, overall and per running skill. G5 (plan_ok in the
    records): also the plans split right and the goals the steps alone completed (with the right plan given)."""
    g5 = bool(records) and "plan_ok" in records[0]
    lines = [f"# 整链闭环重放（{name}，每 {every} 秒问一次，连续 {confirm} 次才切）", "",  # noqa: RUF001
             "| 链长 | 整链做完 | 开头选对 |" + (" 拆对 | 只看执行（计划给对）|" if g5 else ""),
             "|---|---|---|" + ("---|---|" if g5 else "")]  # fmt: skip
    for length in sorted({len(r["chain"]) for r in records}):
        group = [r for r in records if len(r["chain"]) == length]
        done, start = sum(r["completed"] for r in group), sum(r["start_ok"] for r in group)
        extra = ""
        if g5:
            extra = (f" {sum(r['plan_ok'] for r in group)}/{len(group)} |"
                     f" {sum(r['monitor_completed'] for r in group)}/{len(group)} |")  # fmt: skip
        lines.append(f"| {length} | {done}/{len(group)} ({done / len(group):.0%}) | {start}/{len(group)} |{extra}")
    steps = [s for r in records for s in r["steps"]]
    lines += [
        "",
        "| 正在做的技能 | 切换次数 | 抓到 | 提前切 | 切错下一个 | 漏掉 | 延迟中位数(秒) |",
        "|---|---|---|---|---|---|---|",
    ]
    for task in [*pp.SKILLS, "合计"]:
        group = steps if task == "合计" else [s for s in steps if s["task"] == task]
        if not group:
            continue
        c = collections.Counter(s["outcome"] for s in group)
        delays = [s["delay_s"] for s in group if s["outcome"] == "caught"]
        n = len(group)
        lines.append(f"| {task} | {n} | {c['caught']} ({c['caught'] / n:.0%}) | {c['early']} ({c['early'] / n:.0%}) | "
                     f"{c['wrong']} | {c['missed']} ({c['missed'] / n:.0%}) | "
                     f"{np.median(delays) if delays else '—'} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def fire_frame(skills: list, running: str, every: int, confirm: int) -> int | None:
    """The frame a switch happens when asked every ``every`` frames, ``confirm`` replies in a row naming another skill. The
    last answer holds forever: the robot stands still from there. None: never."""
    run, k = 0, 0
    for k in range(0, len(skills), every):
        run = run + 1 if skills[k] is not None and skills[k] != running else 0
        if run >= confirm:
            return k
    return k + every * (confirm - run) if run else None


def sweep_outcome(
    record: dict, episode: dict, takeover: int | None, every: int, confirm: int
) -> tuple[str, int | None]:
    """(outcome, frame of the switch). pretakeover: a switch while the model was failing, before the operator took over."""
    ask_every = record.get("ask_every", 1)
    frame = fire_frame(record["skills"], record["task"], every // ask_every, confirm)
    frame = None if frame is None else frame * ask_every
    labels = episode["labels"]
    if not episode["success"]:
        return ("false" if frame is not None else "quiet"), frame
    if frame is None:
        return "missed", None
    if takeover is not None and frame < takeover:
        return "pretakeover", frame
    return ("early" if labels[min(frame, len(labels) - 1)] == 0 else "caught"), frame


def sweep_report(
    records: list[dict], episodes: dict, takeovers: dict, rates=(2, 4, 6, 8, 10)
) -> tuple[list[dict], str]:
    """Every rate and number of replies in a row, overall; per task at each of them in the json."""
    table = []
    for every, confirm in ((e, c) for e in rates for c in (1, 2, 3, 4, 5, 6)):
        counts, delays = collections.Counter(), []
        tasks = collections.defaultdict(collections.Counter)
        for record in records:
            episode = episodes[record["episode"]]
            outcome, frame = sweep_outcome(record, episode, takeovers.get(record["episode"]), every, confirm)
            counts[outcome] += 1
            tasks[record["task"]][outcome] += 1
            if outcome == "caught":
                first = next((j for j, y in enumerate(episode["labels"]) if y == 1), None)
                if first is not None:
                    delays.append(frame - first)
        table.append({"every": every, "confirm": confirm, "counts": dict(counts), "delay_s": float(np.median(delays)) if delays
                      else None, "tasks": {t: dict(c) for t, c in tasks.items()}})  # fmt: skip
    lines = [
        "| 每几秒问 | 连续几次 | 成功回合抓到 | 提前切 | 接管前误切 | 失败回合误切 | 模型没做成时误切（合计） | 延迟中位数(秒) |",  # noqa: RUF001
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in table:
        c = collections.Counter(row["counts"])
        ok = c["caught"] + c["early"] + c["missed"] + c["pretakeover"]
        bad, n_bad = (
            c["false"] + c["pretakeover"],
            c["false"] + c["quiet"] + sum(e in takeovers for e in (r["episode"] for r in records)),
        )
        lines.append(
            f"| {row['every']} | {row['confirm']} | {c['caught']}/{ok} ({c['caught'] / max(1, ok):.0%}) | {c['early']} ({c['early'] / max(1, ok):.0%}) | "
            f"{c['pretakeover']} | {c['false']} | {bad}/{n_bad} ({bad / max(1, n_bad):.0%}) | {row['delay_s']} |"
        )
    return table, "\n".join(lines) + "\n"


def summarize(results: list[dict]) -> dict:
    counts = collections.Counter(r["outcome"] for r in results)
    caught = [r for r in results if r["outcome"] == "caught"]
    n_success = counts["caught"] + counts["early"] + counts["missed"]
    n_fail = counts["false"] + counts["quiet"]
    delays = [r["delay_s"] for r in caught if "delay_s" in r]
    return {
        **{k: counts[k] for k in ("caught", "early", "missed", "false", "quiet")},
        "recall": counts["caught"] / n_success if n_success else None,
        "false_rate": (counts["early"] + counts["false"]) / max(1, n_success + n_fail),
        "next_ok": sum(r["next_ok"] for r in caught) / len(caught) if caught else None,
        "memory_ok": sum(r["memory_ok"] for r in caught) / len(caught) if caught else None,
        "median_delay_s": float(np.median(delays)) if delays else None,
        "invalid_replies": sum(r["invalid"] for r in results),
        "mean_s": float(np.mean([r["mean_s"] for r in results if r["mean_s"] is not None])) if results else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter", default="base", help="a train_planner.py adapter dir, or base (zero-shot)")
    parser.add_argument("--out", default=None, help="report dir (default: <adapter>/eval, or <data>/eval_base)")
    parser.add_argument("--every", type=int, default=1, help="seconds between planner calls (a chunk: 1 or 2)")
    parser.add_argument("--confirm", type=int, default=2)
    parser.add_argument("--per-task", type=int, default=8, help="held-out episodes per task (half failures if any)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", default=None, help="k/n: replay only every n-th chosen episode, from the k-th")
    parser.add_argument("--merge", type=int, default=0, help="report from the replays of this many finished shards")
    parser.add_argument("--sweep", action="store_true", help="store the answers, score any rate from them; see above")
    parser.add_argument("--ask-every", type=int, default=2, help="--sweep: seconds between planner calls")
    parser.add_argument("--base", default=None, help="the model the adapter was trained on (default: RynnBrain1.1-2B)")
    parser.add_argument("--hands", choices=("on", "unknown"), default="on", help="a planner trained with the hands: "
                        "show them, or always say unknown")  # fmt: skip
    parser.add_argument("--threshold", type=float, default=None, help="--chains: switch when the running skill's p is "
                        "below this (planner_runtime.plan_scored), not on the greedy reply")  # fmt: skip
    parser.add_argument("--goal-style", choices=("named", "abstract", "count"), default="named",
                        help="--chains: goals naming the skills, (G3) purpose words held out of training, or (G5) "
                        "repeats in a row with counts in held-out wordings")  # fmt: skip
    parser.add_argument("--decompose", type=int, default=0, help="G5: only the split, on this many unseen goals")
    parser.add_argument("--chains", type=int, default=0, help="closed-loop goals of each length in CHAIN_LENGTHS "
                        "(run_chain), asked every --every seconds with --confirm; writes <out>/chains*.md")  # fmt: skip
    args = parser.parse_args()

    from policy.longhorizon.planner_runtime import Planner

    data = Path(args.data).expanduser()
    episodes = [json.loads(line) for line in (data / "episodes.jsonl").read_text().splitlines() if line.strip()]
    held = [e for e in episodes if e["split"] == "val"]
    chosen = []
    rng = random.Random(args.seed)
    for task in pp.SKILLS:
        failures = [e for e in held if e["task"] == task and not e["success"]]
        successes = [e for e in held if e["task"] == task and e["success"]]
        n_fail = min(len(failures), args.per_task // 2)
        chosen += rng.sample(failures, n_fail) + rng.sample(successes, min(len(successes), args.per_task - n_fail))
    out = Path(args.out).expanduser() if args.out else (
        Path(args.adapter).expanduser() / "eval" if args.adapter not in ("base", "") else data / "eval_base"
    )  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    if args.decompose:
        planner = Planner(args.adapter, **({"base": args.base} if args.base else {}), device=args.device)
        text = decompose_report(planner, args.decompose, args.seed)
        (out / "decompose.md").write_text(text)
        print(text, flush=True)
        return
    if args.chains:
        tag = f"chains_every{args.every}_confirm{args.confirm}" + (f"_t{args.threshold}" if args.threshold else "")
        path = out / f"{tag}.jsonl"
        by_dir = {e["dir"]: e for e in episodes}
        if not args.merge:
            meta = Path(args.adapter).expanduser() / "planner.json" if args.adapter not in ("base", "") else None
            info = json.loads(meta.read_text()) if meta and meta.is_file() else {}
            specs = chain_specs(held, args.seed, args.chains, goal_style=args.goal_style, listing=bool(info.get("list")),
                                hints=bool(info.get("hints")))
            if args.shard:
                k, n = (int(v) for v in args.shard.split("/"))
                specs, path = specs[k::n], path.with_suffix(f".shard{k}.jsonl")
            planner = Planner(args.adapter, **({"base": args.base} if args.base else {}), device=args.device)
            states = None
            if planner.uses_state and args.hands == "on":
                states = json.loads((data / "states.json").read_text())
            with path.open("w") as f:
                for i, spec in enumerate(specs, 1):
                    if getattr(planner, "uses_steps", False):
                        record = run_chain_steps(planner, data, spec, by_dir, every=args.every, confirm=args.confirm,
                                                 states=states)  # fmt: skip
                    else:
                        record = run_chain(planner, data, spec, by_dir, every=args.every, confirm=args.confirm,
                                           states=states, threshold=args.threshold)  # fmt: skip
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"{i}/{len(specs)} {len(spec['chain'])} skills: "
                          + ("completed" if record["completed"] else "not completed"), flush=True)  # fmt: skip
            if args.shard:
                return
        else:
            path.write_text("".join(path.with_suffix(f".shard{k}.jsonl").read_text() for k in range(args.merge)))
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        text = chain_report(records, args.every, args.confirm, args.adapter)
        (out / f"{tag}.md").write_text(text)
        print(text, flush=True)
        return
    replays = out / f"replays_every{args.every}.jsonl"
    if args.sweep:
        answers = out / "answers.jsonl"
        if not args.merge:
            if args.shard:
                k, n = (int(v) for v in args.shard.split("/"))
                chosen, answers = chosen[k::n], answers.with_suffix(f".shard{k}.jsonl")
            planner = Planner(args.adapter, **({"base": args.base} if args.base else {}), device=args.device)
            states = None
            if planner.uses_state and args.hands == "on":
                states = json.loads((data / "states.json").read_text())
            with answers.open("w") as f:
                for i, episode in enumerate(chosen, 1):
                    f.write(
                        json.dumps(
                            answers_of(planner, data, episode, seed=args.seed, ask_every=args.ask_every, states=states)
                        )
                        + "\n"
                    )
                    f.flush()
                    if i % 10 == 0:
                        print(f"{i}/{len(chosen)}", flush=True)
            if args.shard:
                return
        else:
            shards = [answers.with_suffix(f".shard{k}.jsonl").read_text() for k in range(args.merge)]
            answers.write_text("".join(shards))
        records = [json.loads(line) for line in answers.read_text().splitlines()]
        path = data / "takeovers.json"
        takeovers = json.loads(path.read_text()) if path.is_file() else {}
        by_dir = {e["dir"]: e for e in episodes}
        rates = tuple(args.ask_every * m for m in (1, 2, 3, 4, 5))  # multiples of the rate it was asked at
        table, text = sweep_report(records, by_dir, takeovers, rates)
        (out / "sweep.json").write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n")
        (out / "sweep.md").write_text(text)
        if any(r.get("p_run") and any(p is not None for p in r["p_run"]) for r in records):
            (out / "sweep_thresholds.md").write_text(threshold_report(records, by_dir, takeovers, args.ask_every))
        print(text, flush=True)
        return
    if args.merge:
        results = []
        for k in range(args.merge):
            shard = replays.with_suffix(f".shard{k}.jsonl").read_text()
            results += [json.loads(line) for line in shard.splitlines()]
        info = {"adapter": args.adapter, "prompt_version": pp.version()}
    else:
        if args.shard:
            k, n = (int(v) for v in args.shard.split("/"))
            chosen, replays = chosen[k::n], replays.with_suffix(f".shard{k}.jsonl")
        planner = Planner(args.adapter, **({"base": args.base} if args.base else {}), device=args.device)
        states = None
        if planner.uses_state and args.hands == "on":
            states = json.loads((data / "states.json").read_text())
        info = planner.info()
        results = []
        for i, episode in enumerate(chosen, 1):
            results.append(
                replay(planner, data, episode, every=args.every, confirm=args.confirm, seed=args.seed, states=states)
            )
            r = results[-1]
            print(f"{i}/{len(chosen)} {r['task']} ({'success' if episode['success'] else 'failure'}): {r['outcome']}"
                  + (f" -> {r['switch']['skill']}" if r["switch"] else ""), flush=True)  # fmt: skip
        replays.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
        if args.shard:
            return
    report = {
        "planner": info,
        "every_s": args.every,
        "hands": args.hands,
        "confirm": args.confirm,
        "overall": summarize(results),
        "tasks": {
            t: summarize([r for r in results if r["task"] == t])
            for t in pp.SKILLS
            if any(r["task"] == t for r in results)
        },
    }
    (out / f"report_every{args.every}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.merge:
        replays.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    lines = [
        f"# 高层规划器留出重放（{info['adapter']}，每 {args.every} 秒问一次，连续 {args.confirm} 次才切）",  # noqa: RUF001
        "",
        "| 任务 | 抓到 | 提前 | 漏掉 | 失败误切 | 安静 | 召回 | 误判率 | 下一个对 | 记忆对 | 延迟中位数(秒) | 无效回复 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    fmt = lambda v, f: "—" if v is None else format(v, f)  # noqa: E731
    for name, r in [*report["tasks"].items(), ("合计", report["overall"])]:
        lines.append(
            f"| {name} | {r['caught']} | {r['early']} | {r['missed']} | {r['false']} | {r['quiet']} | "
            f"{fmt(r['recall'], '.0%')} | {fmt(r['false_rate'], '.0%')} | {fmt(r['next_ok'], '.0%')} | "
            f"{fmt(r['memory_ok'], '.0%')} | {fmt(r['median_delay_s'], '+.1f')} | {r['invalid_replies']} |"
        )
    lines += ["", f"每次规划平均 {fmt(report['overall']['mean_s'], '.2f')} 秒。"]
    (out / f"report_every{args.every}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
