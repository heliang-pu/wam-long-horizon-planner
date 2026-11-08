"""A large model as the planner, zero-shot: Claude asked exactly what the fine-tuned planners are asked (the six 320x240
images, planner_prompt.prompt_text, the same two-line reply), plus each skill's success condition in words, since it
never saw our labels. Scored like eval_planner.py --sweep on a subset of the held-out episodes, beside the fine-tuned
planners' answers on the same episodes.

Runs on the CPU (API calls; its own venv venvs/claude_judge with the anthropic SDK). The key is read from
~/.config/claude_judge/key (never from the repository).

    venvs/claude_judge/bin/python policy/longhorizon/claude_judge.py run --data <planner data> --out <dir> \
        [--model claude-opus-5-5 --effort low --per-task 12 --ask-every 3 --workers 8]
    ... report --data <planner data> --out <dir> --compare F=<adapter>/sweep_every3_p/answers.jsonl ...
"""

from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures as cf
import json
import os
from pathlib import Path
import random
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.longhorizon import eval_planner as ep  # noqa: E402
from policy.longhorizon import planner_prompt as pp  # noqa: E402

KEY_FILE = Path.home() / ".config" / "claude_judge" / "key"
PRICES = {  # $ per million input / output tokens (Claude API, cached 2026-10-06)
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
FINISHED_WHEN = {  # the official success conditions (finals FAQ 6), in words
    "click_bell": "the bell's button has been pressed",
    "drawer_open_place": "the drawer is open and the tomato is inside it (closing the drawer is not needed)",
    "handle_basket": "the bottle is in the basket and the basket has been lifted",
    "item_assembly": "the two centrifuge tubes are docked together",
    "items_handover": "the pen was passed to the other arm and is standing in the pen holder",
    "manipulate_pipette": "the pipette's button has been pressed and released once over the beaker",
    "mixer_operating": "the beaker stands on the mixer and the mixer's switch has been pressed",
    "sample_loading": "the test tube was passed to the other arm and is inserted in the rack",
    "table_rearrangement": "the fork and the spoon lie next to the plate",
    "water_pouring": "water has been poured into the cup and both the bottle and the cup are put back on the table",
}
SYSTEM = (
    "You watch a dual-arm robot through its cameras and decide when the skill it is running is finished. "
    "A skill is finished when:\n" + "\n".join(f"- {name}: {text}" for name, text in FINISHED_WHEN.items()) + "\n"
    "Follow the answer format of the request exactly."
)


class ClaudePlanner:
    """planner_runtime.Planner's plan() through the Claude API; usage and latency kept for the report."""

    uses_state, uses_progress, past = False, False, pp.PAST_S

    def __init__(self, data: Path, model: str, effort: str):
        import anthropic

        self.data, self.model, self.effort = data, model, effort
        self.client = anthropic.Anthropic(api_key=KEY_FILE.read_text().strip(), max_retries=8)
        self.lock = threading.Lock()
        self.calls, self.input_tokens, self.output_tokens, self.seconds, self.failed = 0, 0, 0, [], 0

    def plan(self, images, goal, running, memory, state=None):
        content = []
        for image in images:  # PIL images of the planner's files: sent as the JPEGs they are
            import io

            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=90)
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": base64.standard_b64encode(buffer.getvalue()).decode()}})  # fmt: skip
        content.append({"type": "text", "text": pp.prompt_text(goal, running, memory)})
        started = time.monotonic()
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4000,
                system=SYSTEM,
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:  # an API error is an unusable reply, as an unparsable one
            with self.lock:
                self.failed += 1
            return None, None, f"error: {exc!r}"[:300], time.monotonic() - started
        took = time.monotonic() - started
        text = "" if response.stop_reason == "refusal" else "".join(
            b.text for b in response.content if b.type == "text")  # fmt: skip
        with self.lock:
            self.calls += 1
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
            self.seconds.append(took)
        skill, new_memory = pp.parse_reply(text)
        return skill, new_memory, text, took

    def usage(self) -> dict:
        price_in, price_out = PRICES.get(self.model, (0.0, 0.0))
        seconds = sorted(self.seconds)
        return {"model": self.model, "effort": self.effort, "calls": self.calls, "failed": self.failed,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "dollars": round(self.input_tokens / 1e6 * price_in + self.output_tokens / 1e6 * price_out, 2),
                "seconds_median": seconds[len(seconds) // 2] if seconds else None,
                "seconds_p90": seconds[int(0.9 * len(seconds))] if seconds else None}  # fmt: skip


def subset(data: Path, per_task: int, seed: int) -> list[dict]:
    """Per task, half failures (or all there are) and the rest successes among the held-out episodes, seeded."""
    episodes = [json.loads(line) for line in (data / "episodes.jsonl").read_text().splitlines() if line.strip()]
    held, rng, chosen = [e for e in episodes if e["split"] == "val"], random.Random(f"{seed}/claude"), []
    for task in pp.SKILLS:
        failures = [e for e in held if e["task"] == task and not e["success"]]
        successes = [e for e in held if e["task"] == task and e["success"]]
        n_fail = min(len(failures), per_task // 2)
        chosen += rng.sample(failures, n_fail) + rng.sample(successes, min(len(successes), per_task - n_fail))
    return chosen


def run(args: argparse.Namespace) -> None:
    data, out = Path(args.data).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    planner = ClaudePlanner(data, args.model, args.effort)
    chosen = subset(data, args.per_task, args.seed)
    path = out / "answers.jsonl"
    done = {json.loads(line)["episode"] for line in path.read_text().splitlines()} if path.exists() else set()
    todo = [e for e in chosen if e["dir"] not in done]
    print(f"{len(chosen)} episodes, {len(todo)} to ask, {args.model} effort {args.effort}", flush=True)
    write = threading.Lock()

    def one(episode):
        record = ep.answers_of(planner, data, episode, seed=args.seed, ask_every=args.ask_every)
        with write, path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    with cf.ThreadPoolExecutor(args.workers) as pool:
        for n, record in enumerate(pool.map(one, todo), 1):
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {record['task']}: {planner.usage()}", flush=True)
    usage_path = out / "usage.json"
    previous = json.loads(usage_path.read_text()) if usage_path.exists() else []
    usage_path.write_text(json.dumps([*previous, planner.usage()], indent=1) + "\n")


def report(args: argparse.Namespace) -> None:
    """The sweep table at the asked rate for Claude and, on the same episodes, each --compare answers file."""
    data, out = Path(args.data).expanduser(), Path(args.out).expanduser()
    episodes = {json.loads(x)["dir"]: json.loads(x) for x in (data / "episodes.jsonl").read_text().splitlines()}
    takeovers = json.loads((data / "takeovers.json").read_text())
    mine = [json.loads(x) for x in (out / "answers.jsonl").read_text().splitlines() if x.strip()]
    names = {r["episode"] for r in mine}
    runs = {"Claude": mine}
    for item in args.compare:  # NAME=answers.jsonl, or NAME@threshold=answers.jsonl for a run that kept p_run
        name, path = item.split("=", 1)
        records = [r for r in (json.loads(x) for x in Path(path).read_text().splitlines() if x.strip())
                   if r["episode"] in names]  # fmt: skip
        if "@" in name:
            threshold = float(name.split("@", 1)[1])
            records = [{**r, "skills": [ep.decide(sk, p, o, r["task"], threshold)
                                        for sk, p, o in zip(r["skills"], r["p_run"], r["other"], strict=True)]}
                       for r in records]  # fmt: skip
        runs[name] = records
    rate = mine[0]["ask_every"]
    n_success = sum(episodes[n]["success"] for n in names)
    head = f"同一批 {len(names)} 个留出回合（{n_success} 个成功），每 {rate} 秒问一次"  # noqa: RUF001
    lines = [head, "",
             "| 规划器 | 回合 | 连续几次 | 成功回合抓到 | 提前切 | 模型没做成时误切 | 延迟中位数(秒) |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, records in runs.items():
        table, _text = ep.sweep_report(records, episodes, takeovers, rates=(records[0]["ask_every"],))
        n_bad = sum(not episodes[r["episode"]]["success"] for r in records) + sum(
            r["episode"] in takeovers for r in records
        )
        for row in table[:3]:
            c = collections.Counter(row["counts"])
            ok = c["caught"] + c["early"] + c["missed"] + c["pretakeover"]
            bad = c["false"] + c["pretakeover"]
            lines.append(f"| {name} | {len(records)} | {row['confirm']} | {c['caught']}/{ok} ({c['caught'] / max(1, ok):.0%}) "
                         f"| {c['early']} ({c['early'] / max(1, ok):.0%}) | {bad}/{n_bad} ({bad / max(1, n_bad):.0%}) "
                         f"| {row['delay_s']} |")  # fmt: skip
    invalid = sum(s is None for r in mine for s in r["skills"])
    lines += ["", f"Claude: {sum(len(r['skills']) for r in mine)} asks, {invalid} unusable replies."]
    usage_path = out / "usage.json"
    if usage_path.exists():
        for u in json.loads(usage_path.read_text()):
            lines.append(f"{u['model']} effort {u['effort']}: {u['calls']} calls, {u['failed']} failed, ${u['dollars']}, "
                         f"median {u['seconds_median']:.1f} s, p90 {u['seconds_p90']:.1f} s per call")  # fmt: skip
    text = "\n".join(lines) + "\n"
    (out / "compare.md").write_text(text)
    print(text, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default="low", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--per-task", type=int, default=12)
    parser.add_argument("--ask-every", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", nargs="*", default=[], help="NAME=answers.jsonl of other planners")
    args = parser.parse_args()
    if args.command == "run" and not KEY_FILE.is_file():
        raise SystemExit(f"no API key at {KEY_FILE}")
    os.environ.pop("ALL_PROXY", None)  # httpx2 takes no socks:// proxy; the HTTP one is used
    os.environ.pop("all_proxy", None)
    {"run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
