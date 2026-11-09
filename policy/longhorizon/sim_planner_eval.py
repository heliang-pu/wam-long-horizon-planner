"""The planner in the simulator, closed loop: scripts/eval_policy.py (through sim_twin/eval_t70kit.py) with the
long-horizon adapter around a world model, where an episode does not end at the task's success. The simulator's own
success is recorded after every chunk instead, and the episode ends once the planner says the goal is done
(long_horizon_stop_when_finished) or at the step limit. So the planner's switch can be compared with the truth: before
any success (early), after it (caught, with the delay), never (missed), or with no success at all (false / quiet).

A simulator scene holds one task, so the goal is that one task: the planner has to say done when it is.

    LH_SIM_OUT=<dir> <simulator python> policy/longhorizon/sim_planner_eval.py --config policy/longhorizon/deploy_policy.yml \
        --overrides --task_name click_bell --setting t70kit10lh_random --inner_policy phoenix --checkpoint_dir <run dir> \
        --long_horizon_goal 按铃。 --long_horizon_planner <adapter> --long_horizon_stop_when_finished True ...

Writes $LH_SIM_OUT/episodes.jsonl (one line per episode) and prints a summary at the end.
"""

from __future__ import annotations

import atexit
import importlib
import json
import os
from pathlib import Path
import pkgutil
import runpy
import sys

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("LH_SIM_OUT", "lh_sim_out"))
ORIGINAL = {}  # task class -> its own is_task_success


def patch_success() -> None:
    """Every task's is_task_success answers False to the evaluation loop (so success does not end an episode); the
    original is kept for the recorder."""
    import robosynchallenge.tasks as tasks

    for info in pkgutil.walk_packages(tasks.__path__, tasks.__name__ + "."):
        try:
            module = importlib.import_module(info.name)
        except Exception:  # a module that does not import is not a task this run uses
            continue
        for value in vars(module).values():
            if isinstance(value, type) and "is_task_success" in vars(value) and value not in ORIGINAL:
                ORIGINAL[value] = value.is_task_success
                value.is_task_success = lambda self, **kwargs: False


def true_success(env) -> bool:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    for cls in type(base).__mro__:
        if cls in ORIGINAL:
            value = ORIGINAL[cls](base)
            return bool(value.any()) if hasattr(value, "any") else bool(value)
    return False


class Recorder:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / "episodes.jsonl"
        self.episode, self.last_step, self.model, self.successes, self.noted = 0, -1, None, [], False

    def note(self, env, model, step: int) -> None:
        if step < self.last_step:
            self.flush()
        self.model, self.last_step, self.noted = model, step, True
        if true_success(env):
            self.successes.append(step)

    def flush(self) -> None:
        if self.model is None or self.last_step < 0 or not self.noted:  # nothing new since the last flush
            return
        summary = self.model.episode_summary()
        first = (summary.get("subtasks") or [{}])[0]
        record = {
            "episode": self.episode,
            "task": first.get("task"),
            "steps": self.last_step,
            "first_success": self.successes[0] if self.successes else None,
            "last_success": self.successes[-1] if self.successes else None,
            "success_checks": len(self.successes),
            "planner_end": first.get("end_step"),
            "reason": first.get("reason"),
            "next": first.get("next"),
            "finished": summary.get("finished"),
            "plans": first.get("plans") or summary.get("pending_plans"),
            "subtasks": [{k: v for k, v in s.items() if k != "plans"} for s in summary.get("subtasks", [])],
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        print(f"[lh_sim] episode {self.episode}: success at {record['first_success']}, planner ended at "
              f"{record['planner_end']} ({record['reason']}, next {record['next']})", flush=True)  # fmt: skip
        self.episode, self.last_step, self.successes, self.noted = self.episode + 1, -1, [], False


def outcome(r: dict, hz: float = 10.0) -> str:
    switched = r["reason"] == "planner"
    if r["first_success"] is None:
        return "false" if switched else "quiet"
    if not switched:
        return "missed"
    return "early" if r["planner_end"] < r["first_success"] else "caught"


def summarize() -> None:
    path = OUT / "episodes.jsonl"
    if not path.exists():
        return
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    counts = {}
    for r in rows:
        counts[outcome(r)] = counts.get(outcome(r), 0) + 1
    delays = [(r["planner_end"] - r["first_success"]) / 10.0 for r in rows if outcome(r) == "caught"]
    print(f"[lh_sim] {len(rows)} episodes: {counts}; delay after success (s): {delays}", flush=True)


def main() -> None:
    sys.path[:0] = [str(REPO), str(REPO / "policy")]
    patch_success()
    import policy.longhorizon as adapter

    recorder = Recorder()
    inner_eval = adapter.eval

    def eval_and_record(env, model, obs):
        out = inner_eval(env, model, obs)
        recorder.note(env, model, int(out[1]["elapsed_steps"].item()))
        return out

    adapter.eval = eval_and_record
    atexit.register(summarize)
    atexit.register(recorder.flush)
    sys.argv = [str(REPO / "sim_twin" / "eval_t70kit.py"), *sys.argv[1:]]
    runpy.run_path(str(REPO / "sim_twin" / "eval_t70kit.py"), run_name="__main__")
    recorder.flush()


if __name__ == "__main__":
    main()
