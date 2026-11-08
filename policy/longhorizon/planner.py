"""The deploy side of the high-level planner (long_horizon.py's planner mode): the client of planner_worker.py, which
runs RynnBrain1.1-2B under the planner's own interpreter, and the short-term visual memory (the top camera once a
second) the planner is asked with.
"""

from __future__ import annotations

import atexit
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading

import numpy as np

from policy.longhorizon import planner_prompt as pp

POLICY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = POLICY_ROOT.parents[1]


def default_planner_python() -> str:
    """$LONGHORIZON_PLANNER_PYTHON, else WS_ROOT/venvs/planner (a worktree under .worktrees/: three levels up)."""
    if os.environ.get("LONGHORIZON_PLANNER_PYTHON"):
        return os.environ["LONGHORIZON_PLANNER_PYTHON"]
    for ws_root in (REPO_ROOT.parent, REPO_ROOT.parents[2]):
        python = ws_root / "venvs" / "planner" / "bin" / "python"
        if python.exists():
            return str(python)
    raise FileNotFoundError("no planner interpreter: run policy/longhorizon/setup_planner_env.sh")


class FrameMemory:
    """The top camera once a second (episode step, image): the planner's short-term visual memory."""

    def __init__(self, control_hz: float, past=pp.PAST_S):
        self.every = max(1, round(float(control_hz)))
        self.control_hz = float(control_hz)
        self.past_s = tuple(past)
        self.frames: list[tuple[int, np.ndarray]] = []

    def reset(self) -> None:
        self.frames = []

    def due(self, step: int) -> bool:
        return not self.frames or step - self.frames[-1][0] >= self.every

    def note(self, step: int, image: np.ndarray) -> None:
        if self.due(step):
            self.frames.append((step, image))
            del self.frames[: -(max(self.past_s) + 2)]

    def past(self, step: int, current: np.ndarray) -> list[np.ndarray]:
        """The top camera past_s seconds before ``step``: the latest kept frame at or before each, else the
        earliest one (or the current image), as the training data clamps to the episode's first frame."""
        out = []
        for seconds in self.past_s:
            target = step - seconds * self.control_hz
            before = [image for s, image in self.frames if s <= target]
            out.append(before[-1] if before else (self.frames[0][1] if self.frames else current))
        return out


class PlannerClient:
    """Drives planner_worker.py over a stdio JSON pipe; submit() runs a request in the background (one at a time)."""

    def __init__(self, adapter: str, *, python: str | None = None, cuda_visible_devices: str = "0", skills=()):
        self.skills = list(skills)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        env.pop("PYTHONPATH", None)  # the deploy venv's paths must not leak into the planner's interpreter
        cmd = [python or default_planner_python(), str(POLICY_ROOT / "planner_worker.py"), "--adapter", str(adapter)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True,
                                     bufsize=1, env=env)  # fmt: skip
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        atexit.register(self.close)
        self.info = self._rpc({"cmd": "info"})["info"]
        print(f"[planner] {self.info}", flush=True)

    def _rpc(self, payload: dict) -> dict:
        with self.lock:
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("planner worker exited; its traceback is on stderr above")
                try:
                    reply = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not reply.get("ok"):
                    raise RuntimeError(reply.get("error", "planner worker error"))
                return reply

    def plan(self, images: list[np.ndarray], goal: str, running: str | None, memory: str) -> dict:
        """images: cam_high, cam_left_wrist, cam_right_wrist now, then the top camera 3, 2, 1 s before."""
        fd, path = tempfile.mkstemp(prefix="planner_", suffix=".npz")
        os.close(fd)
        try:
            np.savez(path, **{f"image_{k}": np.ascontiguousarray(image) for k, image in enumerate(images)})
            return self._rpc({"cmd": "plan", "images_path": path, "goal": goal, "running": running,
                              "memory": memory, "skills": self.skills})  # fmt: skip
        finally:
            os.remove(path)

    def submit(self, images, goal, running, memory) -> Future:
        return self.executor.submit(self.plan, images, goal, running, memory)

    def close(self) -> None:
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait(timeout=10)
