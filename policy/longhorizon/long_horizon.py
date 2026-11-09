# ----------------------------------------------------------------------------
# Long-horizon module: one episode runs several of the ten tasks in a row, on any world action model adapter
# (policy/phoenix, policy/internw0, policy/openwam: get_model / eval / reset_model), the model unchanged.
#
# Per subtask the inner adapter runs with that task's prompt and per-task settings, and sees an env whose step()
# also feeds the done detector (done_detector.py, trained apart from the WAM: train_done_detector.py). Once the
# detector is sure (its per-task threshold, `confirm` frames in a row, not before `min_steps`), the arms go back to the
# pose the episode started from, the adapter is reset and the next subtask starts from there, as its training episodes
# did. A subtask that runs past `max_steps` is given up (on_timeout next) or ends the chain (stop); request_next() (an
# operator key) ends a subtask by hand. After the last subtask the arms hold the start pose: the episode ends as any
# other (the operator's s / q, or the step limit).
#
# Switching keeps the loaded weights whenever it can: a different prompt, and execution or denoising steps the model
# can change live (Phoenix's set_execution_steps / set_denoise_steps), only retarget the running model; any other
# per-task setting (another checkpoint_dir, ...) builds the model again (Phoenix on its running worker when the weights
# are the same).
#
# A detector on Phoenix's own features (source "phoenix", wam_features.py) needs no images here: the adapter's worker
# is started through phoenix_tap_worker (phoenix_python), which writes the features of every inference to a file, and
# after each chunk the features of that chunk's inference (the observation it started from) go to the detector. The
# Phoenix checkpoint and the prompts must be the ones the features were extracted with.
#
# Planner mode (long_horizon_goal + long_horizon_planner), as π0.5's high-level step with MEM's memories: the goal is one
# sentence and a vision-language model (planner_worker.py: RynnBrain1.1-2B, zero-shot or a train_planner.py adapter)
# picks the skills. It is asked in the background after every chunk (the cameras now, the top camera 3, 2, 1 s before,
# the running skill and its language memory); --confirm replies in a row naming another skill (or done) end the running
# one, the memory becomes what it wrote. After an operator's n or a timeout it is asked again, nothing running, once
# the arms are back at the start pose (a timed-out skill is not finished: it may well pick it again).
# ----------------------------------------------------------------------------

from __future__ import annotations

import contextlib
import dataclasses
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import numpy as np
import torch

from policy.longhorizon import planner_prompt as pp
from policy.longhorizon.done_detector import CAMERAS
from policy.longhorizon.done_detector import TASKS
from policy.longhorizon.done_detector import DoneDetector
from policy.longhorizon.done_detector import extract_image
from policy.longhorizon.done_detector import frame_of
from policy.longhorizon.done_detector import qpos_of
from policy.longhorizon.planner import FrameMemory
from policy.longhorizon.planner import PlannerClient
from policy.longhorizon.wam_features import FEATURES_OUT
from policy.longhorizon.wam_features import LAYERS_ENV
from policy.longhorizon.wam_features import read_features

POLICY_ROOT = Path(__file__).resolve().parent
WAM_PYTHON = "LONGHORIZON_WAM_PYTHON"  # the interpreter phoenix_tap_worker runs the worker with

GRIPPER_DIMS = [6, 13]
ON_TIMEOUT = ("next", "stop")
# the per-task settings a switch passes without building the model again (the rest of a subtask's args must match)
PROMPT_KEYS = ("task_name", "prompt")


HOME_PROMPT = "return both arms to the start pose"  # the recorded task of the frames between two skills


@dataclasses.dataclass
class Settings:
    """long_horizon_* keys of deploy_policy.yml / --overrides (see policy/longhorizon/deploy_policy.yml)."""

    tasks: list[str]
    detector: str = ""
    device: str = "auto"
    threshold: float | None = None  # None: each task's threshold from the detector's validation (config.json)
    confirm: int = 2
    min_steps: int = 30
    max_steps: int = 1000
    on_timeout: str = "next"
    return_home: bool = True
    home_rad_per_step: float = 0.05
    home_steps_per_call: int = 10
    settle_steps: int = 5
    control_hz: float = 10.0
    gripper_scale: float = 0.05
    prompts: dict = dataclasses.field(default_factory=dict)
    task_overrides: dict = dataclasses.field(default_factory=dict)
    goal: str = ""  # planner mode: one sentence; tasks are then the skills the planner may pick (default: all it knows)
    planner: str = ""  # a train_planner.py adapter dir, or base (RynnBrain1.1-2B zero-shot)
    planner_confirm: int = 2  # planner replies in a row naming another skill before the running one ends
    planner_every_s: float = 3.0  # at least this long between planner asks (0: after every chunk); evaluated at 3 s
    # > 0: a reply votes to switch when the running skill's probability (planner_runtime.plan_scored) is below this,
    # to the likeliest other skill, whatever the greedy reply says; 0: the greedy reply (eval_planner.py --threshold)
    planner_threshold: float = 0.0
    stop_when_finished: bool = False  # the simulator: end the episode (truncated) once the goal is finished
    planner_python: str = ""  # default $LONGHORIZON_PLANNER_PYTHON, else WS_ROOT/venvs/planner
    planner_cuda_visible_devices: str = ""  # default the adapter's gpu_id

    @classmethod
    def from_usr_args(cls, usr_args: dict, tasks: list[str] | None = None) -> Settings:
        values = {}
        for field in dataclasses.fields(cls):
            key = f"long_horizon_{field.name}"
            if usr_args.get(key) is not None:
                values[field.name] = usr_args[key]
        if tasks is not None:
            values["tasks"] = list(tasks)
        if isinstance(values.get("tasks"), str):
            values["tasks"] = [t.strip() for t in values["tasks"].split(",") if t.strip()]
        if values.get("goal") and not values.get("tasks"):
            values["tasks"] = list(pp.SKILLS)  # the planner may pick any skill the adapter has a prompt for
        values.setdefault("tasks", [usr_args["task_name"]] if usr_args.get("task_name") else [])
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.tasks:
            raise ValueError("long_horizon_tasks is empty: give the subtasks in order")
        if self.on_timeout not in ON_TIMEOUT:
            raise ValueError(f"long_horizon_on_timeout must be one of {ON_TIMEOUT}, got {self.on_timeout!r}")
        if int(self.confirm) < 1 or float(self.home_rad_per_step) <= 0 or int(self.home_steps_per_call) < 1:
            raise ValueError("long_horizon_confirm, _home_rad_per_step and _home_steps_per_call must be positive")
        unknown = set(self.task_overrides) - set(TASKS) - set(self.tasks)
        if unknown:
            raise ValueError(f"long_horizon_task_overrides names unknown tasks: {sorted(unknown)}")
        if bool(self.goal) != bool(self.planner):
            raise ValueError("long_horizon_goal and long_horizon_planner go together (planner mode)")
        if self.planner and self.detector:
            raise ValueError("planner mode decides the switches itself: no long_horizon_detector with it")
        if self.goal and set(self.tasks) - set(pp.SKILLS):
            raise ValueError(
                f"the planner knows the skills {list(pp.SKILLS)}, not {sorted(set(self.tasks) - set(pp.SKILLS))}"
            )


def env_step_index(env) -> int | None:
    """The env's step count this episode (RealCobotMagicEnv.step_count), or None where the env has none."""
    try:
        value = getattr(env, "step_count", None)
    except Exception:  # gymnasium wrappers raise on unknown attributes
        return None
    return int(value) if isinstance(value, int | np.integer) else None


def _any_true(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    return bool(np.asarray(value).any())


class StepTap:
    """The env as the inner adapter sees it: step() goes through and is counted (and fed to the done detector), the
    instruction is the subtask's prompt, everything else is the env's own."""

    def __init__(self, env, owner: LongHorizonModel, instruction: str):
        self.__dict__.update(_env=env, _owner=owner, _current_instruction=instruction)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        result = self._env.step(action)
        self._owner.on_policy_step(result[0])
        return result


class LongHorizonModel:
    """The chain of subtasks around one inner adapter model. Attributes it does not have are the inner model's (so
    run_real's console still reads the running adapter's steps, stall counts and background inference)."""

    def __init__(self, policy, usr_args: dict, settings: Settings, prepare=None):
        self.policy, self.settings, self.prepare = policy, settings, prepare
        self.base_args = dict(usr_args)
        module = getattr(policy, "__name__", "policy").split(".")  # policy.<name>.deploy_policy
        self.name = str(usr_args.get("policy_name") or module[-2 if len(module) > 1 else -1])
        self.detector = DoneDetector(settings.detector, settings.device) if settings.detector else None
        if self.detector is not None:
            missing = [task for task in settings.tasks if task not in self.detector.tasks]
            if missing:
                raise ValueError(f"the done detector {settings.detector} was not trained on {missing}")
            self.stride = max(1, round(float(settings.control_hz) / self.detector.fps))
        else:
            self.stride = 1
        self.features_path = None  # phoenix source: the file the worker writes every inference's features to
        if self.detector is not None and self.detector.source == "phoenix":
            self._tap_phoenix()
        self.next_requested = threading.Event()
        self.planner, self.top_frames = None, FrameMemory(settings.control_hz)
        if settings.goal:  # the skills the adapter has a prompt for
            settings.tasks = [task for task in settings.tasks if self.prompt_of(task, self.base_args)]
            if not settings.tasks:
                raise ValueError("planner mode: the adapter has a prompt for none of the planner's skills")
        self.index = 0
        self.inner, self.inner_args = None, None
        self.inner = self._build(self.task_args(settings.tasks[0]))
        self.pending_goal, self.pending_planner, self.planner_loading = None, None, None  # taken at an episode's start
        self.planner_error = None
        self.on_skill = None  # (task) -> None at each skill's start: run_real's console sets that task's gripper
        if settings.goal:
            self.planner = self._new_planner(settings.planner)
        # a memory planner (train_planner.py --progress): the memory also keeps the running skill's progress, built
        # from every reply (planner_prompt.MemoryTracker); otherwise it changes only when a skill ends
        self.uses_progress = bool(
            self.planner is not None and (getattr(self.planner, "info", None) or {}).get("progress")
        )
        if self.planner is not None:  # the top-camera history the adapter was trained with
            info = getattr(self.planner, "info", None) or {}
            self.top_frames = FrameMemory(settings.control_hz, past=info.get("past_s") or pp.PAST_S)
        self.started = False  # an episode is under way
        self.reset_pending = False  # reset_model ran since the last eval (episode start in the simulator, a takeover)
        self.last_step = -1
        self.records: list[dict] = []
        if settings.goal:
            print(f"[long_horizon] {self.name}: goal {settings.goal!r}; planner {settings.planner}, skills "
                  f"{settings.tasks}", flush=True)  # fmt: skip
        else:
            chain = " -> ".join(settings.tasks)
            detector = settings.detector or "none (subtasks end by request_next or max_steps only)"
            rate = "one per chunk (Phoenix's features)" if self.features_path else f"every {self.stride} steps"
            print(f"[long_horizon] {self.name}: {chain}; detector {detector}, {rate}", flush=True)

    def _new_planner(self, adapter: str) -> PlannerClient:
        gpu = self.settings.planner_cuda_visible_devices or str(
            self.base_args.get(f"{self.name}_cuda_visible_devices", self.base_args.get("gpu_id", 0))
        )
        return PlannerClient(adapter, python=self.settings.planner_python or None, cuda_visible_devices=gpu,
                             skills=self.settings.tasks)  # fmt: skip

    def frame_prompt(self) -> str:
        """The task string a recorder writes for the step now: the running skill's prompt, or HOME_PROMPT while the
        arms go back to (or hold) the start pose between skills."""
        if getattr(self, "phase", None) == "run" and self.inner_args:
            return str(self.inner_args.get("prompt") or self.settings.goal)
        return HOME_PROMPT

    def set_goal(self, goal: str) -> None:
        """Another goal sentence (the web console's 长任务 tab, no restart): taken at the next episode's start."""
        if not self.settings.goal:
            raise ValueError("not a planner chain (--goal)")
        self.pending_goal = str(goal)

    def set_planner(self, adapter: str) -> None:
        """Another planner adapter (no restart, the world model stays): its worker starts in the background and replaces
        the running one at the next episode's start; until then the running one goes on."""
        if not self.settings.goal:
            raise ValueError("not a planner chain (--goal)")
        if self.planner_loading:
            raise ValueError(f"a planner is still loading ({self.planner_loading})")
        self.planner_loading = str(adapter)

        def load() -> None:
            try:
                self.pending_planner = (str(adapter), self._new_planner(str(adapter)))
            except Exception as exc:  # noqa: BLE001 - shown on the page, the running planner stays
                print(f"[long_horizon] planner {adapter} did not start: {exc}", flush=True)
                self.planner_error = f"{adapter}: {exc}"
            finally:
                self.planner_loading = None

        self.planner_error = None
        threading.Thread(target=load, name="planner-load", daemon=True).start()

    def close_planners(self) -> None:
        """The planner workers (the running one and one loaded for the next episode) give their GPU memory back."""
        pending, self.pending_planner = self.pending_planner, None
        for planner in (self.planner, pending[1] if pending else None):
            if planner is not None:
                planner.close()
        self.planner = None

    def _take_pending(self) -> None:
        """An episode starts: the goal and planner chosen since the one before (set_goal, set_planner)."""
        if self.pending_goal is not None:
            self.settings.goal, self.pending_goal = self.pending_goal, None
            print(f"[long_horizon] goal now {self.settings.goal!r}", flush=True)
        if self.pending_planner is not None:
            (adapter, planner), self.pending_planner = self.pending_planner, None
            old, self.planner, self.settings.planner = self.planner, planner, adapter
            info = getattr(planner, "info", None) or {}
            self.uses_progress = bool(info.get("progress"))
            self.top_frames = FrameMemory(self.settings.control_hz, past=info.get("past_s") or pp.PAST_S)
            if old is not None:
                old.close()
            print(f"[long_horizon] planner now {adapter}", flush=True)

    def _tap_phoenix(self) -> None:
        """Phoenix's own features: its adapter starts the worker through phoenix_tap_worker (phoenix_python, an adapter
        setting), told by environment variables where to write and which layers to take."""
        if self.name != "phoenix":
            raise ValueError(f"the done detector {self.settings.detector} reads Phoenix's features, not {self.name}'s")
        launcher = POLICY_ROOT / "phoenix_tap_worker"
        python = str(self.base_args.get("phoenix_python") or "")
        if not python or Path(python) == launcher:
            python = self.policy.default_worker_python()
        self.features_path = Path(tempfile.mkdtemp(prefix="longhorizon_")) / "features.npz"
        os.environ[FEATURES_OUT] = str(self.features_path)
        os.environ[LAYERS_ENV] = ",".join(str(layer) for layer in self.detector.features["layers"])
        os.environ[WAM_PYTHON] = python
        self.base_args["phoenix_python"] = str(launcher)
        trained = self.detector.features.get("prompts") or {}
        for task in self.settings.tasks:
            prompt = self.task_args(task)["prompt"]
            if trained.get(task) and prompt != trained[task]:
                raise ValueError(
                    f"{task}: the detector's features were taken with the prompt {trained[task]!r}, not {prompt!r}"
                )

    def _check_features_checkpoint(self, model) -> None:
        """Phoenix features belong to one checkpoint: the detector was trained on those of features.json's."""
        info, want = getattr(model, "info", None) or {}, self.detector.features
        if not info:
            return
        same = (
            int(info.get("step", -1)) == int(want["step"])
            and Path(str(info.get("checkpoint"))).name == Path(str(want["checkpoint"])).name
        )
        if not same:
            raise ValueError(
                f"the done detector {self.settings.detector} was trained on the features of {want['checkpoint']} "
                f"(step {want['step']}), this run loads {info.get('checkpoint')} (step {info.get('step')}): extract "
                "the features of these weights (wam_features.py extract) and train a detector on them"
            )

    def __getattr__(self, name):
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    # ---- subtask settings -------------------------------------------------------------------------------------------

    def prompt_of(self, task: str, args: dict) -> str:
        """The subtask's prompt: long_horizon_prompts, else the adapter's trained prompt table (<name>_prompts), else
        the task table's (task_models.<task>.prompt)."""
        if task in self.settings.prompts:
            return str(self.settings.prompts[task])
        table = args.get(f"{self.name}_prompts") or {}
        if table.get(task):
            return str(table[task])
        return str(((args.get("task_models") or {}).get(task) or {}).get("prompt") or "")

    def task_args(self, task: str) -> dict:
        args = dict(self.base_args)
        args.update(self.settings.task_overrides.get(task) or {})
        args["task_name"] = task
        args["prompt"] = self.prompt_of(task, args)
        args[f"{self.name}_prompt"] = args["prompt"]  # phoenix_prompt / internw0_prompt override the env instruction
        return args

    def _build(self, args: dict):
        old = self.inner
        kept = getattr(self.policy, "kept_worker", None)  # run_real's keep_worker_loaded (Phoenix)
        if old is not None and kept is not None:
            kept["client"] = old  # its get_model keeps the worker for the same weights, or closes it
            new = self.policy.get_model(args)
        elif (
            old is not None
            and hasattr(old, "detach")
            and hasattr(self.policy, "worker_key")
            and getattr(old, "worker_key", None) == self.policy.worker_key(args)
        ):
            new = type(old)(args, worker=old.detach())  # Phoenix: same weights, the client built again on its worker
        else:
            self._close_inner()
            new = self.policy.get_model(args)
        counts = getattr(old, "stall_counts", None) if old is not None else None
        if counts is not None and getattr(new, "stall_counts", None) is not None:
            new.stall_counts = counts  # run_real labels an episode with the difference of one running count
        if self.prepare is not None:
            self.prepare(new, args)
        if self.features_path is not None:
            self._check_features_checkpoint(new)
        self.inner_args = args
        return new

    def _retarget(self, args: dict) -> bool:
        """The running model takes ``args`` live (prompt, live-settable steps); False if it has to be built again."""
        setters = {
            f"{self.name}_steps": "set_execution_steps",
            f"{self.name}_denoise_steps": "set_denoise_steps",
        }
        changed = {
            key
            for key in set(args) | set(self.inner_args)
            if args.get(key) != self.inner_args.get(key) and key not in (*PROMPT_KEYS, f"{self.name}_prompt")
        }
        if any(key not in setters or not hasattr(self.inner, setters[key]) for key in changed):
            return False
        for key in changed:
            getattr(self.inner, setters[key])(int(args[key]))
        if hasattr(self.inner, "prompt"):
            self.inner.prompt = args["prompt"]
        self.inner_args = args
        return True

    def _switch(self, index: int) -> None:
        self.index = index
        args = self.task_args(self.settings.tasks[index])
        if not self._retarget(args):
            self.inner = self._build(args)
        self._reset_inner()

    def _settle_inner(self) -> None:
        future = getattr(self.inner, "pending_future", None)  # Phoenix's background chunk (phoenix_async real)
        if future is not None:
            with contextlib.suppress(Exception):
                future.result(timeout=30)

    def _reset_inner(self) -> None:
        self._settle_inner()
        self.policy.reset_model(self.inner)

    def _close_inner(self) -> None:
        if self.inner is None:
            return
        self._settle_inner()
        if hasattr(self.policy, "close_model"):
            self.policy.close_model(self.inner)
        elif hasattr(self.inner, "close"):
            self.inner.close()
        self.inner = None

    # ---- episode ----------------------------------------------------------------------------------------------------

    @property
    def task(self) -> str:
        return self.settings.tasks[self.index]

    def threshold(self, task: str) -> float:
        if self.settings.threshold is not None:
            return float(self.settings.threshold)
        return self.detector.threshold(task) if self.detector is not None else 1.0

    def request_next(self) -> None:
        """End the running subtask by hand (an operator key); thread-safe."""
        self.next_requested.set()

    def _start_episode(self, obs) -> None:
        self.home = qpos_of(obs).copy()  # the pose every subtask starts from
        self.episode_steps = 0
        self.records = []
        self.started = True
        if self.planner is None:
            if self.index != 0:
                self._switch(0)
            self._begin_subtask(obs)
            return
        self._take_pending()
        self.mem = pp.MemoryTracker(progress=self.uses_progress)
        self.memory, self.finished, self.plans = self.mem.text(), self.mem.finished, []
        self.pending_plan = None
        self.top_frames.reset()
        self._choose_next(obs)

    def _planner_images(self, obs) -> list[np.ndarray]:
        """The three cameras now, then the top camera 3, 2 and 1 s before."""
        images = frame_of(obs)[0]
        self.top_frames.note(self.episode_steps, images["cam_high"])
        return [images[cam] for cam in CAMERAS] + self.top_frames.past(self.episode_steps, images["cam_high"])

    def _log_plan(self, result: dict, running: str | None) -> None:
        probs = result.get("probs") or {}
        self.plans.append({"step": self.episode_steps, "running": running, "skill": result["skill"],
                           "memory": result["memory"], "ms": round(1000 * result["infer_s"]),
                           **({"p_run": round(probs[running], 4)} if running in probs else {}),
                           **({"greedy": result["greedy_skill"]} if "greedy_skill" in result else {})})  # fmt: skip

    def _choose_next(self, obs) -> None:
        """Ask the planner what to do, nothing running (the episode's start, or after n / a timeout); start it."""
        result = None
        for _ in range(3):  # a reply naming no skill: ask again
            result = self.planner.plan(self._planner_images(obs), self.settings.goal, None, self.memory)
            self._log_plan(result, None)
            if result["skill"] is not None:
                break
        skill = result["skill"]
        if skill is None or skill == pp.DONE:
            self.phase = "finished"
            why = "the goal is finished" if skill == pp.DONE else "no valid reply"
            print(f"[long_horizon] planner: {why} ({result['reply']!r}); holding the start pose", flush=True)
            return
        if self.index != self.settings.tasks.index(skill) or not self.records:
            self._switch(self.settings.tasks.index(skill))
        self._begin_subtask(obs)

    def _consult_planner(self, obs) -> str | None:
        """After a chunk: read the planner's reply about the chunk before (asked in the background meanwhile), then ask
        about this one. "planner" once --planner_confirm replies in a row name another skill."""
        future, decision = self.pending_plan, None
        if future is not None and future.done():
            self.pending_plan = None
            result = self._thresholded(future.result())  # a worker error surfaces here
            self._log_plan(result, self.task)
            if self.uses_progress:
                self.mem.observe(result["memory"])
                self.memory = self.mem.text()
            if result["skill"] is not None and result["skill"] != self.task:
                self.plan_run, self.proposal = self.plan_run + 1, result
            else:
                self.plan_run = 0
            if self.plan_run >= int(self.settings.planner_confirm):
                decision = "planner"
        since = (self.episode_steps - self.last_ask_step) / float(self.settings.control_hz)
        if decision is None and self.pending_plan is None and since >= float(self.settings.planner_every_s):
            self.last_ask_step = self.episode_steps
            self.pending_plan = self.planner.submit(
                self._planner_images(obs), self.settings.goal, self.task, self.memory
            )
        return decision

    def _thresholded(self, result: dict) -> dict:
        """The reply with its skill decided by planner_threshold on the running skill's probability (if set and the
        worker sent the probabilities); the greedy skill is kept as greedy_skill."""
        threshold, probs = float(self.settings.planner_threshold), result.get("probs")
        if threshold <= 0 or not probs or self.task not in probs:
            return result
        p_run = probs[self.task]
        other = max((name for name in probs if name != self.task), key=probs.get)
        skill = self.task if p_run >= threshold else other
        return {**result, "skill": skill, "greedy_skill": result["skill"], "p_run": round(p_run, 4)}

    def _begin_subtask(self, obs) -> None:
        self.phase = "run"
        self.subtask_steps = 0
        self.above = 0
        self.tracker = self.detector.tracker(self.task) if self.detector is not None else None
        self.pending: list[tuple] = []  # (images or features, qpos, seconds since the one before) for the detector
        self.last_feature_step = None
        self.detector_s = 0.0
        self.next_requested.clear()
        self.plan_run, self.proposal, self.pending_plan = 0, None, None
        self.last_ask_step = -(10**9)  # the first chunk of a skill is asked about at once
        if self.planner is not None:  # the short-term memory starts with the skill, as in its training episodes
            self.top_frames.reset()
        self.records.append(
            {
                "task": self.task,
                "prompt": self.inner_args["prompt"],
                "start_step": self.episode_steps,
            }
        )
        if self.on_skill is not None:
            try:
                self.on_skill(self.task)
            except Exception as exc:  # noqa: BLE001 - the chain goes on with the settings it had
                print(f"[long_horizon] on_skill({self.task}) failed: {exc}", flush=True)
        if self.tracker is not None and self.features_path is None:
            self.pending.append((*frame_of(obs), 0.0))
        if self.features_path is not None:  # the worker's count so far: the next inference is this subtask's
            self.feature_id = (read_features(self.features_path) or {}).get("id")
        print(f"[long_horizon] subtask {self.index + 1}/{len(self.settings.tasks)}: {self.task} "
              f"({self.inner_args['prompt']!r}) from step {self.episode_steps}", flush=True)  # fmt: skip

    def on_policy_step(self, obs) -> None:
        """Every env step the inner adapter takes (StepTap)."""
        self.episode_steps += 1
        self.subtask_steps += 1
        if self.planner is not None and self.top_frames.due(self.episode_steps):
            self.top_frames.note(self.episode_steps, extract_image(obs["sensor"]["cam_high"]["color"]))
        if self.tracker is not None and self.features_path is None and self.subtask_steps % self.stride == 0:
            self.pending.append((*frame_of(obs), self.stride / float(self.settings.control_hz)))

    def _stall_changes(self) -> int:
        counts = getattr(self.inner, "stall_counts", None)  # Phoenix's phoenix_stall_* counters
        return int(counts["changed"]) if counts else 0

    def _take_features(self, qpos: np.ndarray, step: int, changed_before: int) -> None:
        """Phoenix source: the features of the inference this chunk ran on (its starting observation)."""
        latest = read_features(self.features_path)
        if latest is None or latest["id"] == self.feature_id:
            return  # no new inference in this call (async: the next chunk has not landed yet)
        self.feature_id = latest["id"]
        if latest["padded"] or self._stall_changes() != changed_before:
            return  # the adapter covered or masked the top camera for this chunk: not what the detector learnt
        dt = (
            0.0 if self.last_feature_step is None else (step - self.last_feature_step) / float(self.settings.control_hz)
        )
        self.last_feature_step = step
        self.pending.append((latest["features"], qpos, dt))

    def _detect(self) -> bool:
        if self.tracker is None or not self.pending:
            return False
        started = time.monotonic()
        items, self.pending = self.pending, []
        states, dts = [s for _, s, _ in items], [d for _, _, d in items]
        if self.features_path is None:
            probs = self.tracker.push([f for f, _, _ in items], states, dts)
        else:
            probs = self.tracker.push_features(np.stack([f for f, _, _ in items]), states, dts)
        self.detector_s += time.monotonic() - started
        threshold = self.threshold(self.task)
        for prob in probs:
            self.above = self.above + 1 if prob >= threshold else 0
        return self.above >= int(self.settings.confirm) and self.subtask_steps >= int(self.settings.min_steps)

    def _end_subtask(self, reason: str) -> None:
        record = self.records[-1]
        probs = self.tracker.probs if self.tracker is not None else []
        record.update(
            end_step=self.episode_steps,
            steps=self.subtask_steps,
            reason=reason,
            threshold=self.threshold(self.task) if self.tracker is not None else None,
            max_prob=round(max(probs), 4) if probs else None,
            probs=[round(p, 3) for p in probs],
            detector_ms=round(1000 * self.detector_s / max(1, len(probs)), 1),
        )
        print(f"[long_horizon] {self.task} ended at step {self.episode_steps} ({reason}, "
              f"{self.subtask_steps} steps, max p {record['max_prob']})", flush=True)  # fmt: skip
        self._reset_inner()
        stop = reason == "timeout" and self.settings.on_timeout == "stop"
        if self.planner is None:
            last = self.index == len(self.settings.tasks) - 1
            self.after_home = "finish" if last or stop else "next"
        else:
            self.next_task = None  # n, a timeout: asked once the arms are back (nothing running)
            if reason in ("planner", "operator"):  # the running skill's job is finished
                self.mem.finish(self.task)
                self.memory = self.mem.text()
            else:
                self.mem.restart()
                self.memory = self.mem.text()
            if reason == "planner":
                self.next_task = self.proposal["skill"]
                if not self.uses_progress:
                    self.memory = self.proposal["memory"] or self.memory  # what the planner wrote, as in MEM
            record.update(next=self.next_task, memory=self.memory, plans=self.plans)
            self.plans, self.pending_plan = [], None
            self.after_home = "finish" if stop or self.next_task == pp.DONE else "next"
        self.phase = "home" if self.settings.return_home else "between"
        self.home_queue = None

    def _home_targets(self, obs) -> list[np.ndarray]:
        """From the measured pose to the start pose at home_rad_per_step per step (grippers alongside), then held."""
        current = qpos_of(obs).astype(np.float64)
        home = self.home.astype(np.float64)
        arm = [i for i in range(len(home)) if i not in GRIPPER_DIMS]
        distance = float(np.abs(home[arm] - current[arm]).max()) / float(self.settings.home_rad_per_step)
        steps = max(1, math.ceil(distance - 1e-6))  # float32 qpos: 0.20000002 rad at 0.05 is 4 steps
        path = [current + (home - current) * (k / steps) for k in range(1, steps + 1)]
        return path + [home] * int(self.settings.settle_steps)

    def _send(self, env, state: np.ndarray):
        action = np.asarray(state, dtype=np.float32).copy()
        action[GRIPPER_DIMS] *= float(self.settings.gripper_scale)  # opening fraction -> the env's metres per finger
        if hasattr(self.policy, "format_env_action"):
            action = self.policy.format_env_action(action, env)
        result = env.step(action)
        self.episode_steps += 1
        if self.planner is not None and self.top_frames.due(self.episode_steps):
            self.top_frames.note(self.episode_steps, extract_image(result[0]["sensor"]["cam_high"]["color"]))
        return result

    def _drive_home(self, env, obs):
        """Up to home_steps_per_call steps toward the start pose; at the end the next subtask starts (or the chain
        is over and the arms hold the pose)."""
        info, truncated = {}, False
        if self.home_queue is None:
            self.home_queue = self._home_targets(obs) if self.phase == "home" else []
        for _ in range(int(self.settings.home_steps_per_call)):
            if not self.home_queue:
                break
            obs, _reward, terminated, truncated, info = self._send(env, self.home_queue.pop(0))
            if _any_true(terminated) or _any_true(truncated):
                return obs, info, truncated
        if not self.home_queue:
            if self.after_home == "next" and self.planner is not None and self.next_task is None:
                self._choose_next(obs)
            elif self.after_home == "next":
                index = self.index + 1 if self.planner is None else self.settings.tasks.index(self.next_task)
                self._switch(index)
                self._begin_subtask(obs)
            else:
                self.phase = "finished"
                print(
                    f"[long_horizon] chain over after {self.task}: holding the start pose",
                    flush=True,
                )
        return obs, info, truncated

    def _hold(self, env, obs):
        info, truncated = {}, False
        for _ in range(int(self.settings.home_steps_per_call)):
            obs, _reward, terminated, truncated, info = self._send(env, self.home)
            if _any_true(terminated) or _any_true(truncated):
                break
        return obs, info, truncated

    def eval(self, env, obs):
        step = env_step_index(env)
        if not self.started or (step is not None and step < self.last_step) or (step is None and self.reset_pending):
            self._start_episode(obs)
        elif self.reset_pending:  # back from a takeover: the operator moved the arms
            if self.phase == "run" and self.tracker is not None and self.features_path is None:
                self.pending.append((*frame_of(obs), self.stride / float(self.settings.control_hz)))
            self.home_queue = None
        self.reset_pending = False
        times: list[float] = []
        if self.phase in ("home", "between"):
            obs, info, truncated = self._drive_home(env, obs)
        elif self.phase == "finished":
            obs, info, truncated = self._hold(env, obs)
            if self.settings.stop_when_finished:
                truncated = True
        else:
            tap = StepTap(env, self, self.inner_args["prompt"])
            chunk = (qpos_of(obs), self.subtask_steps, self._stall_changes())
            obs, info, truncated, times = self.policy.eval(tap, self.inner, obs)
            if self.features_path is not None and self.tracker is not None:
                self._take_features(*chunk)
            reason = None
            if self.next_requested.is_set():
                reason = "operator"
            elif self.planner is not None:
                reason = self._consult_planner(obs)
            elif self._detect():
                reason = "detector"
            if reason is None and 0 < int(self.settings.max_steps) <= self.subtask_steps:
                reason = "timeout"
            if reason is not None:
                self._end_subtask(reason)
        after = env_step_index(env)
        self.last_step = after if after is not None else self.last_step
        return obs, info, truncated, times

    def reset(self) -> None:
        """reset_model: the inner model's chunk state; the chain goes on (run_real resets at every takeover). A new
        episode is told by the env's step count going back (or, without one, by this reset)."""
        self.reset_pending = True
        self._reset_inner()

    def episode_summary(self) -> dict:
        done = sum(1 for r in self.records if r.get("reason") in ("detector", "operator", "planner"))
        summary = {
            "tasks": list(self.settings.tasks),
            "subtasks": [dict(r) for r in self.records],
            "ended": done,
            "finished": getattr(self, "phase", None) == "finished",
        }
        if self.planner is not None:
            summary.update(goal=self.settings.goal, memory=getattr(self, "memory", None),
                           pending_plans=list(getattr(self, "plans", [])))  # fmt: skip
        return summary

    def close(self) -> None:
        self._close_inner()
        self.close_planners()
        if self.features_path is not None:
            for name in (FEATURES_OUT, LAYERS_ENV, WAM_PYTHON):
                os.environ.pop(name, None)
            shutil.rmtree(self.features_path.parent, ignore_errors=True)


class LongHorizonPolicy:
    """The adapter interface (get_model / eval / reset_model / close_model) around another adapter module; anything
    else is the inner module's."""

    def __init__(self, inner_policy, tasks: list[str] | None = None, prepare=None):
        self.inner_policy, self.tasks, self.prepare = inner_policy, tasks, prepare

    def __getattr__(self, name):
        return getattr(self.inner_policy, name)

    def get_model(self, usr_args: dict) -> LongHorizonModel:
        settings = Settings.from_usr_args(usr_args, self.tasks)
        return LongHorizonModel(self.inner_policy, usr_args, settings, self.prepare)

    def eval(self, env, model: LongHorizonModel, obs):
        return model.eval(env, obs)

    def reset_model(self, model: LongHorizonModel) -> None:
        model.reset()

    def close_model(self, model: LongHorizonModel) -> None:
        model.close()
