# ruff: noqa: RUF001 - Chinese goals and memories as test data
"""policy/longhorizon without a world model, robot or GPU: the subtask chain around a fake adapter, the detector's
directory round trip, and the training labels and threshold rule.

    CUDA_VISIBLE_DEVICES= policy/fmc3_no1/.venv/bin/python -m pytest -q policy/longhorizon/tests
"""

from concurrent.futures import Future
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from policy.longhorizon import success_moments as sm  # noqa: E402
from policy.longhorizon import add_hard_negatives as ahn  # noqa: E402
from policy.longhorizon import build_planner_data as bpd  # noqa: E402
from policy.longhorizon import build_mem_samples as bms  # noqa: E402
from policy.longhorizon import deploy_policy as meta  # noqa: E402
from policy.longhorizon import eval_planner as ep  # noqa: E402
from policy.longhorizon import long_horizon as lh  # noqa: E402
from policy.longhorizon import planner  # noqa: E402
from policy.longhorizon import planner_prompt as pp  # noqa: E402
from policy.longhorizon import train_done_detector as tdd  # noqa: E402
from policy.longhorizon import upstream_planner as up  # noqa: E402
from policy.longhorizon import wam_features as wf  # noqa: E402
from policy.longhorizon.done_detector import CAMERAS  # noqa: E402
from policy.longhorizon.done_detector import DoneDetector  # noqa: E402
from policy.longhorizon.done_detector import DoneHead  # noqa: E402

PROMPTS = {"click_bell": "Click the bell", "water_pouring": "Pour the water", "items_handover": "Hand the pen over"}


class FakeEnv:
    """RealCobotMagicEnv's interface: the arms reach every target at once (qpos = the target, grippers back in
    opening fractions); step_count goes back to 0 on reset (the simulator's env has none: has_step_count False)."""

    has_step_count = True

    def __init__(self, start=None):
        self.qpos = np.zeros(14, np.float32) if start is None else np.asarray(start, np.float32)
        self.steps = 0
        if self.has_step_count:
            self.step_count = 0
        self.actions = []
        self.unwrapped = SimpleNamespace(single_action_space=SimpleNamespace(shape=(14,)), device="cpu")
        self._current_instruction = "the env's own instruction"

    def obs(self):
        image = torch.zeros((1, 6, 8, 3), dtype=torch.uint8)
        return {"sensor": {cam: {"color": image} for cam in CAMERAS}, "robot": {"qpos": torch.tensor(self.qpos)[None]}}

    def reset(self, start=None):
        if start is not None:
            self.qpos = np.asarray(start, np.float32)
        self.steps = 0
        if self.has_step_count:
            self.step_count = 0
        return self.obs()

    def step(self, action):
        target = np.asarray(action, np.float32).reshape(-1)[:14].copy()
        self.actions.append(target.copy())
        target[[6, 13]] /= 0.05
        self.qpos = target
        self.steps += 1
        if self.has_step_count:
            self.step_count = self.steps
        return self.obs(), 0.0, False, False, {}

    def get_wrapper_attr(self, name):
        return lambda: False


class FakeModel:
    def __init__(self, args):
        self.args = dict(args)
        self.prompt = args.get("fake_prompt") or ""
        self.steps = int(args.get("fake_steps", 10))
        self.seen, self.resets, self.closed = [], 0, False

    def set_execution_steps(self, steps):
        self.steps = steps
        return steps


def fake_policy(target_of=None):
    """An adapter module: each chunk moves every arm joint by 0.01 rad, or to target_of(prompt)."""
    built = []

    def get_model(args):
        model = FakeModel(args)
        built.append(model)
        return model

    def eval(env, model, obs):
        model.seen.append((model.prompt, env._current_instruction))
        for _ in range(model.steps):
            q = np.asarray(obs["robot"]["qpos"][0], np.float32).copy()
            q[[0, 7]] += 0.01
            if target_of is not None:
                q = target_of(model.prompt or env._current_instruction, q)
            action = q.copy()
            action[[6, 13]] *= 0.05
            obs = env.step(action)[0]
        return obs, {}, False, [0.1]

    def reset_model(model):
        model.resets += 1

    def close_model(model):
        model.closed = True

    return SimpleNamespace(
        __name__="policy.fake.deploy_policy",
        get_model=get_model,
        eval=eval,
        reset_model=reset_model,
        close_model=close_model,
        built=built,
    )


class ScriptedDetector:
    """A done detector whose probabilities come from script(task, frame number in the subtask)."""

    fps, source = 2.0, "image"

    def __init__(self, script, tasks=tuple(PROMPTS)):
        self.script, self.tasks, self.pushed = script, list(tasks), []

    def threshold(self, task):
        return 0.5

    def tracker(self, task):
        detector = self

        class Tracker:
            def __init__(self):
                self.probs = []

            def push(self, frames, states, dts):
                start = len(self.probs)
                probs = [detector.script(task, start + k) for k in range(len(frames))]
                detector.pushed.append((task, len(frames), list(dts)))
                self.probs = self.probs + probs
                return probs

        return Tracker()


def make_model(policy, tasks, detector=None, **settings):
    args = {"policy_name": "fake", "task_name": tasks[0], "fake_prompts": PROMPTS, "fake_steps": 10}
    args.update({f"long_horizon_{k}": v for k, v in settings.items()})
    args["long_horizon_tasks"] = list(tasks)
    model = lh.LongHorizonPolicy(policy).get_model(args)
    if detector is not None:
        model.detector, model.stride = detector, 5
    return model


def run(model, env, obs, calls):
    for _ in range(calls):
        obs = model.eval(env, obs)[0]
    return obs


def test_chain_runs_each_subtask_with_its_prompt_and_goes_home_between():
    policy = fake_policy()
    # bell done from its 4th detector frame on, water never
    detector = ScriptedDetector(lambda task, k: 0.9 if task == "click_bell" and k >= 3 else 0.1)
    model = make_model(policy, ["click_bell", "water_pouring"], detector, min_steps=10)
    start = np.zeros(14, np.float32)
    start[[6, 13]] = 0.8
    env = FakeEnv(start)
    obs = env.reset()
    obs = run(model, env, obs, 2)  # 20 steps: frames 0, 5, 10, 15, 20 -> 0.9 from the 4th: 2 in a row at step 20
    assert model.records[0]["reason"] == "detector" and model.records[0]["end_step"] == 20
    assert model.phase == "home"
    assert policy.built[0].seen[0] == ("Click the bell", "Click the bell")
    obs = run(model, env, obs, 1)  # 0.2 rad off the start pose at 0.05 per step: 4 steps + 5 held = 9 steps
    np.testing.assert_allclose(env.actions[-1][[0, 7]], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(env.actions[-1][[6, 13]], [0.04, 0.04], atol=1e-6)  # 0.8 opening -> 0.04 m
    assert len(env.actions) == 29
    assert model.phase == "run" and model.task == "water_pouring"
    assert len(policy.built) == 1  # same weights: retargeted, not built again
    obs = run(model, env, obs, 1)
    assert policy.built[0].seen[-1] == ("Pour the water", "Pour the water")
    assert model.records[1]["start_step"] == 29


def test_last_subtask_holds_the_start_pose_and_episode_summary():
    policy = fake_policy()
    model = make_model(policy, ["click_bell"], ScriptedDetector(lambda task, k: 0.9), min_steps=0)
    env = FakeEnv()
    obs = run(model, env, env.reset(), 3)
    assert model.phase == "finished"
    before = len(env.actions)
    run(model, env, obs, 1)
    assert len(env.actions) == before + 10
    np.testing.assert_allclose(env.actions[-1], np.zeros(14), atol=1e-6)
    summary = model.episode_summary()
    assert summary["finished"] and summary["ended"] == 1 and summary["subtasks"][0]["probs"][:2] == [0.9, 0.9]


def test_timeout_next_and_stop():
    model = make_model(fake_policy(), ["click_bell", "water_pouring"], ScriptedDetector(lambda t, k: 0.0), max_steps=30)
    env = FakeEnv()
    run(model, env, env.reset(), 3)
    assert model.records[0]["reason"] == "timeout" and model.after_home == "next"
    stop = make_model(
        fake_policy(),
        ["click_bell", "water_pouring"],
        ScriptedDetector(lambda t, k: 0.0),
        max_steps=30,
        on_timeout="stop",
    )
    env = FakeEnv()
    run(stop, env, env.reset(), 5)
    assert stop.phase == "finished" and len(stop.records) == 1


def test_operator_ends_a_subtask_and_no_detector_needed():
    model = make_model(fake_policy(), ["click_bell", "water_pouring"])
    env = FakeEnv()
    obs = run(model, env, env.reset(), 1)
    model.request_next()
    obs = run(model, env, obs, 1)
    assert model.records[0]["reason"] == "operator"
    run(model, env, obs, 2)
    assert model.task == "water_pouring" and not model.next_requested.is_set()


def test_takeover_reset_keeps_the_chain_new_episode_restarts_it():
    detector = ScriptedDetector(lambda task, k: 0.9 if task == "click_bell" else 0.1)
    model = make_model(fake_policy(), ["click_bell", "water_pouring"], detector, min_steps=0)
    env = FakeEnv()
    obs = run(model, env, env.reset(), 4)
    assert model.task == "water_pouring"
    model.reset()  # run_real: a takeover
    for _ in range(7):  # the operator drives
        obs = env.step(np.zeros(14))[0]
    obs = run(model, env, obs, 1)
    assert model.task == "water_pouring"
    model.reset()  # the episode ends
    obs = run(model, env, env.reset(), 1)
    assert model.task == "click_bell" and len(model.records) == 1


def test_simulator_without_step_count_starts_a_new_episode_on_reset_model():
    class SimEnv(FakeEnv):
        has_step_count = False

    model = make_model(fake_policy(), ["click_bell", "water_pouring"], ScriptedDetector(lambda t, k: 0.9), min_steps=0)
    env = SimEnv()
    run(model, env, env.obs(), 4)
    assert model.task == "water_pouring"
    model.reset()
    run(model, env, env.obs(), 1)
    assert model.task == "click_bell"


def test_per_task_settings_live_or_rebuilt():
    policy = fake_policy()
    model = make_model(
        policy,
        ["click_bell", "water_pouring", "items_handover"],
        task_overrides={"click_bell": {"fake_steps": 5}, "items_handover": {"checkpoint_dir": "/other"}},
    )
    assert policy.built[0].steps == 5
    env = FakeEnv()
    obs = run(model, env, env.reset(), 1)
    model.request_next()
    obs = run(model, env, obs, 3)
    assert model.task == "water_pouring" and len(policy.built) == 1 and policy.built[0].steps == 10
    model.request_next()
    run(model, env, obs, 3)
    assert model.task == "items_handover" and len(policy.built) == 2 and policy.built[0].closed
    assert policy.built[1].args["checkpoint_dir"] == "/other" and policy.built[1].prompt == "Hand the pen over"


def test_settings_validation():
    with pytest.raises(ValueError, match="on_timeout"):
        lh.Settings.from_usr_args({"task_name": "click_bell", "long_horizon_on_timeout": "later"})
    with pytest.raises(ValueError, match="unknown tasks"):
        lh.Settings.from_usr_args({"task_name": "click_bell", "long_horizon_task_overrides": {"bell": {}}})
    settings = lh.Settings.from_usr_args({"task_name": "x", "long_horizon_tasks": "click_bell, water_pouring"})
    assert settings.tasks == ["click_bell", "water_pouring"]


@pytest.mark.skipif(
    not (Path(meta.__file__).resolve().parents[1] / "phoenix" / "deploy_policy.yml").is_file(),
    reason="needs the inner adapter (policy/phoenix of the RoboSynChallenge codebase)",
)
def test_meta_adapter_fills_the_inner_adapters_settings():
    args = meta.inner_args({"inner_policy": "phoenix", "phoenix_steps": 20, "checkpoint_dir": ""})
    assert args["policy_name"] == "phoenix" and args["phoenix_steps"] == 20
    assert args["phoenix_prompts"]["water_pouring"].startswith("Pick the cup")
    with pytest.raises(ValueError, match="inner_policy"):
        meta.inner_args({"inner_policy": "pi0"})


def write_detector(root: Path, bias: float) -> Path:
    """A detector directory with a toy encoder (per-channel means) and a head whose logit is ``bias``."""

    class MeanEncoder(torch.nn.Module):
        def forward(self, x):
            return torch.cat([x.mean((2, 3)), x.std((2, 3))], dim=1)

    root.mkdir(parents=True, exist_ok=True)
    torch.jit.save(torch.jit.trace(MeanEncoder(), torch.randn(2, 3, 16, 16)), str(root / "encoder.pt"))
    meta_json = {"image_size": 16, "mean": [0.5] * 3, "std": [0.25] * 3, "feature_dim": 6}
    (root / "encoder.json").write_text(json.dumps(meta_json))
    head = DoneHead(feature_dim=6, hidden=8, n_tasks=2)
    torch.nn.init.zeros_(head.out.weight)
    torch.nn.init.constant_(head.out.bias, bias)
    torch.save(head.state_dict(), root / "head.pt")
    config = {
        "tasks": ["click_bell", "water_pouring"],
        "head": {"feature_dim": 6, "n_cameras": 3, "state_dim": 14, "hidden": 8, "n_tasks": 2, "dropout": 0.1},
        "fps": 2.0,
        "thresholds": {"click_bell": 0.7},
    }
    (root / "config.json").write_text(json.dumps(config))
    return root


def test_detector_directory_round_trip(tmp_path):
    detector = DoneDetector(write_detector(tmp_path / "det", 3.0), device="cpu")
    assert detector.threshold("click_bell") == 0.7 and detector.threshold("water_pouring") == 0.5
    tracker = detector.tracker("water_pouring")
    frame = {cam: np.full((48, 64, 3), 128, np.uint8) for cam in CAMERAS}
    probs = tracker.push([frame, frame], [np.zeros(14), np.zeros(14)], [0.0, 0.5])
    np.testing.assert_allclose(probs, [1 / (1 + np.exp(-3.0))] * 2, rtol=1e-5)
    with pytest.raises(ValueError, match="knows"):
        detector.tracker("item_assembly")


def test_chain_with_a_real_detector_directory(tmp_path):
    policy = fake_policy()
    args = {"policy_name": "fake", "task_name": "click_bell", "fake_prompts": PROMPTS, "fake_steps": 10}
    args.update(
        long_horizon_tasks=["click_bell", "water_pouring"],
        long_horizon_detector=str(write_detector(tmp_path / "det", 3.0)),
        long_horizon_device="cpu",
        long_horizon_min_steps=0,
    )
    model = lh.LongHorizonPolicy(policy).get_model(args)
    assert model.stride == 5
    env = FakeEnv()
    run(model, env, env.reset(), 1)  # frames at steps 0, 5, 10 all at p 0.95 >= 0.7, 0.5 s apart
    assert model.records[0]["reason"] == "detector" and model.records[0]["threshold"] == 0.7
    args["long_horizon_tasks"] = ["click_bell", "sample_loading"]
    with pytest.raises(ValueError, match="not trained on"):
        lh.LongHorizonPolicy(policy).get_model(args)


def test_frame_labels():
    t = np.arange(0, 200, 5)
    success = {"success": True, "fps": 10.0, "length": 200}
    labels = tdd.frame_labels(success, t, pos_tail_s=1.5)
    assert (labels[t >= 184] == 1).all() and (labels[t < 80] == 0).all() and (labels[(t >= 80) & (t < 184)] == -1).all()
    audited = tdd.frame_labels({**success, "success_frame": 100}, t, pos_tail_s=1.5)
    assert (
        (audited[t >= 100] == 1).all() and (audited[t < 90] == 0).all() and (audited[(t >= 90) & (t < 100)] == -1).all()
    )
    assert (tdd.frame_labels({**success, "success": False}, t, pos_tail_s=1.5) == 0).all()
    bell = tdd.frame_labels({"success": True, "fps": 10.0, "length": 61}, np.arange(0, 61, 5), 1.5)  # 6 s
    np.testing.assert_array_equal(
        bell, [0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1]
    )  # not done up to 2.4 s, done from 4.5 s


def test_runtime_rule_replay_and_threshold_choice():
    t = np.arange(0, 100, 5)
    probs = np.where(t >= 80, 0.9, 0.2)
    labels = tdd.frame_labels({"success": True, "fps": 10.0, "length": 100}, t, pos_tail_s=1.5)
    assert tdd.fire_frame(probs, t, 0.5, 2, 0) == list(t).index(85)
    assert tdd.episode_outcome(probs, t, labels, success=True, threshold=0.5, confirm=2, min_steps=0)[0] == "caught"
    assert tdd.episode_outcome(probs, t, labels, success=False, threshold=0.5, confirm=2, min_steps=0)[0] == "false"
    early = np.full(len(t), 0.6)
    assert tdd.episode_outcome(early, t, labels, success=True, threshold=0.5, confirm=2, min_steps=0)[0] == "early"
    rows = [(probs, t, labels, True, 10.0), (np.full(len(t), 0.3), t, np.zeros(len(t), np.int8), False, 10.0)]
    best = tdd.choose_threshold(rows, confirm=2, min_steps=0, max_false=0.0)
    assert best["threshold"] == 0.31 and best["caught"] == 1 and best["met_max_false"]


# ----- Phoenix's own features ----------------------------------------------------------------------------------------


def test_pool_views_takes_the_current_frame_per_camera():
    # 2 frames of a 2 x 6 token grid (3 views of 2 columns), D = 1: token value = view index + 10 x frame
    grid = np.array([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]], np.float32)
    hidden = torch.tensor(np.concatenate([grid.ravel(), grid.ravel() + 10]))[None, :, None]
    pooled = wf.pool_views(hidden.to(torch.bfloat16), tokens_per_frame=12, grid_h=2, grid_w=6)
    np.testing.assert_allclose(pooled[:, 0].numpy(), [0, 1, 2])
    with pytest.raises(ValueError, match="token grid"):
        wf.pool_views(hidden, tokens_per_frame=12, grid_h=3, grid_w=4)


class FakePhoenixModel:
    """Phoenix's model as the tap sees it: video_expert.pre_dit gives the token grid, mot.forward the tokens and,
    with capture_layers, each captured layer's hidden states (layer L: every token = L + view)."""

    def __init__(self):
        self.mot_calls, self.captures = 0, []
        grid = np.repeat(np.array([[0, 0, 1, 1, 2, 2]], np.float32), 2, axis=0).ravel()

        def pre_dit(**kwargs):
            return {"meta": {"grid_size": (2, 2, 6), "tokens_per_frame": 12}}

        def forward(**kwargs):
            self.mot_calls += 1
            tokens = {"video": torch.zeros(1, 24, 4)}
            layers = (kwargs.get("capture_layers") or {}).get("video")
            if layers is None:
                return tokens
            self.captures.append(sorted(layers))
            hidden = torch.tensor(np.concatenate([grid, grid]))[None, :, None].repeat(1, 1, 4)
            return tokens, {"video": {layer: hidden + layer for layer in layers}}

        self.video_expert = SimpleNamespace(pre_dit=pre_dit)
        self.mot = SimpleNamespace(forward=forward)

    def infer(self, steps=3):
        """One inference: pre_dit and a MoT pass per denoising step."""
        for _ in range(steps):
            self.video_expert.pre_dit()
            self.mot.forward(embeds_all={})


def test_tap_captures_the_first_mot_pass_of_an_armed_inference_only():
    model = FakePhoenixModel()
    tap = wf.PhoenixFeatureTap(model, layers=(10, 20))
    model.infer()
    assert tap.features is None and model.captures == []  # not armed: the model runs as it is
    tap.arm()
    model.infer()
    assert model.captures == [[10, 20]] and model.mot_calls == 6
    assert tap.features.shape == (6, 4) and tap.features.dtype == np.float16
    np.testing.assert_allclose(tap.features[:, 0], [10, 11, 12, 20, 21, 22])  # layer-major, then the 3 views
    stop_model = FakePhoenixModel()
    stop = wf.PhoenixFeatureTap(stop_model, layers=(5,), stop=True)
    stop.arm()
    with pytest.raises(wf.StopAfterCapture):  # the extraction stops the inference right after the first pass
        stop_model.infer()
    assert stop_model.mot_calls == 1 and stop.features is not None


def test_worker_shim_writes_every_acts_features(tmp_path, monkeypatch):
    """wam_features.py worker: a stand-in phoenix_worker.py (Runtime + main), run as the adapter would start it."""
    script = tmp_path / "phoenix_worker.py"
    script.write_text(
        "import sys\n"
        "class Runtime:\n"
        "    def __init__(self, args):\n"
        "        self.model = MODEL\n"
        "    def act(self, obs, instruction, *, pad_cameras=()):\n"
        "        self.model.infer()\n"
        "        return [instruction]\n"
        "def main():\n"
        "    rt = Runtime(sys.argv[1:])\n"
        "    RESULTS.append(rt.act({}, 'Click the bell'))\n"
        "    RESULTS.append(rt.act({}, 'Click the bell', pad_cameras=['cam_high']))\n"
    )
    out = tmp_path / "features.npz"
    monkeypatch.setenv(wf.FEATURES_OUT, str(out))
    monkeypatch.setenv(wf.LAYERS_ENV, "3,7")
    model, results = FakePhoenixModel(), []
    real_loader = wf.load_worker_module

    def loader(path):
        module = real_loader(path)
        module.MODEL, module.RESULTS = model, results
        return module

    monkeypatch.setattr(wf, "load_worker_module", loader)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    wf.run_worker([str(script), "--run-dir", "x"])
    assert results == [["Click the bell"], ["Click the bell"]]  # the worker's replies are untouched
    latest = wf.read_features(out)
    assert latest["id"] == (os.getpid(), 2) and latest["padded"] and latest["features"].shape == (6, 4)
    assert model.captures == [[3, 7], [3, 7]]


def write_phoenix_detector(root: Path, bias: float, *, step: int = 9429) -> Path:
    """A detector on Phoenix's features: 2 layers x 3 views of 4-D, head logit ``bias``."""
    root.mkdir(parents=True, exist_ok=True)
    head = DoneHead(feature_dim=4, n_cameras=6, hidden=8, n_tasks=2)
    torch.nn.init.zeros_(head.out.weight)
    torch.nn.init.constant_(head.out.bias, bias)
    torch.save(head.state_dict(), root / "head.pt")
    config = {
        "source": "phoenix",
        "tasks": ["click_bell", "water_pouring"],
        "head": {"feature_dim": 4, "n_cameras": 6, "state_dim": 14, "hidden": 8, "n_tasks": 2, "dropout": 0.1},
        "fps": 1.0,
        "thresholds": {},
    }
    (root / "config.json").write_text(json.dumps(config))
    features = {"source": "phoenix", "checkpoint": "/a100/run/checkpoints/weights/slot_5.pt", "step": step,
                "layers": [3, 7], "prompts": {"click_bell": "Click the bell", "water_pouring": "Pour the water"}}  # fmt: skip
    (root / "features.json").write_text(json.dumps(features))
    return root


def fake_phoenix_policy(stalled_chunks=()):
    """policy/phoenix's adapter as the wrapper sees it: phoenix_python, default_worker_python, a client with info and
    stall counts; its "worker" writes each chunk's features where wam_features would."""
    policy = fake_policy()
    policy.__name__ = "policy.phoenix.deploy_policy"
    policy.default_worker_python = lambda: "/venvs/phoenix/bin/python"
    get_model, adapter_eval = policy.get_model, policy.eval
    calls = []

    def phoenix_get_model(args):
        model = get_model(args)
        model.info = {"checkpoint": "/bench/run/checkpoints/weights/slot_5.pt", "step": 9429}
        model.stall_counts = {"stalled": 0, "changed": 0}
        return model

    def phoenix_eval(env, model, obs):
        calls.append(1)
        if len(calls) in stalled_chunks:
            model.stall_counts["changed"] += 1  # the adapter covered the top camera for this chunk
        wf.write_features(os.environ[wf.FEATURES_OUT], np.ones((6, 4), np.float16), len(calls), padded=False)
        return adapter_eval(env, model, obs)

    policy.get_model, policy.eval, policy.calls = phoenix_get_model, phoenix_eval, calls
    return policy


def phoenix_args(detector, **settings):
    args = {"policy_name": "phoenix", "task_name": "click_bell", "phoenix_prompts": dict(PROMPTS), "fake_steps": 10}
    args.update(long_horizon_tasks=["click_bell", "water_pouring"], long_horizon_detector=str(detector),
                long_horizon_device="cpu", long_horizon_min_steps=0)  # fmt: skip
    args.update({f"long_horizon_{k}": v for k, v in settings.items()})
    return args


def test_phoenix_source_runs_the_worker_through_the_tap_and_reads_one_feature_per_chunk(tmp_path, monkeypatch):
    for name in (wf.FEATURES_OUT, wf.LAYERS_ENV, lh.WAM_PYTHON):
        monkeypatch.delenv(name, raising=False)
    policy = fake_phoenix_policy(stalled_chunks=(2,))
    model = lh.LongHorizonPolicy(policy).get_model(phoenix_args(write_phoenix_detector(tmp_path / "det", 3.0)))
    built = policy.built[0]
    assert built.args["phoenix_python"] == str(lh.POLICY_ROOT / "phoenix_tap_worker")
    assert os.environ[lh.WAM_PYTHON] == "/venvs/phoenix/bin/python" and os.environ[wf.LAYERS_ENV] == "3,7"
    env = FakeEnv()
    obs = run(model, env, env.reset(), 1)  # chunk 1: its inference's features (p 0.95), one in a row so far
    assert len(model.tracker.probs) == 1
    obs = run(model, env, obs, 1)  # chunk 2: the top camera covered, its features dropped
    assert len(model.tracker.probs) == 1
    run(model, env, obs, 1)  # chunk 3: the second feature, 2 s after the first -> 2 in a row: done
    assert model.records[0]["reason"] == "detector" and len(model.records[0]["probs"]) == 2
    path = model.features_path
    model.close()
    assert not path.exists() and wf.FEATURES_OUT not in os.environ


def test_phoenix_source_refuses_other_weights_prompts_or_models(tmp_path, monkeypatch):
    for name in (wf.FEATURES_OUT, wf.LAYERS_ENV, lh.WAM_PYTHON):
        monkeypatch.delenv(name, raising=False)
    other = phoenix_args(write_phoenix_detector(tmp_path / "other", 3.0, step=9000))
    with pytest.raises(ValueError, match="step 9000"):
        lh.LongHorizonPolicy(fake_phoenix_policy()).get_model(other)
    detector = write_phoenix_detector(tmp_path / "det", 3.0)
    with pytest.raises(ValueError, match="taken with the prompt"):
        lh.LongHorizonPolicy(fake_phoenix_policy()).get_model(
            phoenix_args(detector, prompts={"water_pouring": "pour it"})
        )
    args = phoenix_args(detector)
    args["policy_name"] = "fake"
    with pytest.raises(ValueError, match="reads Phoenix's features"):
        lh.LongHorizonPolicy(fake_policy()).get_model(args)


def test_trainer_views_subsample_with_dt_and_read_the_source():
    episode = {
        "task": 0,
        "row": {"fps": 10.0},
        "features": np.arange(5)[:, None, None].astype(np.float16),
        "state": np.zeros((5, 14), np.float32),
        "frame_index": np.array([0, 10, 20, 30, 40]),
        "labels": np.array([0, 0, -1, 1, 1], np.int8),
    }
    full, half = tdd.view_of(episode), tdd.view_of(episode, 2, 1)
    np.testing.assert_allclose(full["dt"], [0, 1, 1, 1, 1])
    np.testing.assert_array_equal(half["frame_index"], [10, 30])
    np.testing.assert_allclose(half["dt"], [0, 2])
    np.testing.assert_array_equal(half["labels"], [0, 1])
    mean, std, _s_mean, _s_std = tdd.normalization([episode, episode])
    assert mean.shape == (1, 1) and np.isclose(mean[0, 0], 2.0) and np.isclose(std[0, 0], np.sqrt(2.0), atol=1e-5)


# ----- planner mode (π0.5-style high level with MEM-style memory) -----------------------------------------------------


class FakePlanner:
    """PlannerClient's interface; reply(running, memory, call) -> (skill, memory) scripts the planner."""

    instances: ClassVar[list] = []

    def __init__(self, adapter, *, python=None, cuda_visible_devices="0", skills=()):
        self.adapter, self.skills, self.calls, self.closed = adapter, list(skills), [], False
        FakePlanner.instances.append(self)

    def plan(self, images, goal, running, memory):
        assert len(images) == 6 and all(image.ndim == 3 for image in images)
        self.calls.append({"goal": goal, "running": running, "memory": memory})
        skill, new_memory = self.reply(running, memory, len(self.calls))
        return {"skill": skill, "memory": new_memory, "reply": f"skill: {skill}", "infer_s": 0.01}

    def submit(self, images, goal, running, memory):
        future = Future()
        future.set_result(self.plan(images, goal, running, memory))
        return future

    def close(self):
        self.closed = True


def planner_model(monkeypatch, reply, **settings):
    FakePlanner.instances = []
    monkeypatch.setattr(lh, "PlannerClient", FakePlanner)
    monkeypatch.setattr(FakePlanner, "reply", staticmethod(reply), raising=False)
    policy = fake_policy()
    args = {"policy_name": "fake", "task_name": "click_bell", "fake_prompts": PROMPTS, "fake_steps": 10}
    args.update(long_horizon_goal="先按铃，再倒水", long_horizon_planner="base", long_horizon_min_steps=0)
    args.setdefault("long_horizon_planner_every_s", 0)  # these tests ask after every chunk
    args.update({f"long_horizon_{k}": v for k, v in settings.items()})
    return lh.LongHorizonPolicy(policy).get_model(args), policy, FakePlanner.instances[0]


def test_planner_picks_the_first_skill_switches_on_confirmed_replies_and_finishes(monkeypatch):
    def reply(running, memory, call):
        if running is None:
            return ("click_bell", memory) if memory == pp.NO_MEMORY else ("water_pouring", memory)
        if running == "click_bell":
            return ("water_pouring", "已完成：按铃。") if call >= 3 else ("click_bell", memory)
        return (pp.DONE, "已完成：按铃；倒水。") if call >= 8 else ("water_pouring", memory)

    model, policy, planner = planner_model(monkeypatch, reply)
    assert model.settings.tasks == ["click_bell", "items_handover", "water_pouring"]  # the skills with a prompt
    env = FakeEnv()
    obs = run(model, env, env.reset(), 1)
    assert planner.calls[0] == {"goal": "先按铃，再倒水", "running": None, "memory": pp.NO_MEMORY}
    assert policy.built[0].seen[0] == ("Click the bell", "Click the bell")
    obs = run(model, env, obs, 3)  # replies 2 (bell), 3 (water), 4 (water): confirmed at the third chunk
    assert model.records[0]["reason"] == "planner" and model.records[0]["next"] == "water_pouring"
    assert model.memory == "已完成：按铃。" and model.finished == ["click_bell"]
    obs = run(model, env, obs, 2)  # home, then water starts with the proposed skill, no new question
    assert model.task == "water_pouring" and planner.calls[-1]["running"] == "click_bell"
    obs = run(model, env, obs, 6)  # replies are read a chunk late: calls 8 and 9 (done) confirm at the 6th chunk
    assert planner.calls[-1]["running"] == "water_pouring" and planner.calls[-1]["memory"] == "已完成：按铃。"
    assert model.records[1]["reason"] == "planner" and model.after_home == "finish"
    run(model, env, obs, 3)
    summary = model.episode_summary()
    assert model.phase == "finished" and summary["memory"] == "已完成：按铃；倒水。" and summary["ended"] == 2
    model.close()
    assert planner.closed


def test_planner_asked_again_after_operator_or_timeout_with_the_memory_then(monkeypatch):
    def reply(running, memory, call):
        if running is None:
            return ("click_bell", memory) if memory == pp.NO_MEMORY and call == 1 else ("items_handover", memory)
        return (running, memory)  # never switches by itself

    model, _policy, planner = planner_model(monkeypatch, reply, max_steps=20)
    env = FakeEnv()
    obs = run(model, env, env.reset(), 1)
    model.request_next()  # the operator: the bell is done
    obs = run(model, env, obs, 3)
    asked = [c for c in planner.calls if c["running"] is None]
    assert asked[-1]["memory"] == "已完成：按铃。" and model.task == "items_handover"
    run(model, env, obs, 2)  # 20 steps: timeout; home; asked again with the memory unchanged; handover again
    assert [r.get("reason") for r in model.records] == ["operator", "timeout", None]
    assert [c for c in planner.calls if c["running"] is None][-1]["memory"] == "已完成：按铃。"
    assert model.task == "items_handover"  # not finished, so the planner picks it again


def test_planner_done_at_once_holds_and_settings_pair_up(monkeypatch):
    model, _policy, _planner = planner_model(monkeypatch, lambda running, memory, call: (pp.DONE, memory))
    env = FakeEnv()
    run(model, env, env.reset(), 1)
    assert model.phase == "finished" and model.records == []
    with pytest.raises(ValueError, match="go together"):
        lh.Settings.from_usr_args({"task_name": "click_bell", "long_horizon_goal": "按铃"})
    with pytest.raises(ValueError, match="no long_horizon_detector"):
        lh.Settings.from_usr_args(
            {
                "task_name": "x",
                "long_horizon_goal": "按铃",
                "long_horizon_planner": "base",
                "long_horizon_detector": "d",
            }
        )


def test_frame_memory_gives_the_top_camera_seconds_before():
    memory = planner.FrameMemory(control_hz=10)
    for step in range(0, 61, 5):
        memory.note(step, np.full((2, 2, 3), step, np.uint8))
    past = memory.past(60, np.zeros((2, 2, 3), np.uint8))
    assert [int(image[0, 0, 0]) for image in past] == [30, 40, 50]  # one per second kept: 3, 2, 1 s before step 60
    fresh = planner.FrameMemory(control_hz=10)
    fresh.note(0, np.full((2, 2, 3), 7, np.uint8))
    assert [int(image[0, 0, 0]) for image in fresh.past(4, np.zeros((2, 2, 3), np.uint8))] == [7, 7, 7]


def test_planner_prompt_round_trip_and_training_samples():
    rng = random.Random(0)
    chain = ["click_bell", "water_pouring"]
    goal = pp.goal_text(chain, rng)
    assert goal and pp.memory_text(chain) == "已完成：按铃；倒水。" and pp.memory_text([]) == pp.NO_MEMORY
    assert pp.parse_reply(pp.target_text("water_pouring", "已完成：按铃。")) == ("water_pouring", "已完成：按铃。")
    assert pp.parse_reply("skill: <name or done>\nmemory: x") == (None, "x")
    assert "Now running: nothing yet" in pp.prompt_text(goal, None, pp.NO_MEMORY)
    episode = {"dir": "click_bell/ours/r0/episode_000001", "task": "click_bell", "labels": [0, 0, -1, 1, 1]}
    finished, going = bpd.sample_of(episode, 3, random.Random(1)), bpd.sample_of(episode, 1, random.Random(1))
    assert finished["finished"] and finished["target_skill"] == pp.next_skill(finished["chain"], finished["index"])
    assert finished["target_memory"] == pp.memory_text(finished["chain"][: finished["index"] + 1])
    assert going["target_skill"] == "click_bell" and going["target_memory"] == going["memory"]
    assert finished["images"][3].endswith("0000_cam_high.jpg")  # 3 s before frame 3: the first frame
    picked = bpd.pick_frames(np.array([0] * 10 + [-1] * 3 + [1] * 6), random.Random(0), n_negative=3, n_finished=2)
    assert 13 in picked and 9 in picked and sum(k >= 13 for k in picked) == 2 and sum(k < 10 for k in picked) == 4
    failed = bpd.pick_frames(np.zeros(8, np.int8), random.Random(0), n_negative=3, n_finished=2)  # no finished frames
    assert 7 in failed and 3 <= len(failed) <= 4


def test_upstream_planner_reply_handling():
    bell, water = "click_bell", "water_pouring"
    water_text = pp.SKILLS[water][0]
    done = {"scene": "s", "previous_command": "completed", "instruction": water_text + "."}
    assert up.skill_of(done, bell) == water
    assert up.skill_of({**done, "instruction": "DONE"}, bell) == "done"
    assert up.skill_of({**done, "previous_command": "in progress"}, bell) == bell  # the command stays
    assert up.skill_of({**done, "previous_command": "in progress", "instruction": "done"}, bell) == "done"
    assert up.skill_of({**done, "instruction": "wave at the camera"}, bell) is None  # outside the vocabulary
    assert up.skill_of(None, bell) is None
    assert up.json_reply('```json\n{"instruction": "x"}\n```') == {"instruction": "x"}
    assert up.json_reply('here it is: {"instruction": {"a": 1}} thanks') == {"instruction": {"a": 1}}
    assert up.json_reply("no json") is None
    assert up.ask_frames(25) == [0, 10, 20, 30]  # the last one past the end + HOLD_S: standing still
    text = up.request_text("press the bell", "goal", [water_text])
    assert "Current instruction: press the bell" in text
    assert water_text in text
    assert text.endswith("Never reply with the instruction alone.")


def test_hard_negative_frames():
    episode = {"labels": [0, 0, 0, 0, 0, 0, -1, 0, 1, 1]}
    assert ahn.hard_frames(episode, 6, random.Random(0), 9, 3) == [3, 4, 5]  # the last 3 s before frame 6
    assert ahn.hard_frames(episode, 9, random.Random(0), 9, 4) == [5, 7]  # only frames that are not finished
    assert len(ahn.hard_frames(episode, 6, random.Random(0), 2, 6)) == 2
    sample = {"images": ["top", "left", "right", "top_3", "top_2", "top_1"], "finished": True}
    seen = set()
    for seed in range(20):
        still = ahn.standing_still(sample, random.Random(seed))
        assert still["images"][:3] == sample["images"][:3]
        assert still["finished"]
        seen.add((still["still"], tuple(still["images"][3:])))
    assert seen == {
        (1, ("top_3", "top_2", "top")),
        (2, ("top_3", "top", "top")),
        (3, ("top", "top", "top")),
    }


def test_planner_sweep_rule():
    bell, water = "click_bell", "water_pouring"
    skills = [bell, water, bell, water, water, water, water]
    assert ep.fire_frame(skills, bell, 1, 2) == 4  # the lone reply at 1 does not count
    assert ep.fire_frame(skills, bell, 1, 3) == 5
    assert ep.fire_frame(skills, bell, 2, 2) == 6  # asked at 0, 2, 4, 6
    assert ep.fire_frame(skills, bell, 1, 6) == 8  # past the end the last answer holds: 3, 4, 5, 6, then 7, 8
    assert ep.fire_frame([bell, water, bell], bell, 1, 2) is None
    episode = {"success": True, "labels": [0, 0, 0, -1, 1, 1, 1]}
    record = {"skills": skills, "task": bell}
    assert ep.sweep_outcome(record, episode, None, 1, 2) == ("caught", 4)
    assert ep.sweep_outcome(record, episode, 5, 1, 2) == ("pretakeover", 4)
    assert ep.sweep_outcome({"skills": [water] * 7, "task": bell}, episode, None, 1, 2) == ("early", 1)
    assert ep.sweep_outcome({"skills": [bell] * 7, "task": bell}, episode, None, 1, 2) == ("missed", None)
    assert ep.sweep_outcome(record, {**episode, "success": False}, None, 1, 2) == ("false", 4)
    assert ep.held_images({"dir": "d", "frame_index": [0, 10]}, 3)[0] == "frames/d/0001_cam_high.jpg"
    # asked every 2 s, scored at 2 s and 4 s
    asked = {"skills": [bell, water, water, water], "task": bell, "ask_every": 2}
    assert ep.sweep_outcome(asked, episode, None, 2, 2) == ("caught", 4)
    assert ep.sweep_outcome(asked, episode, None, 4, 2) == ("caught", 8)  # asked at 0, 4, then held


def test_planner_hands_in_the_prompt():
    rows = [[0.1 * j, -0.05, 0.25, 0.8, 0.3, 0.05, 0.17, 0.0] for j in range(5)]
    text = pp.hands_text(rows, 4)
    assert text.startswith("3 s ago: L 10 -5 25 open 80, R 30 5 17 open 0; 2 s ago: L 20")
    assert text.endswith("now: L 40 -5 25 open 80, R 30 5 17 open 0")
    assert pp.hands_text(rows, 0).startswith("3 s ago: L 0 -5 25")  # before the episode: its first frame
    assert "Hands" not in pp.prompt_text("g", None, pp.NO_MEMORY)
    assert text in pp.prompt_text("g", None, pp.NO_MEMORY, state=text)
    assert pp.version(state=True) != pp.version()


def test_success_moment_rules():
    # a release is the last rise with the arm out over the table: the opening at home afterwards does not count
    opening = np.array([0.0] * 5 + [0.8] * 15 + [0.3] * 20 + [0.31, 0.5, 0.9] + [0.9] * 10 + [0.0] * 10 + [0.9] * 3)
    reach = np.array([0.03] * 5 + [0.3] * 53 + [0.03] * 10)[: len(opening)]
    assert sm.last_release(opening, reach) == 39
    assert sm.last_release(opening, np.zeros(len(opening))) is None
    # a press: the first frame at the lowest height
    assert sm.first_at_lowest(np.array([0.3, 0.25, 0.2, 0.17, 0.169, 0.17, 0.25])) == 3
    # the pipette: lowest point out at the pipette that the arm still rises from, not the sinking at the end
    left = np.zeros((12, 3))
    left[:, 0] = [0.03, 0.03, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    left[:, 2] = [0.25, 0.25, 0.35, 0.33, 0.30, 0.27, 0.272, 0.285, 0.33, 0.30, 0.26, 0.24]
    assert sm.pipette_released(left) == 7  # pressed at 5, up 1 cm at 7
    # stopped during the press: its lowest point
    assert (
        sm.pipette_released(left[:7] * [1, 1, 0] + [0, 0, 1] * np.array([[0, 0, z] for z in np.linspace(0.3, 0.27, 7)]))
        == 6
    )
    # docking: the left y falls and the right y rises until the tubes are in, then both hold
    flanges = np.zeros((10, 2, 3))
    flanges[:, 0, 1] = [0, -0.05, -0.1, -0.14, -0.15, -0.15, -0.15, -0.15, -0.1, 0]
    flanges[:, 1, 1] = [0, 0.05, 0.1, 0.14, 0.15, 0.15, 0.151, 0.15, 0.1, 0]
    assert sm.assembly_docked(flanges) == 4


# ----- memory planner: any goal of the ten skills, the running skill's progress in the memory -------------------------


def test_any_chain_and_goal_cover_long_goals_with_repeats():
    lengths, repeated = set(), 0
    for seed in range(400):
        chain, index = pp.any_chain("click_bell", random.Random(seed))
        assert chain[index] == "click_bell" and 1 <= len(chain) <= pp.MAX_CHAIN
        assert all(chain[i] != chain[i + 1] for i in range(len(chain) - 1))  # a switch needs another skill
        lengths.add(len(chain))
        repeated += len(set(chain)) < len(chain)
        goal = pp.any_goal_text(chain, random.Random(seed))
        assert goal and ("最后" in goal or "依次" in goal or "finally" in goal or "then" in goal or len(chain) == 1)
    assert lengths == set(range(1, 11)) and repeated > 50


def test_memory_with_progress_round_trip_and_stages():
    assert pp.memory_text(["click_bell"]) == "已完成：按铃。"  # the old planners' memory is unchanged
    text = pp.memory_text(["click_bell", "water_pouring", "click_bell"], 2)
    assert text == "已完成：按铃；倒水；按铃。当前进度：2/4。"
    assert pp.parse_memory(text) == (3, 2) and pp.parse_memory(pp.memory_text([], 0)) == (0, 0)
    assert pp.parse_memory("已完成：按铃。") is None and pp.parse_memory(None) is None
    assert [pp.progress_of(k, 8) for k in (0, 1, 2, 4, 6, 7, 8, 9)] == [0, 0, 1, 2, 3, 3, 4, 4]
    assert pp.progress_of(5, None) == 0
    assert pp.version(progress=True) != pp.version()
    assert "当前进度" not in pp.prompt_text("g", None, pp.NO_MEMORY) and "stage 0 to 3" in pp.prompt_text(
        "g", None, pp.NO_MEMORY, progress=True
    )


def test_memory_tracker_takes_a_stage_on_two_of_three_replies_and_never_goes_back():
    mem = pp.MemoryTracker(progress=True, finished=["water_pouring"])
    assert mem.text() == "已完成：倒水。当前进度：0/4。"
    mem.observe(pp.memory_text(["water_pouring"], 2))
    assert mem.progress == 0  # one reply is not enough
    mem.observe(pp.memory_text(["water_pouring"], 2))
    assert mem.progress == 2
    mem.observe(pp.memory_text(["water_pouring"], 1))
    assert mem.progress == 2  # never back
    mem.observe(pp.memory_text(["water_pouring", "click_bell"], 0))  # this reply says the skill is finished
    assert mem.progress == 2  # votes 2, 1, 4: the middle one
    mem.observe(pp.memory_text(["water_pouring", "click_bell"], 0))
    assert mem.progress == pp.STAGES and mem.text().endswith("当前进度：4/4。")
    mem.observe("garbage")
    assert mem.progress == pp.STAGES
    mem.finish("click_bell")
    assert mem.text() == "已完成：倒水；按铃。当前进度：0/4。"
    old = pp.MemoryTracker(progress=False, finished=["click_bell"])
    old.observe(pp.memory_text(["click_bell", "water_pouring"], 0))
    assert old.text() == "已完成：按铃。"


def test_mem_samples_targets():
    bell = {"dir": "b", "task": "click_bell", "success": True, "labels": [0, 0, 0, 0, 0, 0, -1, 1, 1, 1]}
    failed = {"dir": "f", "task": "click_bell", "success": False, "labels": [0] * 30}
    medians = bms.median_finish([bell])
    assert medians == {"click_bell": 7} and bms.stage_at(failed, 29, medians) == pp.STAGES - 1
    for seed in range(30):
        rng = random.Random(seed)
        done = bms.mem_sample({"episode": "b", "frame": 8, "images": ["x"] * 6, "finished": True}, bell, medians, rng)
        chain, index = done["chain"], done["index"]
        assert done["target_skill"] == pp.next_skill(chain, index)
        assert done["target_memory"] == pp.memory_text(chain[: index + 1], 0)
        if done["running"] is not None:
            assert pp.parse_memory(done["memory"]) == (index, pp.parse_memory(done["memory"])[1])
        going = bms.mem_sample({"episode": "b", "frame": 4, "images": ["x"] * 6, "finished": False}, bell, medians, rng)
        assert going["target_skill"] == "click_bell"
        assert going["target_memory"] == pp.memory_text(going["chain"][: going["index"]], 2)  # 4 of 7 frames
        lagged = pp.parse_memory(going["memory"])[1]
        assert lagged <= 2 or going["running"] is None or lagged == 3  # 1-4 s before, or one stage off


def write_frames(root: Path, episode: dict) -> None:
    from PIL import Image

    folder = root / "frames" / episode["dir"]
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(len(episode["labels"])):
        for cam in pp.CAMERAS:
            Image.new("RGB", (4, 3), (40 * k,) * 3).save(folder / f"{k:04d}_{cam}.jpg")  # grey: survives the JPEG


class ScriptedPlanner:
    """Planner's plan(); reads the frame from the top image's grey level and says a skill is finished from frame 4."""

    uses_state, uses_progress = False, True

    def __init__(self):
        self.calls = []

    def plan(self, images, goal, running, memory, state=None):
        k = round(images[0].getpixel((0, 0))[0] / 40)
        self.calls.append((running, memory, k))
        chain = ["click_bell", "water_pouring", "click_bell"]
        n_done = pp.parse_memory(memory)[0]
        if running is None:
            return chain[n_done], memory, "", 0.0
        if k >= 4:
            return pp.next_skill(chain, n_done), pp.memory_text(chain[: n_done + 1], 0), "", 0.0
        return running, pp.memory_text(chain[:n_done], min(3, k)), "", 0.0


def test_run_chain_closed_loop_scores_every_switch(tmp_path):
    episodes = {}
    for name, task in (("bell/e0", "click_bell"), ("water/e0", "water_pouring")):
        episodes[name] = {"dir": name, "task": task, "success": True, "labels": [0, 0, 0, -1, 1, 1],
                          "frame_index": list(range(0, 60, 10)), "split": "val"}  # fmt: skip
        write_frames(tmp_path, episodes[name])
    spec = {"chain": ["click_bell", "water_pouring", "click_bell"], "episodes": ["bell/e0", "water/e0", "bell/e0"],
            "goal": "g"}  # fmt: skip
    planner_ = ScriptedPlanner()
    record = ep.run_chain(planner_, tmp_path, spec, episodes, every=1, confirm=2)
    assert record["start_ok"] and record["completed"]
    assert [s["outcome"] for s in record["steps"]] == ["caught"] * 3 and record["steps"][0]["switch"]["frame"] == 5
    assert record["steps"][0]["trace"][3] == [3, "click_bell", 2, None]  # stage 2 needs two replies at 2 or above
    assert planner_.calls[-1][1] == pp.memory_text(["click_bell", "water_pouring"], 3)  # votes 2, 3, 4 at frame 5
    early = ep.run_chain(planner_, tmp_path, spec, episodes, every=1, confirm=1)
    assert [s["outcome"] for s in early["steps"]] == ["caught"] * 3  # frame 4 is finished
    specs = ep.chain_specs(list(episodes.values()), 0, per_length=2, lengths=(2, 10))
    assert len(specs) == 4 and len(specs[-1]["chain"]) == 10
    text = ep.chain_report([record], 1, 2, "x")
    assert "| 3 | 1/1 (100%) | 1/1 |" in text


def test_runtime_memory_planner_keeps_the_progress(monkeypatch):
    def reply(running, memory, call):
        if running is None:
            return "click_bell", memory
        if running == "click_bell":
            stage = min(3, call - 1)
            return ("water_pouring", pp.memory_text(["click_bell"], 0)) if call >= 6 else (
                "click_bell", pp.memory_text([], stage))  # fmt: skip
        return "water_pouring", memory

    monkeypatch.setattr(FakePlanner, "info", {"progress": True}, raising=False)
    try:
        model, _policy, planner_ = planner_model(monkeypatch, reply)
        assert model.uses_progress
        env = FakeEnv()
        obs = run(model, env, env.reset(), 1)
        assert planner_.calls[0]["memory"] == pp.memory_text([], 0)
        run(model, env, obs, 8)  # replies 6 and 7 name water (read a chunk late)
        memories = [c["memory"] for c in planner_.calls if c["running"] == "click_bell"]
        assert pp.memory_text([], 2) in memories  # a stage taken from the replies on the way
        assert model.finished == ["click_bell"] and model.memory == pp.memory_text(["click_bell"], 0)
    finally:
        monkeypatch.delattr(FakePlanner, "info", raising=False)


def test_lerobot_format_training_samples():
    episode = {"dir": "b", "task": "click_bell", "labels": [0] * 12 + [-1] + [1] * 10}
    rows = [[0.1, 0.0, 0.2, 0.5, 0.3, 0.0, 0.2, 0.0]] * 23  # never moves
    for seed in range(30):
        s = up.lerobot_sample({"episode": "b", "frame": 20, "finished": True}, episode, rows, random.Random(seed))
        frames = [h["frame"] for h in s["history"]]
        assert s["chain"][0] == "click_bell"
        assert frames == [20 - s["spacing"] * j for j in range(len(frames), 0, -1)]
        assert 1 <= len(frames) <= up.HISTORY
        assert all(f >= 0 for f in frames)
        reply = s["reply"]
        assert reply["previous_command"] == "completed"
        assert reply["scene"].startswith("The arms are still")
        assert up.skill_of(reply, "click_bell") == pp.next_skill(s["chain"], 0)
        early = up.lerobot_sample({"episode": "b", "frame": 1, "finished": False}, episode, rows, random.Random(seed))
        if not early["history"]:  # nothing asked before: no previous command, the running skill first
            assert early["reply"]["previous_command"] == "none"
            assert up.skill_of(early["reply"], "click_bell") == "click_bell"
        else:
            assert early["reply"]["previous_command"] == "in progress"
            assert early["history"][0]["assessment"].endswith("previous command: none")
    assert up.ask_frames(5, 2) == [0, 2, 4, 6, 8]
    assert not up.arms_still(rows, 0)
    assert up.version() == up.version()


def test_planner_asked_at_most_every_planner_every_s(monkeypatch):
    model, _policy, planner_ = planner_model(
        monkeypatch, lambda running, memory, call: ("click_bell", memory), planner_every_s=3
    )
    env = FakeEnv()
    obs = run(model, env, env.reset(), 1)
    asked = len(planner_.calls)
    run(model, env, obs, 6)  # 10 steps a chunk at 10 Hz: 1 s a chunk, an ask every third chunk
    assert len(planner_.calls) - asked == 2


class ScoredPlanner:
    """plan_scored(): the running skill's p falls by 0.2 a frame from 1; the greedy reply switches below 0.5."""

    uses_state, uses_progress = False, False

    def plan_scored(self, images, goal, running, memory, state=None):
        k = round(images[0].getpixel((0, 0))[0] / 40)
        if running is None:
            return {"skill": "click_bell", "memory": memory, "reply": "", "seconds": 0.0, "probs": None}
        p = max(0.0, 1.0 - 0.2 * k)
        skill = "water_pouring" if p < 0.5 else running
        probs = {name: 0.0 for name in [*pp.SKILLS, pp.DONE]}
        probs[running], probs["water_pouring" if running != "water_pouring" else pp.DONE] = p, 1 - p
        return {"skill": skill, "memory": memory, "reply": "", "seconds": 0.0, "probs": probs}


class StillPlanner:
    """plan(): another skill only when every image is the same frame (the robot has stood still for HOLD_S)."""

    uses_state, uses_progress = False, False

    def plan(self, images, goal, running, memory, state=None):
        frozen = len({image.getpixel((0, 0)) for image in images}) == 1
        return ("water_pouring" if frozen and running else running or "click_bell"), memory, "", 0.0


def test_threshold_switching_and_asks_while_standing_still(tmp_path):
    bell = {"dir": "bell/e0", "task": "click_bell", "success": True, "labels": [0, -1, 1, 1, 1, 1],
            "frame_index": list(range(0, 60, 10)), "split": "val"}  # fmt: skip
    write_frames(tmp_path, bell)
    spec = {"chain": ["click_bell", "water_pouring"], "episodes": ["bell/e0", "bell/e0"], "goal": "g"}
    greedy = ep.run_chain(ScoredPlanner(), tmp_path, spec, {"bell/e0": bell}, every=1, confirm=2)
    assert greedy["steps"][0]["switch"]["frame"] == 4  # p 0.4 at 3, 0.2 at 4
    early = ep.run_chain(ScoredPlanner(), tmp_path, spec, {"bell/e0": bell}, every=1, confirm=2, threshold=0.9)
    assert early["steps"][0]["switch"]["frame"] == 2  # p 0.8 at 1, 0.6 at 2: below 0.9 twice
    assert early["steps"][0]["outcome"] == "caught"
    # switches only once all six images are the last frame (asked every 3 s: at 9 only, before): asked again at 12
    rows = {"bell/e0": [[0.1] * 8] * 6}  # the hands too, past the end of the recording
    late = ep.run_chain(StillPlanner(), tmp_path, spec, {"bell/e0": bell}, every=3, confirm=2, states=rows)
    assert late["steps"][0]["switch"]["frame"] == 12
    assert late["steps"][0]["outcome"] == "caught"
    assert ep.decide("x", 0.3, "y", "x", None) == "x"
    assert ep.decide("x", 0.3, "y", "x", 0.5) == "y"
    assert ep.decide("x", 0.3, None, "x", 0.2) == "x"


def test_long_history_images_and_version():
    episode = {"dir": "d", "task": "click_bell", "labels": [0] * 30}
    images = bms.history_images(episode, 25, pp.LONG_PAST_S, None)
    assert [Path(p).name[:4] for p in images[3:]] == ["0005", "0015", "0022", "0023", "0024"]
    still = bms.history_images(episode, 25, pp.LONG_PAST_S, 10)  # still for 10 s: all but the 20 s image are now
    assert still[3].endswith("0005_cam_high.jpg") and all(p == still[0] for p in still[4:])
    assert bms.history_images(episode, 4, pp.LONG_PAST_S, None)[3].endswith("0000_cam_high.jpg")  # clamped
    assert pp.version(progress=True, past=pp.LONG_PAST_S) != pp.version(progress=True)
    assert pp.version(progress=True, past=pp.PAST_S) == pp.version(progress=True)  # old adapters still load
    assert "20, 10, 3, 2, 1 seconds ago" in pp.prompt_text("g", None, pp.NO_MEMORY, progress=True, past=pp.LONG_PAST_S)
    memory = planner.FrameMemory(control_hz=10, past=pp.LONG_PAST_S)
    for step in range(0, 301, 10):
        memory.note(step, np.full((2, 2, 3), step // 10, np.uint8))
    assert [int(i[0, 0, 0]) for i in memory.past(300, np.zeros((2, 2, 3), np.uint8))] == [10, 20, 27, 28, 29]


def test_training_batches_pad_rows_and_concatenate_images():
    from policy.longhorizon import train_planner as tp

    a = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
         "labels": torch.tensor([[-100, 2, 3]]), "pixel_values": torch.zeros(4, 2),
         "image_grid_thw": torch.tensor([[1, 2, 2]])}  # fmt: skip
    b = {"input_ids": torch.tensor([[5]]), "attention_mask": torch.ones(1, 1, dtype=torch.long),
         "labels": torch.tensor([[5]]), "pixel_values": torch.ones(8, 2), "image_grid_thw": torch.tensor([[1, 2, 4]])}  # fmt: skip
    batch = tp.collate([a, b], pad_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [5, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [5, -100, -100]]
    assert batch["pixel_values"].shape == (12, 2) and batch["pixel_values"][4:].sum() == 16
    assert batch["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 4]]
