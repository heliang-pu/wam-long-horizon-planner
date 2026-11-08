"""The world action model's own features for the done detector (source "phoenix"): Phoenix-MWA, frozen, not trained.

What: the current frame's hidden states in Phoenix's video expert at a few layers (default 10, 20, 28 of 30), averaged
over each camera's tokens -> (layers x 3 cameras, 3072). Upstream's joint96 attention keeps the current frame causal
(it reads only itself, the prompt and the proprio appended to it, at timestep 0), so these states are a deterministic
function of the three images, the prompt and qpos: the same at every denoising step and for every noise seed. They are
taken with upstream's own MoT capture_layers on the first MoT pass of an inference. The last layer is not used by
default: its current-frame outputs feed nothing (only the future frames and the actions are decoded).

Three uses, all under Phoenix's interpreter (venvs/phoenix), none of them changes policy/phoenix or the upstream code:

- worker: what policy/phoenix's adapter starts when the long-horizon wrapper sets phoenix_python to
  phoenix_tap_worker (the launcher next to this file): phoenix_worker.py runs as always, and after every "act" the
  features of that inference go to $LONGHORIZON_FEATURES_OUT (an npz, replaced atomically; a counter tells new ones).
- extract: the features of recorded frames for training (train_done_detector.py), from the same Runtime and the same
  inputs as on the robot; the inference stops right after the first MoT pass, so a frame costs the encoders and one
  pass instead of the whole denoising.

    PHOENIX_ROOT=<a Phoenix-MWA checkout> ../venvs/phoenix/bin/python policy/longhorizon/wam_features.py extract \
        --run-dir <Phoenix training output> --step 9429 --encoders <run>/encoders --wan-base <Wan2.2 VAE dir> \
        --out ~/workspace/models/longhorizon/phoenix_features_<run>_<step>

The features belong to that checkpoint: another Phoenix checkpoint needs its own extraction and detector.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

POLICY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = POLICY_ROOT.parents[1]
DEFAULT_LAYERS = (10, 20, 28)
N_VIEWS = 3  # Phoenix's latent_horizontal layout: main | left | right, side by side in every token row
FEATURES_OUT = "LONGHORIZON_FEATURES_OUT"
LAYERS_ENV = "LONGHORIZON_WAM_LAYERS"


class StopAfterCapture(Exception):  # noqa: N818 - not an error: extraction needs the first MoT pass only
    pass


def pool_views(hidden: torch.Tensor, tokens_per_frame: int, grid_h: int, grid_w: int, views: int = N_VIEWS):
    """(1, S, D) hidden states of the video tokens -> (views, D): the current frame's (the first tokens_per_frame,
    row-major over a grid_h x grid_w grid whose columns are the views side by side) mean per view, in float32."""
    if grid_h * grid_w != tokens_per_frame or grid_w % views:
        raise ValueError(f"token grid {grid_h}x{grid_w} does not hold {tokens_per_frame} tokens of {views} views")
    frame = hidden[0, :tokens_per_frame].float().reshape(grid_h, views, grid_w // views, -1)
    return frame.mean(dim=(0, 2))


class PhoenixFeatureTap:
    """Captures the current-frame features of the next inference of ``model`` (arm() first): it wraps the model's
    video_expert.pre_dit (for the token grid) and mot.forward (adding upstream's capture_layers) as instance
    attributes, nothing else changes. ``stop``: raise StopAfterCapture once captured (extraction)."""

    def __init__(self, model, layers=DEFAULT_LAYERS, *, stop: bool = False):
        self.layers, self.stop = tuple(int(layer) for layer in layers), stop
        self.armed, self.features, self.meta = False, None, None
        expert, mot = model.video_expert, model.mot
        pre_dit, mot_forward = expert.pre_dit, mot.forward

        def tapped_pre_dit(*args, **kwargs):
            out = pre_dit(*args, **kwargs)
            self.meta = out["meta"]
            return out

        def tapped_forward(*args, **kwargs):
            if not self.armed:
                return mot_forward(*args, **kwargs)
            self.armed = False
            kwargs["capture_layers"] = {"video": set(self.layers)}
            tokens, captured = mot_forward(*args, **kwargs)
            self.features = self.pool(captured["video"])
            if self.stop:
                raise StopAfterCapture
            return tokens

        expert.pre_dit, mot.forward = tapped_pre_dit, tapped_forward

    def arm(self) -> None:
        self.armed, self.features = True, None

    def pool(self, captured: dict) -> np.ndarray:
        """-> (layers x views, D) float16, layer-major."""
        _frames, grid_h, grid_w = (int(v) for v in self.meta["grid_size"])
        tokens_per_frame = int(self.meta["tokens_per_frame"])
        rows = [pool_views(captured[layer], tokens_per_frame, grid_h, grid_w) for layer in self.layers]
        return torch.cat(rows).cpu().numpy().astype(np.float16)


def write_features(path: str | Path, features: np.ndarray, count: int, *, padded: bool) -> None:
    """The latest inference's features for the deploy process, replaced atomically."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez(tmp, features=features, count=np.int64(count), pid=np.int64(os.getpid()), padded=np.bool_(padded),
             time=np.float64(time.time()))  # fmt: skip
    os.replace(tmp, path)


def read_features(path: str | Path) -> dict | None:
    """The deploy side: {"features", "id", "padded"} of the latest inference, or None before the first. "id" (worker
    process, its inference count) tells a new inference, also after the worker was started again."""
    try:
        with np.load(path) as data:
            ident = (int(data["pid"]), int(data["count"]))
            return {"features": data["features"], "id": ident, "padded": bool(data["padded"])}
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None


def load_worker_module(script: Path):
    """policy/phoenix/phoenix_worker.py (whichever one the adapter started) as a module, without running main."""
    spec = importlib.util.spec_from_file_location("phoenix_worker", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["phoenix_worker"] = module
    spec.loader.exec_module(module)
    return module


def layers_from_env() -> tuple[int, ...]:
    raw = os.environ.get(LAYERS_ENV, "")
    return tuple(int(v) for v in raw.split(",") if v.strip()) or DEFAULT_LAYERS


def run_worker(argv: list[str]) -> None:
    """phoenix_worker.py <its args> with every act's features written to $LONGHORIZON_FEATURES_OUT."""
    script, out, layers = Path(argv[0]), os.environ.get(FEATURES_OUT), layers_from_env()
    sys.argv = [str(script), *argv[1:]]
    worker = load_worker_module(script)
    init, act = worker.Runtime.__init__, worker.Runtime.act

    def init_with_tap(self, args):
        init(self, args)
        self.feature_tap = PhoenixFeatureTap(self.model, layers)
        self.feature_count = 0
        print(f"[phoenix] long-horizon features: layers {layers} -> {out}", file=sys.stderr, flush=True)

    def act_with_features(self, obs, instruction, *args, **kwargs):
        self.feature_tap.arm()
        result = act(self, obs, instruction, *args, **kwargs)
        if out and self.feature_tap.features is not None:
            self.feature_count += 1
            padded = bool(kwargs.get("pad_cameras"))  # a camera encoded as missing: not what the detector learnt
            write_features(out, self.feature_tap.features, self.feature_count, padded=padded)
        return result

    worker.Runtime.__init__, worker.Runtime.act = init_with_tap, act_with_features
    worker.main()


def phoenix_prompts() -> dict[str, str]:
    import yaml

    return yaml.safe_load((REPO_ROOT / "policy" / "phoenix" / "deploy_policy.yml").read_text())["phoenix_prompts"]


def extract(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Phoenix features of the recorded frames (see the module docstring)")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, default=0, help="weights slot of this step (0: the latest)")
    parser.add_argument("--encoders", required=True, help="csvae.pt + dino110k.pt")
    parser.add_argument("--wan-base", required=True, help="Wan2.2 VAE + tokenizer (DIFFSYNTH_MODEL_BASE_PATH)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--fps", type=float, default=1.0, help="frames per second encoded (a chunk is 1-2 s)")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0, help="encode at most this many episodes per dataset this run")
    parser.add_argument("--no-official", action="store_true")
    parser.add_argument("--no-ours", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT))
    from policy.longhorizon import build_done_features as bdf

    worker = load_worker_module(REPO_ROOT / "policy" / "phoenix" / "phoenix_worker.py")
    worker_args = worker.parse_args(
        ["--run-dir", args.run_dir, "--step", str(args.step), "--encoders", args.encoders, "--wan-base", args.wan_base,
         "--device", args.device, "--seed", "0", "--denoise-steps", "5"]
    )  # fmt: skip
    runtime = worker.Runtime(worker_args)
    tap = PhoenixFeatureTap(runtime.model, args.layers, stop=True)
    prompts = phoenix_prompts()
    meta = {
        "source": "phoenix",
        "checkpoint": str(runtime.checkpoint),
        "step": int(runtime.step),
        "run_dir": str(Path(args.run_dir).resolve()),
        "layers": list(tap.layers),
        "views": list(worker.CAMERAS),
        "fps": args.fps,
        "prompts": prompts,
    }

    def encode(row: dict, stride: int):
        views, state, keep = bdf.decode_frames(row, stride)
        prompt = prompts[row["task"]]
        features = []
        for k in range(len(keep)):
            obs = {cam: views[cam][k] for cam in worker.CAMERAS}
            obs["state"] = state[k]
            tap.arm()
            with contextlib.suppress(StopAfterCapture):
                runtime.act(obs, prompt)
            if tap.features is None:
                raise RuntimeError("the first MoT pass was not captured")
            features.append(tap.features)
        return np.stack(features), state, keep

    bdf.run_extraction(
        args.out, meta, encode, tasks=args.tasks, fps=args.fps, limit=args.limit, official=not args.no_official,
        ours=not args.no_ours,
    )  # fmt: skip


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("worker", "extract"):
        raise SystemExit(__doc__)
    (run_worker if sys.argv[1] == "worker" else extract)(sys.argv[2:])


if __name__ == "__main__":
    main()
