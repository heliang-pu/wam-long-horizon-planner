# ----------------------------------------------------------------------------
# Done detector of policy/longhorizon: has the running subtask finished?
#
# Per frame, one feature vector per "view", then a small causal GRU over the subtask's frames so far (from its first
# frame on) gives one done probability per task. Two feature sources (config.json "source"):
#   phoenix  the world action model's own features (wam_features.py): Phoenix-MWA's hidden states of the current frame
#            at a few layers of its video expert, pooled per camera. They depend on the prompt and qpos as well, and
#            come for free from the inference the robot runs anyway (one per chunk); the WAM is not trained.
#   image    a frozen image encoder (DINOv3-S/16, the CLS token and the mean patch token, 768-D), a TorchScript file
#            (export_encoder.py) so any venv with torch runs it; works with any WAM.
# Each frame also gets the seconds since the frame before (dt), so chunks of 10 or 20 steps both fit.
#
# Inputs are the official ones only: the three cameras and qpos (14-D, sim units).
#
# Detector directory (train_done_detector.py writes it):
#   config.json                 source, tasks, head sizes, per-task thresholds chosen on validation
#   head.pt                     DoneHead state dict (normalization buffers included)
#   encoder.pt + encoder.json   image source only: the TorchScript encoder and its input (image size, mean, std)
#   features.json               phoenix source only: the checkpoint and layers the features were taken from
# ----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
TASKS = (
    "click_bell",
    "drawer_open_place",
    "handle_basket",
    "item_assembly",
    "items_handover",
    "manipulate_pipette",
    "mixer_operating",
    "sample_loading",
    "table_rearrangement",
    "water_pouring",
)
STATE_DIM = 14
SOURCES = ("image", "phoenix")
DT_SCALE_S = 2.0  # dt enters the head divided by this


def resolve_device(device: str | None) -> torch.device:
    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def preprocess(images: torch.Tensor, size: int, mean, std) -> torch.Tensor:
    """(N, H, W, 3) uint8 RGB -> (N, 3, size, size) normalized float: the whole view squashed, as the WAMs see it.

    Training (build_done_features.py) and deployment both go through here, so the encoder sees the same pixels."""
    x = images.permute(0, 3, 1, 2).float().div_(255.0)
    x = nn.functional.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class FrameEncoder:
    """The TorchScript encoder of a detector (or feature) directory: camera images -> per-camera features."""

    def __init__(self, root: str | Path, device: str | None = "auto"):
        root = Path(root)
        self.device = resolve_device(device)
        self.meta = json.loads((root / "encoder.json").read_text())
        self.size = int(self.meta["image_size"])
        self.feature_dim = int(self.meta["feature_dim"])
        self.module = torch.jit.load(str(root / "encoder.pt"), map_location=self.device).eval()

    @torch.no_grad()
    def __call__(self, images: np.ndarray | torch.Tensor, batch: int = 96) -> np.ndarray:
        """(N, H, W, 3) uint8 -> (N, feature_dim) float32."""
        images = torch.as_tensor(np.asarray(images))
        out = []
        for start in range(0, images.shape[0], batch):
            x = preprocess(
                images[start : start + batch].to(self.device),
                self.size,
                self.meta["mean"],
                self.meta["std"],
            )
            out.append(self.module(x).float().cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, self.feature_dim), np.float32)

    def frames(self, frames: list[dict[str, np.ndarray]]) -> np.ndarray:
        """Frames ({camera: (H, W, 3) uint8}) -> (T, len(CAMERAS), feature_dim)."""
        if not frames:
            return np.zeros((0, len(CAMERAS), self.feature_dim), np.float32)
        stacked = np.stack([frame[cam] for frame in frames for cam in CAMERAS])
        return self(stacked).reshape(len(frames), len(CAMERAS), self.feature_dim)


class DoneHead(nn.Module):
    """Per frame: normalized features (one row per view) + qpos + dt -> projection; a GRU over the frames; one logit
    per task."""

    def __init__(
        self,
        feature_dim: int = 768,
        n_cameras: int = len(CAMERAS),
        state_dim: int = STATE_DIM,
        hidden: int = 256,
        n_tasks: int = len(TASKS),
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = n_cameras * feature_dim + state_dim + 1
        self.register_buffer("feature_mean", torch.zeros(n_cameras, feature_dim))
        self.register_buffer("feature_std", torch.ones(n_cameras, feature_dim))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_tasks)

    def set_normalization(self, feature_mean, feature_std, state_mean, state_std) -> None:
        for name, value in (
            ("feature_mean", feature_mean),
            ("feature_std", feature_std),
            ("state_mean", state_mean),
            ("state_std", state_std),
        ):
            getattr(self, name).copy_(torch.as_tensor(value, dtype=torch.float32))

    def embed(self, features: torch.Tensor, state: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """(B, T, C, F) features + (B, T, 14) state + (B, T) seconds since the frame before -> (B, T, hidden)."""
        features = (features - self.feature_mean) / self.feature_std
        state = (state - self.state_mean) / self.state_std
        return self.proj(torch.cat([features.flatten(2), state, dt.unsqueeze(-1) / DT_SCALE_S], dim=-1))

    def forward(self, features, state, dt, hidden=None):
        """Logits (B, T, n_tasks) and the GRU state."""
        y, hidden = self.gru(self.embed(features, state, dt), hidden)
        return self.out(y), hidden


def extract_image(value) -> np.ndarray:
    """One camera of env 0 (EmbodiChain / RealCobotMagicEnv layout) as contiguous HWC uint8 RGB."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)[0][..., :3]
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0 if float(np.nanmax(image)) <= 1.0 else image, 0, 255)
    return np.ascontiguousarray(image.astype(np.uint8))


def frame_of(obs) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """An environment observation -> ({camera: HWC uint8}, qpos (14,))."""
    return {cam: extract_image(obs["sensor"][cam]["color"]) for cam in CAMERAS}, qpos_of(obs)


def qpos_of(obs) -> np.ndarray:
    qpos = obs["robot"]["qpos"]
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    return np.asarray(qpos, dtype=np.float32)[0].reshape(-1)[:STATE_DIM]


class DoneDetector:
    """A trained detector directory, loaded for deployment."""

    def __init__(self, root: str | Path, device: str | None = "auto"):
        self.root = Path(root).expanduser()
        self.config = json.loads((self.root / "config.json").read_text())
        self.source = str(self.config.get("source", "image"))
        if self.source not in SOURCES:
            raise ValueError(f"{self.root}: unknown feature source {self.source!r}")
        self.tasks = list(self.config["tasks"])
        self.fps = float(self.config["fps"])  # image source: the frame rate the head was trained on
        self.device = resolve_device(device)
        self.encoder = FrameEncoder(self.root, device) if self.source == "image" else None
        self.features = json.loads((self.root / "features.json").read_text()) if self.source == "phoenix" else {}
        self.head = DoneHead(**self.config["head"])
        self.head.load_state_dict(torch.load(self.root / "head.pt", map_location="cpu"))
        self.head.to(self.device).eval()

    def threshold(self, task: str) -> float:
        return float(self.config.get("thresholds", {}).get(task, self.config.get("default_threshold", 0.5)))

    def tracker(self, task: str) -> DoneTracker:
        if task not in self.tasks:
            raise ValueError(f"the done detector in {self.root} knows {self.tasks}, not {task!r}")
        return DoneTracker(self, self.tasks.index(task))


class DoneTracker:
    """One subtask: frames (or their features) go in as they come, done probabilities come out."""

    def __init__(self, detector: DoneDetector, task_index: int):
        self.detector, self.task_index = detector, task_index
        self.hidden = None
        self.probs: list[float] = []

    def push(self, frames: list[dict[str, np.ndarray]], states: list[np.ndarray], dts: list[float]) -> list[float]:
        """Image source: camera frames ({camera: HWC uint8})."""
        if not frames:
            return []
        return self.push_features(self.detector.encoder.frames(frames), states, dts)

    @torch.no_grad()
    def push_features(self, features, states: list[np.ndarray], dts: list[float]) -> list[float]:
        """(T, C, F) features as the head was trained on, the qpos and the seconds since the frame before of each."""
        if len(features) == 0:
            return []
        device = self.detector.device
        features = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device).unsqueeze(0)
        state = torch.from_numpy(np.stack(states).astype(np.float32)).to(device).unsqueeze(0)
        dt = torch.as_tensor(np.asarray(dts, dtype=np.float32), device=device).unsqueeze(0)
        logits, self.hidden = self.detector.head(features, state, dt, self.hidden)
        probs = torch.sigmoid(logits[0, :, self.task_index]).cpu().tolist()
        self.probs += probs
        return probs
