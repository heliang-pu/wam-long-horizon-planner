"""Export the done detector's frozen image encoder as TorchScript, so the detector runs in any venv with torch.

Needs timm and the model's weights in the Hugging Face cache; on the 4090 that is OpenWAM's venv (the socks proxy
in the environment breaks httpx even offline, so unset it):

    env -u all_proxy -u ALL_PROXY HF_HUB_OFFLINE=1 ../venvs/openwam/bin/python -I \
        policy/longhorizon/export_encoder.py --out ~/workspace/models/longhorizon/encoder_dinov3s224

Writes encoder.pt (input: (N, 3, S, S) normalized as encoder.json says; output: (N, 2 x width), the CLS token and the
mean patch token of the last layer) and encoder.json. The traced module is checked against timm's on random input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch


class ClsMeanPatch(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.prefix = int(getattr(model, "num_prefix_tokens", 1))

    def forward(self, x):
        tokens = self.model.forward_features(x)
        return torch.cat([tokens[:, 0], tokens[:, self.prefix :].mean(1)], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = timm.create_model(args.model, pretrained=True, num_classes=0).eval()
    data = timm.data.resolve_data_config({}, model=model)
    module = ClsMeanPatch(model).eval()
    example = torch.randn(4, 3, args.image_size, args.image_size)
    with torch.no_grad():
        traced = torch.jit.freeze(torch.jit.trace(module, example, check_trace=False))
        probe = torch.randn(3, 3, args.image_size, args.image_size)
        expected, got = module(probe), traced(probe)
    diff = float((expected - got).abs().max())
    if diff > 1e-4:
        raise RuntimeError(f"traced encoder differs from timm's by {diff}")

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(out / "encoder.pt"))
    meta = {
        "model": args.model,
        "features": "cls + mean patch token of the last layer",
        "image_size": args.image_size,
        "mean": list(data["mean"]),
        "std": list(data["std"]),
        "feature_dim": int(got.shape[1]),
        "timm": timm.__version__,
        "torch": torch.__version__,
        "trace_max_diff": diff,
    }
    (out / "encoder.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{out}: {meta['feature_dim']}-D features of {args.model} at {args.image_size}px (traced, diff {diff:.1e})")


if __name__ == "__main__":
    main()
