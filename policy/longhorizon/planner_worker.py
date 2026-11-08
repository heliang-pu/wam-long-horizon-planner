"""High-level planner worker for policy/longhorizon (planner.PlannerClient): one JSON request per line on stdin, one
reply per line on stdout. Runs under the planner's interpreter (venvs/planner, setup_planner_env.sh).

    {"cmd": "info"}                                   -> {"ok": true, "info": {...}}
    {"cmd": "plan", "images_path": <npz: image_0..image_5, HWC uint8 RGB: cam_high, cam_left_wrist, cam_right_wrist
     now, cam_high 3, 2, 1 s ago>, "goal": str, "running": str | null, "memory": str, "skills": [str]}
                                                      -> {"ok": true, "skill": str | null, "memory": str | null,
                                                          "reply": str, "infer_s": float}
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.longhorizon import planner_prompt as pp  # noqa: E402
from policy.longhorizon.planner_runtime import BASE  # noqa: E402
from policy.longhorizon.planner_runtime import Planner  # noqa: E402
from policy.longhorizon.planner_runtime import planner_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", default="base", help="a train_planner.py adapter dir, or base (zero-shot)")
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    # everything the model libraries print goes to stderr; stdout carries only the JSON replies
    stdout = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    planner = Planner(args.adapter, base=args.base, device=args.device)
    blank = planner_image(np.full((480, 640, 3), 128, np.uint8))
    _skill, _memory, _reply, took = planner.plan([blank] * 6, "warm up", None, pp.NO_MEMORY)  # kernels compile here
    print(f"[planner] ready ({took:.1f} s warm-up): {json.dumps(planner.info())}", file=sys.stderr, flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if request["cmd"] == "info":
                reply = {"ok": True, "info": planner.info()}
            elif request["cmd"] == "plan":
                with np.load(request["images_path"]) as data:
                    images = [planner_image(data[f"image_{k}"]) for k in range(6)]
                skills = tuple(request.get("skills") or ())
                skill, memory, text, took = planner.plan(
                    images,
                    request["goal"],
                    request.get("running"),
                    request["memory"],
                    **({"skills": skills} if skills else {}),
                )
                reply = {"ok": True, "skill": skill, "memory": memory, "reply": text, "infer_s": took}
            else:
                raise ValueError(f"unknown cmd {request['cmd']!r}")
        except Exception as exc:  # report and keep serving; the client raises it
            import traceback

            traceback.print_exc()
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
    with contextlib.suppress(Exception):
        stdout.close()


if __name__ == "__main__":
    main()
