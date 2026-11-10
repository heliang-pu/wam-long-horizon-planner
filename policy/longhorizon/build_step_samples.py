"""G5 samples: one model that first splits the goal into a plan, then runs the plan step by step.

- Step questions: every memory planner sample (samples_mem2.jsonl, the data G2 was trained on) as a step sample
  (planner_prompt.step_sample), so that judging a skill finished is learnt from all of G2's data; and --repeat-share of
  them once more with one or two skills doubled in the plan (repeats in a row), so that the end of a skill can be a
  switch to the same skill.
- Plan questions (text only): --plans random plans (repeats in a row too) with goals in any wording: G2's named
  sentences, purpose words (G3's "train" phrasings) and counts (按三次铃, "press the bell three times").

    python policy/longhorizon/build_step_samples.py --data D --out samples_g5.jsonl
"""

import argparse
import json
import pathlib
import random

from policy.longhorizon import planner_prompt as pp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--src", default="samples_mem2.jsonl")
    parser.add_argument("--out", default="samples_g5.jsonl")
    parser.add_argument("--repeat-share", type=float, default=0.3, help="share of step samples copied with repeats")
    parser.add_argument("--plans", type=int, default=50000, help="plan questions (text only)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data = pathlib.Path(args.data)
    rng = random.Random(f"{args.seed}/g5")
    counts = {"step": 0, "step_repeat": 0, "plan": 0, "skipped": 0}
    with (data / args.src).open() as src, (data / args.out).open("w") as out:
        for line in src:
            sample = json.loads(line)
            step = pp.step_sample(sample, rng)
            if step is None:
                counts["skipped"] += 1
                continue
            out.write(json.dumps(step, ensure_ascii=False) + "\n")
            counts["step"] += 1
            if rng.random() < args.repeat_share:
                again = pp.step_sample(sample, rng, repeats=1 if rng.random() < 0.7 else 2)
                if again is not None:
                    out.write(json.dumps(again, ensure_ascii=False) + "\n")
                    counts["step_repeat"] += 1
        for _ in range(args.plans):
            out.write(json.dumps(pp.plan_sample(rng), ensure_ascii=False) + "\n")
            counts["plan"] += 1
    print(f"{counts} -> {data / args.out}")


if __name__ == "__main__":
    main()
