"""G3 samples: the memory planner's samples (samples_mem2.jsonl) with a share of the goals rewritten in purpose words
(planner_prompt.abstract_goal_text, the "train" phrasings), the rest unchanged; images, memory and targets stay.

    python policy/longhorizon/add_abstract_goals.py --data D --src samples_mem2.jsonl --out samples_mem3.jsonl --share 0.5
"""

import argparse
import json
import pathlib
import random

from policy.longhorizon import planner_prompt as pp


def rewrite(sample: dict, rng: random.Random, share: float) -> dict:
    if rng.random() >= share:
        return sample
    return {**sample, "goal": pp.abstract_goal_text(sample["chain"], rng, split="train"), "goal_style": "abstract"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--src", default="samples_mem2.jsonl")
    parser.add_argument("--out", default="samples_mem3.jsonl")
    parser.add_argument("--share", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data = pathlib.Path(args.data)
    rng = random.Random(f"{args.seed}/abstract")
    n = changed = 0
    with (data / args.src).open() as src, (data / args.out).open("w") as out:
        for line in src:
            sample = json.loads(line)
            new = rewrite(sample, rng, args.share)
            changed += new is not sample
            n += 1
            out.write(json.dumps(new, ensure_ascii=False) + "\n")
    print(f"{n} samples, {changed} goals in purpose words -> {data / args.out}")


if __name__ == "__main__":
    main()
