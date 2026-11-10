"""G4 samples: the memory planner's samples (samples_mem2.jsonl) as asks with a skill list (planner_prompt.Listing).

- Held-out skills (--held-out, planner_prompt.HELD_OUT) never reach training: their episodes are dropped, and where a
  goal of another episode had one, a training skill takes its place (no skill twice in a row), the goal, memory and
  target written again for the new chain.
- Every sample gets its own list: the chain's skills and distractors (training skills only), shuffled, letters drawn
  anew, descriptions in varied wordings; running, memory and target in letters.
- --hints: every line also says what the scene looks like once the skill is done (planner_prompt.DONE_HINTS).
- --abstract: that share of the goals in purpose words (planner_prompt.abstract_goal_text, as G3).

    python policy/longhorizon/build_list_samples.py --data D --held-out A --out samples_list_A.jsonl
"""

import argparse
import json
import pathlib
import random

from policy.longhorizon import planner_prompt as pp


def remap_chain(chain: list[str], index: int, held: set[str], train: list[str], rng: random.Random) -> list[str]:
    """``chain`` with every held-out skill (never the one at ``index``) replaced by a training skill, no skill twice in
    a row."""
    new = list(chain)
    for j, task in enumerate(chain):
        if j == index or task not in held:
            continue
        banned = {new[j - 1]} if j > 0 else set()
        if j + 1 < len(new) and (j + 1 == index or new[j + 1] not in held):
            banned.add(new[j + 1])
        new[j] = rng.choice([t for t in train if t not in banned])
    return new


def convert(sample: dict, held: set[str], train: list[str], rng: random.Random, abstract: float,
            hints: bool = False) -> dict | None:
    """One sample in the list format, or None (a held-out skill's episode, or a memory that does not parse)."""
    chain, index, running = sample["chain"], sample["index"], sample["running"]
    if chain[index] in held:
        return None
    new = remap_chain(chain, index, held, train, rng)
    listing = pp.Listing.build(new, train, rng, vary=True, hints=hints)
    letter = listing.letter
    memory, target_memory = pp.parse_memory(sample["memory"]), pp.parse_memory(sample["target_memory"])
    if memory is None or target_memory is None:
        return None
    target = sample["target_skill"]
    if target == pp.DONE:
        target_letter = pp.DONE
    elif target == chain[index]:
        target_letter = letter(new[index])
    else:  # the next skill of the goal
        if index + 1 >= len(chain) or target != chain[index + 1]:
            return None
        target_letter = letter(new[index + 1])
    goal = (pp.abstract_goal_text(new, rng, split="train") if rng.random() < abstract else pp.any_goal_text(new, rng))
    return {
        **sample,
        "chain": new,
        "goal": goal,
        "running": letter(running) if running else None,
        "memory": pp.memory_text(new[: memory[0]], memory[1], namer=letter),
        "target_skill": target_letter,
        "target_memory": pp.memory_text(new[: target_memory[0]], target_memory[1], namer=letter),
        "listing": listing.to_json(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--src", default="samples_mem2.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--held-out", default="A", help="a key of planner_prompt.HELD_OUT, or none")
    parser.add_argument("--abstract", type=float, default=0.3, help="share of goals in purpose words")
    parser.add_argument("--hints", action="store_true", help="each list line also says when the skill is done")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    held = set() if args.held_out == "none" else set(pp.HELD_OUT[args.held_out])
    train = [t for t in pp.SKILLS if t not in held]
    data = pathlib.Path(args.data)
    rng = random.Random(f"{args.seed}/list/{args.held_out}")
    n = kept = 0
    with (data / args.src).open() as src, (data / args.out).open("w") as out:
        for line in src:
            n += 1
            new = convert(json.loads(line), held, train, rng, args.abstract, args.hints)
            if new is not None:
                kept += 1
                out.write(json.dumps(new, ensure_ascii=False) + "\n")
    print(f"{n} samples, {kept} kept (held out: {sorted(held) or 'none'}) -> {data / args.out}")


if __name__ == "__main__":
    main()
