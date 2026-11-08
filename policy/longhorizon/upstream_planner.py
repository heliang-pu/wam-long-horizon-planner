"""LeRobot's external VLM planner on our held-out episodes, to compare with the fine-tuned planner on the same scoring.

From LeRobot PR #4675 (297774f, src/lerobot/rollout/planner.py, which follows Steerable Vision-Language-Action Policies,
arXiv 2602.13193): its system prompt, rules, reply format, closed vocabulary, history of 4 (observation, command) pairs with
the model's own assessments, its reply handling (a command "in progress" sends nothing, "done" ends) and its defaults (ask
every 10 s, temperature 0.2, up to 512 tokens). No fine-tuning. It runs here in-process instead of behind a server.

Fine-tuned (Qwen3-VL-8B + LoRA, user 11-07): `build` writes <data>/samples_lerobot.jsonl in this format from the samples
of add_hard_negatives.py (not the standing-still ones: the history here is 10 s apart), and train_planner.py --format
lerobot trains on it. Each sample's history is the asks 2, 4, 6 or 10 s apart before it (mostly 10, the recipe), with
the assessments the labels give; the reply is "none" with no history, else "completed" and the next skill (or done)
once the frame is finished, else "in progress" and the same skill. Its "scene" is a template (the arms moving or still,
the running skill finished or not): there is nothing to learn a free description from. Images 320x240 (frames/, as
the other planners), not 640x480, to keep 15 images per sample trainable; replay --frames frames does the same.

What differs, because a replay is not a rollout:
- the running skill counts as issued from the first ask, and the goal names only what is left of the chain: the recording
  shows one skill, and upstream has no language memory to say that the ones before it are done;
- the vocabulary is our skills' descriptions (planner_prompt.SKILLS), which a skill is read back from.

    ../venvs/planner/bin/python policy/longhorizon/upstream_planner.py extract --data <planner data>   # 640x480 frames
    CUDA_VISIBLE_DEVICES=0 ../venvs/planner/bin/python policy/longhorizon/upstream_planner.py replay --data <planner data> \
        --base Qwen3-VL-8B-Instruct --out <dir> [--shard k/n]
    ../venvs/planner/bin/python policy/longhorizon/upstream_planner.py report --data <planner data> --out <dir> --shards n
    ../venvs/planner/bin/python policy/longhorizon/upstream_planner.py build --data <planner data>   # training samples
    ... replay --adapter <train_planner.py --format lerobot out>/adapter --frames frames [--ask-every 2]

The report is eval_planner.py --sweep's, at this rate.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.longhorizon import build_done_features as bdf  # noqa: E402
from policy.longhorizon import planner_prompt as pp  # noqa: E402
from policy.longhorizon.build_planner_data import chain_with  # noqa: E402
from policy.longhorizon.build_planner_data import episode_dir  # noqa: E402
from policy.longhorizon.eval_planner import HOLD_S  # noqa: E402
from policy.longhorizon.eval_planner import sweep_report  # noqa: E402

ASK_EVERY_S = 10  # --autosteer_interval_s
TRAIN_SPACINGS = (10, 10, 10, 2, 4, 6)  # seconds between the asks of a training sample's history: the recipe's mostly
HISTORY = 4  # EXTERNAL_HISTORY_DEFAULT
TEMPERATURE = 0.2  # VlmConfig
MAX_NEW_TOKENS = 512  # VlmConfig
ROBOT_TYPE = "dual-arm CobotMagic"
SYSTEM_PROMPT = "Help operate the robot using its current camera images."

# word for word from planner.py
REPLY_FORMAT = (
    "Reply as one JSON object with exactly these three keys, in this order, all filled in:\n"
    '"scene": one short sentence on the scene now and what changed since the previous observation;\n'
    '"previous_command": one of "completed", "in progress", "failed", "none";\n'
    '"instruction": the next instruction, or "done" when every object named in the request is already in '
    "its goal state. Never reply with the instruction alone."
)
SUBTASK_RULES = (
    "Each earlier observation is followed by the command the robot was given right after it, and the next "
    "observation shows what the robot did with that command. A command in the history was only sent to the "
    "robot; judge whether it was completed from the images alone.\n"
    "Think in this order: describe the current scene and what changed since the previous observation; decide "
    "whether the previous command is completed, still in progress, or failed; then choose the next instruction. "
    "Repeat the previous instruction word for word while the robot is still working on it. Change it only when "
    "its object is visibly in the goal state, or when the same instruction produced no visible change over two "
    "or more consecutive observations. A change means picking an instruction whose object is not yet in its "
    "goal state — re-issuing the stalled instruction counts. Never pick an instruction whose object is already "
    'in its goal state; when every object named in the request is done, answer "done".\n'
    'When no command appears in the history yet, there is no previous command: answer "none" and choose the '
    "first instruction.\n"
    "Identify objects by their shape and printed label in the images; the command text does not tell you which "
    "object moved.\n"
    "After each command in the history you will find the assessment you gave at the time; stay consistent with "
    "it unless the images show it was wrong."
)
MULTI_STEP = (
    "For a multi-step goal, choose one concrete subtask, not the entire goal. "
    "Assess previous_command against the last issued subtask, not overall goal completion."
)


def request_text(current: str, goal: str, instructions: list[str]) -> str:
    return "\n".join(
        [
            f"Robot: {ROBOT_TYPE}",
            f"Current instruction: {current}",
            f"Request: {goal}",
            SUBTASK_RULES,
            MULTI_STEP,
            f"Choose exactly one of these allowed instructions: {instructions!r}",
            REPLY_FORMAT,
        ]
    )


def normalize(text: str) -> str:
    return " ".join(text.lower().rstrip(".").split())


def json_reply(text: str) -> dict | None:
    """The reply's JSON object, as vlm_client: code fences stripped, else the first object in the text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = text[: text.rfind("```")] if "```" in text else text
    try:
        reply = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        depth = 0
        for i, char in enumerate(text[start:], start) if start >= 0 else ():
            depth += (char == "{") - (char == "}")
            if depth == 0:
                try:
                    reply = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                break
        else:
            return None
    return reply if isinstance(reply, dict) else None


def skill_of(reply: dict | None, running: str) -> str | None:
    """As VlmPlanner.parse_reply: done ends, "in progress" keeps the command whatever the instruction says; the reply must
    be in the closed vocabulary. None: not a usable reply (upstream raises; the command stays)."""
    text = reply.get("instruction") if reply else None
    if not isinstance(text, str) or not text.strip():
        return None
    if normalize(text) == pp.DONE:
        return pp.DONE
    if reply.get("previous_command") == "in progress":
        return running
    return {normalize(pp.SKILLS[name][0]): name for name in pp.SKILLS}.get(normalize(text))


@dataclass
class Ask:
    """The frames of one ask: the file of each camera, oldest ask first with the command and assessment that followed."""

    frame: int
    command: str
    assessment: str = ""


def ask_frames(n: int, every: int = ASK_EVERY_S) -> list[int]:
    """Asked every ``every`` s up to HOLD_S after the end of a recording of n frames; later ones stand still at its end."""
    return list(range(0, n + HOLD_S + every, every))


def frame_path(data: Path, name: str, k: int, cam: str, frames: str = "frames_full") -> Path:
    return data / frames / name / f"{k:04d}_{cam}.jpg"


def extract_one(item: tuple[dict, str, int]) -> None:
    from PIL import Image

    row, out, n = item
    views, _state, _keep = bdf.decode_frames(row, max(1, round(row["fps"])))
    name = episode_dir(row)
    for k in {min(k, n - 1) for k in ask_frames(n)}:
        for cam in pp.CAMERAS:
            path = frame_path(Path(out), name, k, cam)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".tmp")
            Image.fromarray(views[cam][k]).save(partial, "JPEG", quality=90)
            partial.replace(path)


def held_out(data: Path) -> list[dict]:
    episodes = [json.loads(line) for line in (data / "episodes.jsonl").read_text().splitlines()]
    return sorted((e for e in episodes if e["split"] == "val"), key=lambda e: e["dir"])


def extract(args: argparse.Namespace) -> None:
    data = Path(args.data).expanduser()
    held = {e["dir"]: e for e in held_out(data)}
    datasets = bdf.official_datasets(REPO_ROOT / "Real", pp.SKILLS) + bdf.our_datasets(
        REPO_ROOT / "real_rollouts", pp.SKILLS
    )
    rows = [row for dataset in datasets for row in bdf.episodes_of(dataset, {}) if episode_dir(row) in held]
    todo = [(row, str(data), len(held[episode_dir(row)]["frame_index"])) for row in rows]
    print(f"{len(todo)} of {len(held)} held-out episodes", flush=True)
    with cf.ProcessPoolExecutor(args.workers) as pool:
        for n, _ in enumerate(pool.map(extract_one, todo, chunksize=4), 1):
            if n % 100 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def observation_blocks(data: Path, name: str, label: str, k: int, frames: str = "frames_full") -> list[dict]:
    from PIL import Image

    blocks = [{"type": "text", "text": label}]
    for cam in pp.CAMERAS:
        blocks.append({"type": "text", "text": f"Camera: {cam}"})
        blocks.append({"type": "image", "image": Image.open(frame_path(data, name, k, cam, frames)).convert("RGB")})
    return blocks


def messages_of(
    data: Path, name: str, current: str, goal: str, history: list[Ask], k: int, frames: str = "frames_full"
) -> list[dict]:
    instructions = [pp.SKILLS[skill][0] for skill in pp.SKILLS]
    content = [{"type": "text", "text": request_text(current, goal, instructions)}]
    for index, past in enumerate(history, start=1):
        content += observation_blocks(data, name, f"Observation {index}:", past.frame, frames)
        content.append({"type": "text", "text": f"Command given after observation {index}: {past.command}"})
        if past.assessment:
            content.append({"type": "text", "text": f"Your assessment after observation {index}: {past.assessment}"})
    content += observation_blocks(data, name, "Current observation:", k, frames)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def version() -> str:
    """The request, reply and training formats, hashed: a fine-tuned adapter is replayed with the format it learnt."""
    import hashlib

    rendered = [SYSTEM_PROMPT, request_text("<current>", "<goal>", [pp.SKILLS[t][0] for t in pp.SKILLS]),
                scene_text("click_bell", finished=True, still=True), list(TRAIN_SPACINGS), HISTORY]  # fmt: skip
    return hashlib.sha1(json.dumps(rendered).encode()).hexdigest()[:12]


def arms_still(rows: list | None, k: int) -> bool:
    """The hands (states.json rows, one a second) moved under 1 cm and the grippers under 2 % since the second before."""
    if not rows or k <= 0:
        return False
    now, before = rows[min(k, len(rows) - 1)], rows[min(k - 1, len(rows) - 1)]
    moved = max(abs(now[i] - before[i]) for i in (0, 1, 2, 4, 5, 6))
    return moved < 0.01 and max(abs(now[3] - before[3]), abs(now[7] - before[7])) < 0.02


def scene_text(running: str, *, finished: bool, still: bool) -> str:
    motion = "The arms are still" if still else "The arms are moving"
    state = "done: its objects are in the goal state" if finished else "not finished yet"
    return f"{motion}; '{pp.SKILLS[running][0]}' is {state}."


def verdict_at(episode: dict, k: int, *, first: bool) -> str:
    return "none" if first else ("completed" if episode["labels"][k] == 1 else "in progress")


def lerobot_sample(sample: dict, episode: dict, rows: list | None, rng: random.Random) -> dict:
    """A training sample in this format from an add_hard_negatives.py one (its episode, frame and finished label)."""
    k, running = sample["frame"], episode["task"]
    chain, index = chain_with(running, rng)
    chain = chain[index:]  # the goal names what is left, as in the replay
    spacing = rng.choice(TRAIN_SPACINGS)
    frames = [k - spacing * j for j in range(HISTORY, 0, -1) if k - spacing * j >= 0]
    history = []
    for j, frame in enumerate(frames):
        scene = scene_text(running, finished=episode["labels"][frame] == 1, still=arms_still(rows, frame))
        verdict = verdict_at(episode, frame, first=j == 0 and frame - spacing < 0)
        history.append({"frame": frame, "assessment": f"scene: {scene}; previous command: {verdict}"})
    finished = bool(sample["finished"])
    nxt = pp.next_skill(chain, 0) if finished else running
    reply = {
        "scene": scene_text(running, finished=finished, still=arms_still(rows, k)),
        "previous_command": "none" if not frames else ("completed" if finished else "in progress"),
        "instruction": pp.DONE if nxt == pp.DONE else pp.SKILLS[nxt][0],
    }
    return {"episode": sample["episode"], "frame": k, "task": running, "chain": chain, "spacing": spacing,
            "goal": pp.goal_text(chain, rng), "history": history, "reply": reply, "finished": finished}  # fmt: skip


def encode_lerobot(processor, data: Path, sample: dict) -> tuple[dict, str]:
    """A training sample's input (chat template with the generation prompt, images from frames/) and its reply text."""
    current = pp.SKILLS[sample["task"]][0]
    history = [Ask(h["frame"], current, h["assessment"]) for h in sample["history"]]
    messages = messages_of(data, sample["episode"], current, sample["goal"], history, sample["frame"], "frames")
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    return inputs, json.dumps(sample["reply"], ensure_ascii=False)


def write_needed(item: tuple) -> None:
    """The 320x240 JPEGs of these frames that are not there yet: (row, data dir, episode dir, frames[, cameras])."""
    from PIL import Image

    row, out, name, frames = item[:4]
    cams = item[4] if len(item) > 4 else pp.CAMERAS
    root = Path(out) / "frames" / name
    missing = [(k, cam) for k in sorted(set(frames)) for cam in cams if not (root / f"{k:04d}_{cam}.jpg").exists()]
    if not missing:
        return
    views, _state, _keep = bdf.decode_frames(row, max(1, round(row["fps"])))
    root.mkdir(parents=True, exist_ok=True)
    for k, cam in missing:
        path = root / f"{k:04d}_{cam}.jpg"
        partial = path.with_suffix(".tmp")
        Image.fromarray(views[cam][k]).resize(pp.IMAGE_SIZE, Image.BILINEAR).save(partial, "JPEG", quality=90)
        partial.replace(path)


def extract_needed(data: Path, needed: dict[str, list[int]], cams=pp.CAMERAS, workers: int = 16) -> None:
    """write_needed for every episode of ``needed`` (episode dir -> frames), from the recordings."""
    import multiprocessing as mp

    datasets = bdf.official_datasets(REPO_ROOT / "Real", pp.SKILLS) + bdf.our_datasets(
        REPO_ROOT / "real_rollouts", pp.SKILLS
    )
    rows = {episode_dir(row): row for dataset in datasets for row in bdf.episodes_of(dataset, {})}
    todo = [(rows[name], str(data), name, frames, tuple(cams)) for name, frames in sorted(needed.items())]
    print(f"frames of {len(todo)} episodes", flush=True)
    with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as executor:
        for n, _ in enumerate(executor.map(write_needed, todo, chunksize=4), 1):
            if n % 500 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def build(args: argparse.Namespace) -> None:
    import multiprocessing as mp

    data = Path(args.data).expanduser()
    episodes = {e["dir"]: e for e in (json.loads(line) for line in (data / "episodes.jsonl").read_text().splitlines())}
    states = json.loads((data / "states.json").read_text())
    datasets = bdf.official_datasets(REPO_ROOT / "Real", pp.SKILLS) + bdf.our_datasets(
        REPO_ROOT / "real_rollouts", pp.SKILLS
    )
    rows = {episode_dir(row): row for dataset in datasets for row in bdf.episodes_of(dataset, {})}
    rng = random.Random(args.seed)
    pool = [json.loads(line) for line in (data / args.samples).read_text().splitlines() if line.strip()]
    pool = [s for s in pool if "still" not in s and s["episode"] in rows]
    pool = [s for s in pool if len(range(0, rows[s["episode"]]["length"], max(1, round(rows[s["episode"]]["fps"]))))
            == len(episodes[s["episode"]]["frame_index"])]  # fmt: skip
    chosen = rng.sample(pool, min(args.count, len(pool)))
    samples = [lerobot_sample(s, episodes[s["episode"]], states.get(s["episode"]), rng) for s in chosen]
    needed: dict[str, list[int]] = {}
    for s in samples:
        needed.setdefault(s["episode"], []).extend([s["frame"], *(h["frame"] for h in s["history"])])
    todo = [(rows[name], str(data), name, frames) for name, frames in sorted(needed.items())]
    print(f"{len(samples)} samples from {len(todo)} episodes; writing the frames they need", flush=True)
    with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as executor:
        for n, _ in enumerate(executor.map(write_needed, todo, chunksize=4), 1):
            if n % 200 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    partial = data / "samples_lerobot.jsonl.tmp"
    partial.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples))
    partial.replace(data / "samples_lerobot.jsonl")
    finished = sum(s["finished"] for s in samples)
    print(f"-> {data / 'samples_lerobot.jsonl'} ({finished} finished)", flush=True)


def replay_episode(
    model, processor, data: Path, episode: dict, device: str, seed: int, every: int = ASK_EVERY_S, frames="frames_full"
) -> dict:
    import torch

    name, running = episode["dir"], episode["task"]
    rng = random.Random(f"{seed}/{name}")
    chain, index = chain_with(running, rng)
    goal, current = pp.goal_text(chain[index:], rng), pp.SKILLS[running][0]
    n = len(episode["frame_index"])
    history, skills, replies = [], [], []
    for frame in ask_frames(n, every):
        k = min(frame, n - 1)
        inputs = processor.apply_chat_template(
            messages_of(data, name, current, goal, history, k, frames),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE)
        text = processor.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        reply = json_reply(text)
        skills.append(skill_of(reply, running))
        replies.append(text)
        scene = reply.get("scene", "") if reply else ""
        verdict = reply.get("previous_command", "") if reply else ""
        assessment = "; ".join(
            p for p in (f"scene: {scene}" if scene else "", f"previous command: {verdict}" if verdict else "") if p
        )
        history = [*history, Ask(k, current, assessment)][-HISTORY:]
    return {"episode": name, "task": running, "next": pp.next_skill(chain, index), "ask_every": every,
            "goal": goal, "skills": skills, "replies": replies}  # fmt: skip


def replay(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForImageTextToText
    from transformers import AutoProcessor

    data, out = Path(args.data).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    k, n = (int(v) for v in args.shard.split("/"))
    chosen = held_out(data)[k::n]
    if args.limit:
        chosen = chosen[: args.limit]
    torch.manual_seed(args.seed + k)
    processor = AutoProcessor.from_pretrained(args.base)
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16)
    if args.adapter:
        from peft import PeftModel

        meta = json.loads((Path(args.adapter) / "planner.json").read_text())
        if meta.get("prompt_version") != version():
            raise ValueError(f"{args.adapter} learnt format {meta.get('prompt_version')}, this file is {version()}")
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to(args.device).eval()
    with (out / f"answers.shard{k}.jsonl").open("w") as f:
        for i, episode in enumerate(chosen, 1):
            record = replay_episode(
                model, processor, data, episode, args.device, args.seed, args.ask_every, args.frames
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i}/{len(chosen)} {record['task']}: {record['skills']}", flush=True)


def report(args: argparse.Namespace) -> None:
    data, out = Path(args.data).expanduser(), Path(args.out).expanduser()
    records = []
    for k in range(args.shards):
        records += [json.loads(line) for line in (out / f"answers.shard{k}.jsonl").read_text().splitlines()]
    (out / "answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    episodes = {
        json.loads(line)["dir"]: json.loads(line) for line in (data / "episodes.jsonl").read_text().splitlines()
    }
    takeovers = json.loads((data / "takeovers.json").read_text())
    table, text = sweep_report(records, episodes, takeovers, rates=(records[0]["ask_every"],))
    asks = sum(len(r["skills"]) for r in records)
    invalid = sum(s is None for r in records for s in r["skills"])
    text += f"\n{len(records)} episodes, {asks} asks, {invalid} unusable replies.\n"
    (out / "sweep.json").write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n")
    (out / "sweep.md").write_text(text)
    print(text, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("extract", "replay", "report", "build"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--base", default=None)
    parser.add_argument("--shard", default="0/1", help="k/n: every n-th held-out episode, from the k-th")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="at most this many episodes (smoke tests)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter", default=None, help="replay: a train_planner.py --format lerobot adapter")
    parser.add_argument("--frames", default="frames_full", help="replay: frames_full (640x480) or frames (320x240)")
    parser.add_argument("--ask-every", type=int, default=ASK_EVERY_S, help="replay: seconds between asks")
    parser.add_argument("--samples", default="samples_hard.jsonl", help="build: the samples to take frames from")
    parser.add_argument("--count", type=int, default=32000, help="build: training samples")
    args = parser.parse_args()
    {"extract": extract, "replay": replay, "report": report, "build": build}[args.command](args)


if __name__ == "__main__":
    main()
