# ruff: noqa: RUF001 - the Chinese goal and memory templates use Chinese punctuation
"""The high-level planner's text, shared by the data builder, training, the worker and the evaluation (pure Python), so
what it is trained on is what it is asked on the robot.

The planner works as π0.5's high-level step with MEM's two memories (Physical Intelligence, arXiv 2504.16054 and
2603.03596): from the goal (one sentence), the skill running now, its language memory (what is finished) and the images
(the three cameras now, plus the top camera 3, 2 and 1 s ago: the short-term visual memory), it names the skill to run
(one of the skills the WAM was trained on, or done) and writes the memory back. The memory only changes when a skill's
job is finished, as in MEM.
"""

from __future__ import annotations

import random
import re

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
PAST_S = (3, 2, 1)  # the top camera this many seconds ago: the short-term visual memory
LONG_PAST_S = (20, 10, 3, 2, 1)  # G2: also 10 and 20 s before (planner.json "past_s"), for asks a few seconds apart
IMAGE_SIZE = (320, 240)  # width, height of every image the planner sees
DONE = "done"
NO_MEMORY = "无"
STATE_UNKNOWN = "unknown"  # the hands' line of a planner trained with them, when they are dropped or not available

# skill -> (English description in the prompt, Chinese name in the memory, Chinese phrasings for goals)
SKILLS = {
    "click_bell": ("press the bell", "按铃", ["按铃", "按一下铃", "把铃按响", "按响服务铃"]),
    "drawer_open_place": (
        "open the drawer and put the tomato into it",
        "开抽屉放番茄",
        ["打开抽屉把番茄放进去", "把番茄放进抽屉", "开抽屉放番茄"],
    ),
    "handle_basket": (
        "put the bottle into the basket and carry the basket",
        "瓶子放进篮子并提篮子",
        ["把瓶子放进篮子再提起篮子", "把瓶子装进篮子并拎走", "瓶子放篮子里提起来"],
    ),
    "item_assembly": (
        "dock the two centrifuge tubes",
        "对接离心管",
        ["把两根离心管对接起来", "组装离心管", "对接两根管子"],
    ),
    "items_handover": (
        "hand the pen to the other arm and put it into the pen holder",
        "递笔放进笔筒",
        ["把笔递给另一只手再放进笔筒", "递笔并插进笔筒", "把笔放进笔筒"],
    ),
    "manipulate_pipette": (
        "transfer liquid between the beakers with the pipette",
        "用移液枪移液",
        ["用移液枪把液体移到另一个烧杯", "用移液枪移液", "拿移液枪转移液体"],
    ),
    "mixer_operating": (
        "put the beaker on the mixer and switch it on",
        "烧杯放搅拌器并开机",
        ["把烧杯放到搅拌器上再打开开关", "把烧杯放上搅拌器并启动", "用搅拌器搅一下烧杯"],
    ),
    "sample_loading": (
        "pass the test tube to the other arm and insert it into the rack",
        "试管插进试管架",
        ["把试管递过去插进试管架", "把试管插到架子上", "装载样品试管"],
    ),
    "table_rearrangement": (
        "place the fork and the spoon next to the plate",
        "叉勺摆到盘边",
        ["把叉子和勺子摆到盘子两边", "摆好餐具", "把叉勺放到盘子旁边"],
    ),
    "water_pouring": (
        "pour water from the bottle into the cup",
        "倒水",
        ["把瓶子里的水倒进杯子", "倒一杯水", "倒水"],
    ),
}

GOAL_TEMPLATES_ZH = {
    1: ["{0}。", "请{0}。", "帮我{0}"],
    2: ["先{0}，然后{1}。", "请依次{0}、{1}。", "{0}，再{1}", "先{0}，接着{1}"],
    3: ["先{0}，然后{1}，最后{2}。", "请依次{0}、{1}、{2}。", "{0}，{1}，再{2}", "先{0}，接着{1}，最后{2}"],
    4: ["先{0}，然后{1}，再{2}，最后{3}。", "请依次{0}、{1}、{2}、{3}。", "{0}，{1}，{2}，最后{3}"],
}
GOAL_TEMPLATES_EN = {
    1: ["Please {0}.", "{0}."],
    2: ["First {0}, then {1}.", "Please {0} and then {1}."],
    3: ["First {0}, then {1}, and finally {2}.", "Please {0}, {1}, then {2}."],
    4: ["First {0}, then {1}, then {2}, and finally {3}.", "Please {0}, {1}, {2} and then {3}."],
}


def goal_text(chain: list[str], rng: random.Random) -> str:
    """A one-sentence goal asking for the skills of ``chain`` in this order (Chinese mostly, English sometimes)."""
    if rng.random() < 0.25:
        return rng.choice(GOAL_TEMPLATES_EN[len(chain)]).format(*(SKILLS[t][0] for t in chain))
    return rng.choice(GOAL_TEMPLATES_ZH[len(chain)]).format(*(rng.choice(SKILLS[t][2]) for t in chain))


MAX_CHAIN = 10  # memory planners: goals of up to this many skills, a skill may come back (not twice in a row)


def any_chain(task: str, rng: random.Random, max_len: int = MAX_CHAIN) -> tuple[list[str], int]:
    """A random goal of 1..max_len skills with ``task`` in it, repeats allowed but never the same skill twice in a row
    (a switch is a reply naming another skill); (chain, task's index). Short goals are the most common."""
    length = rng.choice([1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10])
    length = min(length, max_len)
    index = rng.randrange(length)
    chain = [task]
    for _ in range(index):  # leftwards from task
        chain.insert(0, rng.choice([t for t in SKILLS if t != chain[0]]))
    while len(chain) < length:
        chain.append(rng.choice([t for t in SKILLS if t != chain[-1]]))
    return chain, index


def any_goal_text(chain: list[str], rng: random.Random) -> str:
    """goal_text for any length: Chinese mostly, English sometimes, the phrasings of each skill varied."""
    if rng.random() < 0.25:
        parts = [SKILLS[t][0] for t in chain]
        if len(parts) == 1:
            return rng.choice(["Please {0}.", "{0}."]).format(parts[0])
        if rng.random() < 0.5:
            return "First " + ", then ".join(parts[:-1]) + f", and finally {parts[-1]}."
        return "Please " + ", ".join(parts[:-1]) + f" and then {parts[-1]}."
    parts = [rng.choice(SKILLS[t][2]) for t in chain]
    if len(parts) == 1:
        return rng.choice(["{0}。", "请{0}。", "帮我{0}"]).format(parts[0])
    style = rng.randrange(3)
    if style == 0:
        middle = "".join(f"，{rng.choice(['然后', '再', '接着'])}{p}" for p in parts[1:-1])
        return f"先{parts[0]}{middle}，最后{parts[-1]}。"
    if style == 1:
        return "请依次" + "、".join(parts) + "。"
    return "，".join(parts[:-1]) + f"，最后{parts[-1]}"


STAGES = 4  # memory planners: how far the running skill is, 0..STAGES-1 of STAGES (STAGES: it looks finished)


def memory_text(finished: list[str], progress: int | None = None) -> str:
    """The language memory: the skills finished so far, in order. A memory planner (progress not None) also keeps how
    far the running skill is, as the stage tracking of the 2025 BEHAVIOR winner: start and end of a skill can look the
    same (a bell before and after the press), what was seen in between is in the memory."""
    if progress is None:
        return "已完成：" + "；".join(SKILLS[t][1] for t in finished) + "。" if finished else NO_MEMORY
    done = "；".join(SKILLS[t][1] for t in finished) if finished else "无"
    return f"已完成：{done}。当前进度：{progress}/{STAGES}。"


def parse_memory(text: str | None) -> tuple[int, int] | None:
    """(number of finished skills, progress) of a memory planner's memory, or None."""
    if not text:
        return None
    done = re.search(r"已完成：(.*?)。", text)
    stage = re.search(r"当前进度：(\d+)\s*/\s*\d+", text)
    if not done or not stage:
        return None
    items = [item for item in done.group(1).split("；") if item.strip() and item.strip() != "无"]
    return len(items), min(int(stage.group(1)), STAGES)


def progress_of(k: int, finished_at: int | None) -> int:
    """The progress stage at planner frame k of a skill finished from frame finished_at (None: never): its share of the
    time to there, in STAGES; STAGES from there on."""
    if finished_at is None or finished_at <= 0:
        return 0
    if k >= finished_at:
        return STAGES
    return min(STAGES - 1, STAGES * k // finished_at)


class MemoryTracker:
    """The memory a planner is asked with, as the robot keeps it: the finished skills are the switches that happened
    (the planner's own decisions), and for a memory planner the progress it writes, a stage taken once two of its last
    three replies are at least there (the BEHAVIOR winner's vote) and never going back within a skill."""

    def __init__(self, *, progress: bool, finished: list[str] | None = None):
        self.uses_progress = progress
        self.finished = list(finished or [])
        self.progress, self.votes = 0, []

    def text(self) -> str:
        return memory_text(self.finished, self.progress if self.uses_progress else None)

    def observe(self, reply_memory: str | None) -> None:
        """A reply about the running skill: its memory counts as a vote (STAGES if it lists the skill as finished)."""
        if not self.uses_progress:
            return
        parsed = parse_memory(reply_memory)
        if parsed is None:
            return
        n_finished, stage = parsed
        self.votes = [*self.votes, STAGES if n_finished > len(self.finished) else stage][-3:]
        if len(self.votes) >= 2:
            self.progress = max(self.progress, sorted(self.votes)[(len(self.votes) - 1) // 2])

    def finish(self, skill: str) -> None:
        """The running skill ended as finished (a switch, or the operator)."""
        self.finished.append(skill)
        self.progress, self.votes = 0, []

    def restart(self) -> None:
        """The running skill starts over (a timeout, a takeover): its progress is not kept."""
        self.progress, self.votes = 0, []


def hands_text(rows: list[list[float]], k: int, rng: random.Random | None = None) -> str:
    """The hands at the moments of the images (the PAST_S seconds before frame k, then k), oldest first. rows: per frame
    at the planner's rate, left x, y, z (m), opening (0-1), right the same. With rng each hand is shifted by up to 1 cm
    for the whole text, as tables and arm mounts differ by that much."""
    shifts = [[rng.uniform(-0.01, 0.01) for _ in range(3)] if rng else [0.0] * 3 for _ in range(2)]
    parts = []
    for name, j in [*((f"{s} s ago", max(0, k - s)) for s in PAST_S), ("now", k)]:
        hands = []
        for side, shift, values in zip("LR", shifts, (rows[j][:4], rows[j][4:]), strict=True):
            x, y, z = (round(100 * (v + d)) for v, d in zip(values[:3], shift, strict=True))
            hands.append(f"{side} {x} {y} {z} open {5 * round(20 * values[3])}")
        parts.append(f"{name}: " + ", ".join(hands))
    return "; ".join(parts)


def prompt_text(
    goal: str,
    running: str | None,
    memory: str,
    skills=tuple(SKILLS),
    state: str | None = None,
    *,
    progress: bool = False,
    past=PAST_S,
) -> str:
    lines = "\n".join(f"- {name}: {SKILLS[name][0]}" for name in skills)
    hands = "" if state is None else f"Hands (x y z in cm, each in its own arm's frame; gripper open %): {state}\n"
    if progress:
        return (
            "You are the high-level planner of a dual-arm robot.\n"
            f"Goal: {goal}\n"
            f"Now running: {running or 'nothing yet'}\n"
            f"Memory: {memory}\n"
            f"{hands}"
            "Images: the current top, left-wrist and right-wrist cameras, then the top camera "
            + ", ".join(str(s) for s in past)
            + " seconds ago.\n"
            f"Skills:\n{lines}\n"
            "The goal's skills run in its order; a skill may come more than once. The memory lists the finished "
            f"skills in order and how far the running skill is (stage 0 to {STAGES - 1} of {STAGES}): a skill can end "
            "where it started, so trust the memory for what happened before the images.\n"
            "Which skill should run now? Keep the running skill until its job is finished; then pick the next skill "
            "the goal needs after the finished ones, or done if the whole goal is finished. Write the memory as it is "
            "now.\nAnswer in two lines:\nskill: <a skill name, or done>\nmemory: <the memory now>"
        )
    return (
        "You are the high-level planner of a dual-arm robot.\n"
        f"Goal: {goal}\n"
        f"Now running: {running or 'nothing yet'}\n"
        f"Memory: {memory}\n"
        f"{hands}"
        "Images: the current top, left-wrist and right-wrist cameras, then the top camera "
        + ", ".join(str(s) for s in PAST_S)
        + " seconds ago.\n"
        f"Skills:\n{lines}\n"
        "Which skill should run now? Keep the running skill until its job is visibly finished; then pick the next "
        "skill the goal needs, or done if the whole goal is finished. Update the memory only when a skill's job is "
        "finished.\nAnswer in two lines:\nskill: <a skill name, or done>\nmemory: <what is finished>"
    )


def target_text(skill: str, memory: str) -> str:
    return f"skill: {skill}\nmemory: {memory}"


def parse_reply(text: str, skills=tuple(SKILLS)) -> tuple[str | None, str | None]:
    """(skill, memory) of a reply; skill is None when the reply names none of ``skills`` / done."""
    skill = re.search(r"skill\s*:\s*([A-Za-z_]+)", text)
    memory = re.search(r"memory\s*:\s*(.+)", text)
    name = skill.group(1).strip().lower() if skill else None
    return (name if name in (*skills, DONE) else None), (memory.group(1).strip() if memory else None)


def next_skill(chain: list[str], index: int) -> str:
    return chain[index + 1] if index + 1 < len(chain) else DONE


def version(*, state: bool = False, progress: bool = False, past=PAST_S) -> str:
    """What the planner is asked with (prompt, reply and memory formats, skills, images), hashed: an adapter is only
    used with the version it was trained on. Comments and formatting of this file do not change it."""
    import hashlib
    import json

    rendered = {
        "prompt": prompt_text("<goal>", "<running>", "<memory>"),
        "first": prompt_text("<goal>", None, NO_MEMORY),
        "reply": target_text("<skill>", "<memory>"),
        "memory": memory_text(["click_bell", "water_pouring"]),
        "skills": {name: description for name, (description, _zh, _goals) in SKILLS.items()},
        "images": [list(CAMERAS), list(PAST_S), list(IMAGE_SIZE)],
    }
    if state:
        rendered["hands"] = prompt_text("<goal>", "<running>", "<memory>", state=hands_text([[0.1] * 8] * 4, 3))
        rendered["hands_unknown"] = STATE_UNKNOWN
    if progress:
        rendered["progress"] = [
            prompt_text("<goal>", "<running>", "<memory>", progress=True),
            memory_text(["click_bell"], 2),
            STAGES,
        ]
    if tuple(past) != PAST_S:
        rendered["past"] = [list(past), prompt_text("<goal>", "<running>", "<memory>", progress=progress, past=past)]
    return hashlib.sha1(json.dumps(rendered, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]
