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


# G3 (11-09, the user: "我写提示词，他自动划分任务"): goals that say what for, not which skill. Each skill has purpose
# phrasings; "train" ones go into the training samples (add_abstract_goals.py), "eval" ones are held out for
# eval_planner.py --goal-style abstract, so the test asks in words the planner never saw.
ABSTRACT = {
    "click_bell": {"train": ["叫一下人", "通知一下前台", "提醒大家一声", "打个招呼叫人过来"],
                   "eval": ["喊人过来", "发个信号通知一下"]},
    "drawer_open_place": {"train": ["把番茄收起来", "把桌上的番茄收好", "把番茄放到看不见的地方", "收拾一下番茄"],
                          "eval": ["把水果收进柜子", "番茄别放桌上了，收好"]},
    "handle_basket": {"train": ["把水瓶装好拎走", "把瓶子带上走", "收拾瓶子准备带走", "把那瓶水打包拿起来"],
                      "eval": ["把水装篮子里带走", "准备带瓶水出门"]},
    "item_assembly": {"train": ["把离心管装好", "把两截管子接上", "组装一下样品管", "把管子拼起来"],
                      "eval": ["把分开的管子合上", "接好那两根管"]},
    "items_handover": {"train": ["把笔收好", "把桌上的笔归位", "整理一下笔", "笔别乱放，放回去"],
                       "eval": ["把笔插回去", "收一下桌上的笔"]},
    "manipulate_pipette": {"train": ["转移一点液体", "分一点溶液到另一个烧杯", "把液体分装一下", "取点液体挪到另一边"],
                           "eval": ["给另一个烧杯加点溶液", "移一点液体过去"]},
    "mixer_operating": {"train": ["把溶液搅匀", "搅拌一下烧杯里的东西", "把烧杯拿去搅一搅", "开始搅拌样品"],
                        "eval": ["让溶液混合均匀", "把样品搅一下"]},
    "sample_loading": {"train": ["把试管放回去", "把样品试管归位", "把试管收到架子上", "整理一下试管"],
                       "eval": ["试管别放桌上，放好", "把样品放回原位"]},
    "table_rearrangement": {"train": ["把餐具摆好", "准备开饭", "把叉子勺子放好位置", "布置一下餐桌"],
                            "eval": ["摆一下吃饭的家伙", "把餐桌整理好准备吃饭"]},
    "water_pouring": {"train": ["给杯子加点水", "来杯水", "准备一杯水", "把杯子倒满水"],
                      "eval": ["我渴了，弄杯水", "往杯子里添点水"]},
}


def abstract_goal_text(chain: list[str], rng: random.Random, split: str = "train", named_share: float = 0.3) -> str:
    """A goal in purpose words (ABSTRACT[task][split]); each skill keeps its usual words with ``named_share``, so goals
    that mix both come up too. Orders: 先..然后..最后, 第一..第二.., a list, or (two skills) "B 之前先 A"."""
    parts = [rng.choice(SKILLS[t][2]) if rng.random() < named_share else rng.choice(ABSTRACT[t][split]) for t in chain]
    if len(parts) == 1:
        return rng.choice(["{0}。", "帮我{0}", "能不能{0}？", "麻烦{0}。"]).format(parts[0])
    style = rng.randrange(5 if len(parts) == 2 else 4)
    if style == 0:
        middle = "".join(f"，{rng.choice(['然后', '再', '接着'])}{p}" for p in parts[1:-1])
        return f"先{parts[0]}{middle}，最后{parts[-1]}。"
    if style == 1:
        nums = "一二三四五六七八九十"
        return "；".join(f"第{nums[i]}，{p}" for i, p in enumerate(parts)) + "。"
    if style == 2:
        return "帮我" + "，".join(parts[:-1]) + f"，完了{parts[-1]}"
    if style == 3:
        return "，".join(f"{p}" for p in parts[:-1]) + f"，之后{parts[-1]}。"
    return f"{parts[1]}之前，先{parts[0]}。"


STAGES = 4  # memory planners: how far the running skill is, 0..STAGES-1 of STAGES (STAGES: it looks finished)


def memory_text(finished: list[str], progress: int | None = None, namer=None) -> str:
    """The language memory: the skills finished so far, in order. A memory planner (progress not None) also keeps how
    far the running skill is, as the stage tracking of the 2025 BEHAVIOR winner: start and end of a skill can look the
    same (a bell before and after the press), what was seen in between is in the memory."""
    name = namer or (lambda t: SKILLS[t][1])  # G4 (skill list): the skills' letters in that list
    if progress is None:
        return "已完成：" + "；".join(name(t) for t in finished) + "。" if finished else NO_MEMORY
    done = "；".join(name(t) for t in finished) if finished else "无"
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

    def __init__(self, *, progress: bool, finished: list[str] | None = None, namer=None):
        self.uses_progress, self.namer = progress, namer
        self.finished = list(finished or [])
        self.progress, self.votes = 0, []

    def text(self) -> str:
        return memory_text(self.finished, self.progress if self.uses_progress else None, self.namer)

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


# ----- G4: the skills as an outside list (11-09, the user: 可以调度什么技能可以使用外部的技能列表) -------------------
# The planner is not told the robot's skills by name: every ask carries a list of skills, each a letter and a
# description, letters and order drawn anew, distractors among them. It answers with a letter (or done) and writes its
# memory in letters, so it has to read the list instead of knowing the ten skills by heart; skills held out of
# training (HELD_OUT) test whether a skill it never saw, given in the list, can be planned and judged.
LIST_LETTERS = "ABCDEFGHIJ"
HELD_OUT = {
    "A": ("mixer_operating", "table_rearrangement", "sample_loading"),
    "B": ("water_pouring", "items_handover", "handle_basket"),
}


# G4 round 2 (hints): each line also says what the scene looks like when the skill is finished, so that a skill held
# out of training can be judged finished from its description (round 1: held-out split A missed 336/470 ends).
DONE_HINTS = {
    "click_bell": ("the bell is pressed and the hand lifts off it", "铃被按下、手抬起离开"),
    "drawer_open_place": ("the tomato is inside the open drawer", "番茄已在打开的抽屉里"),
    "handle_basket": ("the bottle is in the basket and the basket is lifted", "瓶子在篮子里且篮子被提起"),
    "item_assembly": ("the two tubes are joined into one", "两根离心管已接成一根"),
    "items_handover": ("the pen stands in the pen holder", "笔已插在笔筒里"),
    "manipulate_pipette": ("the pipette was pressed and released over the other beaker", "移液枪在另一个烧杯上方按下又松开"),
    "mixer_operating": ("the beaker is on the mixer and its left button is pressed", "烧杯在搅拌器上且左边按钮已按下"),
    "sample_loading": ("the test tube stands in the rack", "试管已插在试管架上"),
    "table_rearrangement": ("the fork and the spoon lie beside the plate", "叉子和勺子已摆在盘子两边"),
    "water_pouring": ("the cup has water and the bottle and the cup are put back", "杯里有水，瓶子和杯子都放回原处"),
}


def skill_description(task: str, rng: random.Random | None = None, *, hints: bool = False) -> str:
    """A skill's line in the list: English (Chinese) as an outside list would give it; in training (rng) one of a few
    wordings, so that no wording is learnt by heart. ``hints``: and what the scene looks like once it is finished."""
    en, zh, goals = SKILLS[task]
    if rng is None:
        text = f"{en}（{zh}）"
        return f"{text}; done when: {DONE_HINTS[task][0]}（{DONE_HINTS[task][1]}）" if hints else text
    text = rng.choice([en, zh, f"{en}（{zh}）", f"{zh}：{en}", rng.choice(goals)])
    if not hints:
        return text
    hen, hzh = DONE_HINTS[task]
    return text + rng.choice([f"; done when: {hen}", f"；完成时：{hzh}", f"; done when: {hen}（{hzh}）", f"（做完：{hzh}）"])


class Listing:
    """An ask's skill list: entries (letter, task, description) in the order shown."""

    def __init__(self, entries):
        self.entries = [tuple(e) for e in entries]
        self._task = {letter: task for letter, task, _d in self.entries}
        self._letter = {task: letter for letter, task, _d in self.entries}

    @classmethod
    def build(cls, chain, pool, rng: random.Random, *, vary: bool = True, size: int | None = None,
              hints: bool = False) -> "Listing":
        """The chain's skills and distractors from ``pool``, shuffled, under letters drawn from LIST_LABELS."""
        tasks = list(dict.fromkeys(chain))
        others = [t for t in pool if t not in tasks]
        rng.shuffle(others)
        n = size if size is not None else rng.randint(len(tasks), max(len(tasks), min(len(LIST_LETTERS), len(tasks) + len(others))))
        tasks += others[: max(0, n - len(tasks))]
        rng.shuffle(tasks)
        letters = rng.sample(LIST_LETTERS, len(tasks))
        return cls([(letter, task, skill_description(task, rng if vary else None, hints=hints))
                    for letter, task in zip(letters, tasks)])

    def letter(self, task):
        return None if task is None else self._letter[task]

    def task(self, letter):
        return self._task.get(letter)

    def lines(self) -> str:
        return "\n".join(f"- {letter}: {description}" for letter, _t, description in self.entries)

    def to_json(self) -> list:
        return [list(e) for e in self.entries]


def list_prompt_text(goal: str, running: str | None, memory: str, lines: str, state: str | None = None, *,
                     past=PAST_S) -> str:
    """G4's prompt (a memory planner's, progress always): the skills are the list's letters."""
    hands = "" if state is None else f"Hands (x y z in cm, each in its own arm's frame; gripper open %): {state}\n"
    return (
        "You are the high-level planner of a dual-arm robot.\n"
        f"Goal: {goal}\n"
        f"Now running: {running or 'nothing yet'}\n"
        f"Memory: {memory}\n"
        f"{hands}"
        "Images: the current top, left-wrist and right-wrist cameras, then the top camera "
        + ", ".join(str(s) for s in past)
        + " seconds ago.\n"
        f"Skills you can use (a letter each, only these):\n{lines}\n"
        "Split the goal into these skills, in its order; a skill may come more than once. The memory lists the "
        f"finished skills by letter in order and how far the running skill is (stage 0 to {STAGES - 1} of {STAGES}): "
        "a skill can end where it started, so trust the memory for what happened before the images.\n"
        "Which skill should run now? Keep the running skill until its job is finished; then pick the next skill the "
        "goal needs after the finished ones, or done if the whole goal is finished. Write the memory as it is now.\n"
        "Answer in two lines:\nskill: <a letter from the list, or done>\nmemory: <the memory now>"
    )


def parse_list_reply(text: str, listing: "Listing"):
    """(task or DONE or None, memory) of a G4 reply."""
    skill = re.search(r"skill\s*:\s*([A-Za-z]+)", text)
    memory = re.search(r"memory\s*:\s*(.+)", text)
    word = skill.group(1).strip() if skill else ""
    task = DONE if word.lower() == DONE else listing.task(word.upper()) if len(word) == 1 else None
    return task, (memory.group(1).strip() if memory else None)


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


def version(*, state: bool = False, progress: bool = False, past=PAST_S, listing: bool = False,
            steps: bool = False) -> str:
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
    if listing:  # G4: the list prompt, its memory in letters, the letters
        rendered["listing"] = [list_prompt_text("<goal>", "<running>", "<memory>", "<lines>", past=past),
                               memory_text(["click_bell"], 2, namer=lambda t: "C"), LIST_LETTERS]
    if steps:  # G5: split the goal once (plan prompt), then the steps of the plan
        rendered["steps"] = [plan_prompt_text("<goal>"), plan_target_text(["click_bell", "click_bell"]),
                             step_prompt_text(["click_bell", "water_pouring"], 1, step_memory(0, 1), past=past),
                             step_prompt_text(["click_bell", "water_pouring"], None, step_memory(1, 0), past=past),
                             step_target_text(2, step_memory(1, 0))]
    return hashlib.sha1(json.dumps(rendered, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]



# ----- G5 (11-10, the user: "先拆再执行你不能放到一个模型里面吗" / "要能实现按三次铃铛"): one model, two questions ------
# 1. plan: the goal sentence -> its skills in order, once at the start (text only). Repeats in a row are fine
#    (按三次铃 -> 按铃；按铃；按铃), any wording the planner learnt to read (named, purpose words, counts).
# 2. step: the plan, the running step and a memory of steps -> keep the step, or the next one, or done. The goal sentence
#    is not in this question at all, so how it was worded cannot change when a skill is judged finished; and a switch is
#    the step number going up, so the same skill twice in a row is just two steps.
COUNT_WORDS = {2: ["两", "2"], 3: ["三", "3"], 4: ["四", "4"], 5: ["五", "5"]}
COUNT_EN = {2: ["twice", "2 times"], 3: ["three times", "3 times"], 4: ["four times", "4 times"], 5: ["five times", "5 times"]}
COUNT_TEMPLATES = {  # {p}: the skill's phrase, {n}: the count; "eval" ones are held out of training
    "train": ["{p}{n}次", "{p}做{n}遍", "连续{p}{n}次", "重复{n}次{p}", "{p}{n}遍"],
    "eval": ["{p}共{n}次", "来{n}遍{p}", "{p}一共{n}回"],
}
BELL_COUNT = {"train": ["按{n}次铃", "按{n}下铃", "把铃按{n}下"], "eval": ["连按{n}下铃", "铃按{n}次"]}
COUNT_EN_TEMPLATES = {"train": ["{p} {n}"], "eval": ["{p}, {n} in a row"]}


def repeat_chain(rng: random.Random, max_len: int = MAX_CHAIN, repeat: float = 0.3, lengths=None) -> list[str]:
    """A random plan of 1..max_len skills; each next skill is the one before with ``repeat`` (repeats in a row)."""
    length = rng.choice(lengths or [1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10])
    chain = [rng.choice(list(SKILLS))]
    while len(chain) < min(length, max_len):
        chain.append(chain[-1] if rng.random() < repeat else rng.choice([t for t in SKILLS if t != chain[-1]]))
    return chain


def runs_of(chain: list[str]) -> list[tuple[str, int]]:
    """[(skill, how many times in a row)]."""
    out: list[tuple[str, int]] = []
    for task in chain:
        if out and out[-1][0] == task:
            out[-1] = (task, out[-1][1] + 1)
        else:
            out.append((task, 1))
    return out


def join_zh(parts: list[str], rng: random.Random) -> str:
    """Chinese parts as one goal: 先..然后..最后, 请依次.., 第一..第二.., a list, ..之后.."""
    if len(parts) == 1:
        return rng.choice(["{0}。", "请{0}。", "帮我{0}", "能不能{0}？", "麻烦{0}。"]).format(parts[0])
    style = rng.randrange(5)
    if style == 0:
        middle = "".join(f"，{rng.choice(['然后', '再', '接着'])}{p}" for p in parts[1:-1])
        return f"先{parts[0]}{middle}，最后{parts[-1]}。"
    if style == 1:
        return "请依次" + "、".join(parts) + "。"
    if style == 2:
        nums = "一二三四五六七八九十"
        return "；".join(f"第{nums[i]}，{p}" for i, p in enumerate(parts)) + "。"
    if style == 3:
        return "帮我" + "，".join(parts[:-1]) + f"，完了{parts[-1]}"
    return "，".join(parts[:-1]) + f"，之后{parts[-1]}。"


def step_goal_text(chain: list[str], rng: random.Random, split: str = "train", purpose: float = 0.4,
                   counted: float = 0.75) -> str:
    """A goal for G5: skills in a row written with a count (``counted`` of the time, else one by one), each skill named
    or in purpose words (``purpose``), Chinese mostly; ``split`` "eval": purpose words and count templates never in
    training. English (a fifth of the goals) names the skills."""
    groups = runs_of(chain)
    if rng.random() < 0.2:
        parts = []
        for task, n in groups:
            en = SKILLS[task][0]
            if n > 1 and rng.random() < counted:
                parts.append(rng.choice(COUNT_EN_TEMPLATES[split]).format(p=en, n=rng.choice(COUNT_EN[min(n, 5)])))
            else:
                parts += [en] * n
        if len(parts) == 1:
            return f"Please {parts[0]}."
        return "First " + ", then ".join(parts[:-1]) + f", and finally {parts[-1]}."
    parts = []
    for task, n in groups:
        def phrase():
            return rng.choice(ABSTRACT[task][split]) if rng.random() < purpose else rng.choice(SKILLS[task][2])
        if n > 1 and rng.random() < counted:
            word = rng.choice(COUNT_WORDS[min(n, 5)])
            if task == "click_bell" and rng.random() < 0.5:
                parts.append(rng.choice(BELL_COUNT[split]).format(n=word))
            else:
                parts.append(rng.choice(COUNT_TEMPLATES[split]).format(p=phrase(), n=word))
        else:
            parts += [phrase() for _ in range(n)]
    return join_zh(parts, rng)


ZH_TO_TASK = {zh: task for task, (_en, zh, _goals) in SKILLS.items()}


def plan_prompt_text(goal: str) -> str:
    """G5's first question (text only): split the goal into the skills."""
    lines = "\n".join(f"- {zh}: {en}" for _t, (en, zh, _g) in SKILLS.items())
    return (
        "You are the high-level planner of a dual-arm robot.\n"
        f"Goal: {goal}\n"
        f"Skills (by their Chinese names):\n{lines}\n"
        "Split the goal into these skills, in its order. A skill may come more than once, also several times in a row "
        "(a goal can say how many times).\n"
        "Answer in one line:\nplan: <the skills' Chinese names separated by ；>"
    )


def plan_target_text(chain: list[str]) -> str:
    return "plan: " + "；".join(SKILLS[t][1] for t in chain)


def parse_plan(text: str) -> list[str] | None:
    """The skills of a plan reply, or None (no plan line, or a name that is no skill)."""
    found = re.search(r"plan\s*:\s*(.+)", text)
    if not found:
        return None
    names = [n.strip().strip("。") for n in re.split(r"[；;]", found.group(1)) if n.strip().strip("。")]
    tasks = [ZH_TO_TASK.get(n) for n in names]
    return tasks if tasks and all(tasks) else None


def step_memory(finished: int, progress: int) -> str:
    """G5's memory: how many steps are finished and how far the running one is."""
    return f"已完成：{finished} 步。当前进度：{progress}/{STAGES}。"


def parse_step_memory(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    done = re.search(r"已完成：\s*(\d+)\s*步", text)
    stage = re.search(r"当前进度：(\d+)\s*/\s*\d+", text)
    if not done or not stage:
        return None
    return int(done.group(1)), min(int(stage.group(1)), STAGES)


def step_prompt_text(plan: list[str], running: int | None, memory: str, state: str | None = None, *,
                     past=PAST_S) -> str:
    """G5's second question: the plan (numbered), the running step (1-based) or None between steps, the memory."""
    lines = "\n".join(f"{i}. {SKILLS[t][1]}（{SKILLS[t][0]}）" for i, t in enumerate(plan, 1))
    now = f"step {running}（{SKILLS[plan[running - 1]][1]}）" if running else "nothing (between two steps)"
    hands = "" if state is None else f"Hands (x y z in cm, each in its own arm's frame; gripper open %): {state}\n"
    return (
        "You are the high-level planner of a dual-arm robot.\n"
        f"Plan (the goal split into skills, in order):\n{lines}\n"
        f"Now running: {now}\n"
        f"Memory: {memory}\n"
        f"{hands}"
        "Images: the current top, left-wrist and right-wrist cameras, then the top camera "
        + ", ".join(str(s) for s in past)
        + " seconds ago.\n"
        f"The memory counts the finished steps and how far the running step is (stage 0 to {STAGES - 1} of {STAGES}): "
        "a step may do the same skill as the one before, and a skill can end where it started, so trust the memory "
        "for what happened before the images.\n"
        "Which step should run now? Keep the running step until its job is finished; then go on to the next step, or "
        "done after the last one. Write the memory as it is now.\n"
        "Answer in two lines:\nstep: <a step number, or done>\nmemory: <the memory now>"
    )


def step_target_text(step, memory: str) -> str:
    return f"step: {step}\nmemory: {memory}"


def parse_step_reply(text: str, n_steps: int) -> tuple[int | str | None, str | None]:
    """(step number 1..n_steps or DONE or None, memory)."""
    step = re.search(r"step\s*:\s*([A-Za-z0-9]+)", text)
    memory = re.search(r"memory\s*:\s*(.+)", text)
    word = step.group(1).strip().lower() if step else ""
    value = DONE if word == DONE else int(word) if word.isdigit() and 1 <= int(word) <= n_steps else None
    return value, (memory.group(1).strip() if memory else None)


class StepTracker(MemoryTracker):
    """MemoryTracker for G5: the finished steps are counted (step_memory), the progress voted as before."""

    def text(self) -> str:
        return step_memory(len(self.finished), self.progress)

    def observe(self, reply_memory: str | None) -> None:
        parsed = parse_step_memory(reply_memory)
        if parsed is None:
            return
        n_finished, stage = parsed
        self.votes = [*self.votes, STAGES if n_finished > len(self.finished) else stage][-3:]
        if len(self.votes) >= 2:
            self.progress = max(self.progress, sorted(self.votes)[(len(self.votes) - 1) // 2])


def step_sample(sample: dict, rng: random.Random, *, repeats: int = 0) -> dict | None:
    """A memory planner sample (samples_mem2) as a G5 step sample; ``repeats`` times one of its skills doubled in the
    plan (a repeat in a row: before or after the running one, at the start of the next, or elsewhere), so that the step
    question also learns that the same skill can follow itself. None if the memories do not parse."""
    chain = list(sample["chain"])
    now, target = parse_memory(sample["memory"]), parse_memory(sample["target_memory"])
    if now is None or target is None:
        return None
    (f, stage), (ft, t_stage) = now, target
    running = sample["running"] is not None
    done = sample["target_skill"] == DONE
    if running and (f >= len(chain) or chain[f] != sample["running"]):
        return None
    if not done and (ft >= len(chain) or chain[ft] != sample["target_skill"]):
        return None
    for _ in range(repeats):
        if len(chain) >= MAX_CHAIN:
            break
        ops = ["late"] if max(ft, f) + 1 < len(chain) else []
        if running:
            ops += ["after", "after", "before"]
        elif f >= 1:
            ops += ["start", "start"]
        if f >= 2:
            ops.append("early")
        if not ops:
            break
        op = rng.choice(ops)
        if op == "after":  # the running skill once more right after it: its end is a switch to the same skill
            chain.insert(f + 1, chain[f])
            if done:
                done, ft = False, f + 1
        elif op == "before":  # the running one is the second of two in a row
            chain.insert(f, chain[f])
            f, ft = f + 1, ft + 1
        elif op == "start":  # between steps: the skill just finished comes again next
            chain.insert(f, chain[f - 1])
            if done:
                done = False
            ft = f
        elif op == "early":  # an earlier finished skill doubled
            q = rng.randrange(f - 1)
            chain.insert(q + 1, chain[q])
            f, ft = f + 1, ft + 1
        else:  # a later skill doubled (after the target step)
            lo = max(ft, f) + 1
            if lo >= len(chain):
                continue
            q = rng.randrange(lo, len(chain))
            chain.insert(q + 1, chain[q])
    return {
        "kind": "step", "episode": sample["episode"], "frame": sample["frame"], "images": sample["images"],
        "plan": chain, "running": f + 1 if running else None, "memory": step_memory(f, stage),
        "target_step": DONE if done else ft + 1, "target_memory": step_memory(ft, t_stage),
        "repeats": repeats,
    }


def plan_sample(rng: random.Random, images=(), split: str = "train") -> dict:
    """A G5 plan sample (text only): a random plan (repeats in a row too) and its goal in any wording."""
    chain = repeat_chain(rng)
    if rng.random() < 0.25 and len(set(chain)) == len(runs_of(chain)) == len(chain):
        goal = any_goal_text(chain, rng)  # the G2 wordings, as before
    else:
        goal = step_goal_text(chain, rng, split=split)
    return {"kind": "plan", "goal": goal, "plan": chain, "images": list(images), "episode": None, "frame": 0}


def g5_prompt(sample: dict, state: str | None = None, past=PAST_S) -> str:
    if sample["kind"] == "plan":
        return plan_prompt_text(sample["goal"])
    return step_prompt_text(sample["plan"], sample["running"], sample["memory"], state, past=past)


def g5_target(sample: dict) -> str:
    if sample["kind"] == "plan":
        return plan_target_text(sample["plan"])
    return step_target_text(sample["target_step"], sample["target_memory"])
