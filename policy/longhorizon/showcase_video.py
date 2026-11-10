"""A long-horizon episode as a video to show (11-10): the cameras, the sub-task running now on top, the planner on the right.

The bench's episode video (eval_result_real/long_horizon/<run>/episode_NNN*.mp4: the three cameras side by side, one
frame per control step) with, around it:
- on top, the sub-task running now (第 k / N 步), the goal sentence and the plan with ✓ / ▶;
- on the right, the planner's log: its plan, each reply (still on step k / step k done), each switch;
- at the bottom, a one-line note on what is shown.

The timeline comes from results.json (long_horizon.subtasks: start / end step, plans with the step each reply was read
at); a frame is a control step, so frame i is step i.

    python policy/longhorizon/showcase_video.py --run eval_result_real/long_horizon/<run> --episode 0 \
        --out demo/按三次铃.mp4
"""

# ruff: noqa: RUF001, RUF002, RUF003  (the overlay text is Chinese)
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

W, H = 1920, 1080
SRC_W, SRC_H = 1920, 480  # cam_high | cam_right_wrist | cam_left_wrist, 640 x 480 each (run_real CAMS)
FPS = 10  # one frame per control step at 10 Hz

SKILL_ZH = {
    "click_bell": "按铃",
    "drawer_open_place": "开抽屉放番茄",
    "handle_basket": "瓶子装篮提起",
    "item_assembly": "对接离心管",
    "items_handover": "把笔放进笔筒",
    "manipulate_pipette": "移液枪移液",
    "mixer_operating": "开搅拌器",
    "sample_loading": "把试管插进试管架",
    "table_rearrangement": "摆好叉子勺子",
    "water_pouring": "倒水",
}

BG = (16, 18, 24)
PANEL = (28, 32, 42)
TEXT = (232, 236, 244)
MUTED = (150, 160, 178)
ACCENT = (255, 196, 64)
DONE = (96, 210, 140)
HOME = (120, 170, 255)
LINE = (52, 58, 72)

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
SC = 2  # the Simplified Chinese face in the Noto CJK collections


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size, index=SC)


def mmss(frame: int) -> str:
    seconds = frame // FPS
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> list[str]:
    lines, line = [], ""
    for ch in text:
        if draw.textlength(line + ch, font=fnt) > width and line:
            lines.append(line)
            line = ch.lstrip()
        else:
            line += ch
    if line:
        lines.append(line)
    return lines


def parse_memory(memory: str | None) -> tuple[int, int] | None:
    found = re.search(r"已完成：\s*(\d+)\s*步。当前进度：(\d+)\s*/\s*\d+", memory or "")
    return (int(found.group(1)), int(found.group(2))) if found else None


def timeline(episode: dict) -> tuple[list[str], list[dict], list[tuple[int, str, str]], bool]:
    """The plan (task names; for a planner that does not split ahead, G2, the skills it went on to pick), the
    subtasks, the log events (frame, kind, text) in order, and whether the planner split the goal ahead (G5)."""
    lh = episode["long_horizon"]
    subs = lh["subtasks"]
    split = bool(subs[0].get("plan"))
    plan = list(subs[0].get("plan") or [s["task"] for s in subs])
    first = f"拆好计划，共 {len(plan)} 步" if split else "每做完一步，看画面再选下一个技能"
    events: list[tuple[int, str, str]] = [(0, "plan", first)]
    for k, sub in enumerate(subs):
        name = SKILL_ZH.get(sub["task"], sub["task"])
        events.append((sub["start_step"], "start", f"开始第 {k + 1} 步：{name}"))
        votes = 0
        for reply in sub.get("plans", []):
            if reply.get("running") is None:
                continue
            step = reply.get("plan_step")
            parsed = parse_memory(reply.get("memory"))
            moved = step == "done" or (isinstance(step, int) and step > k + 1)
            if not split:  # G2: a reply naming another skill (or done) is a vote to switch
                moved = reply.get("skill") not in (None, sub["task"])
            if moved:
                votes += 1
                events.append((reply["step"], "vote", f"第 {k + 1} 步做完了（第 {votes} 票）"))
            else:
                votes = 0
                stage = f"，进度 {parsed[1]}/4" if parsed else ""
                events.append((reply["step"], "keep", f"第 {k + 1} 步还在做{stage}"))
        end = sub.get("end_step")
        if end is not None:
            nxt = sub.get("next")
            if nxt == "done":
                events.append((end, "finish", "全部完成"))
            else:
                events.append((end, "switch", f"{name} 完成 → 回起始姿态"))
        elif episode.get("success"):  # the operator saw the last step done before the planner said so
            events.append((int(episode["steps"]) - 1, "finish", "操作员确认：最后一步做完"))
    events.sort(key=lambda e: e[0])
    return plan, subs, events, split


def state_at(frame: int, subs: list[dict]) -> tuple[str, int | None]:
    """("run", k) / ("home", k: the one just finished) / ("finished", None) at a frame."""
    for k, sub in enumerate(subs):
        end = sub.get("end_step")
        if sub["start_step"] <= frame and (end is None or frame < end):
            return "run", k
        nxt = subs[k + 1]["start_step"] if k + 1 < len(subs) else None
        if end is not None and frame >= end and (nxt is None or frame < nxt):
            return ("finished", None) if sub.get("next") == "done" else ("home", k)
    return "run", 0


class Renderer:
    def __init__(self, episode: dict, title: str | None):
        self.plan, self.subs, self.events, self.split = timeline(episode)
        self.goal = episode["long_horizon"].get("goal") or ""
        self.title = title
        self.f_big = font(54, bold=True)
        self.f_mid = font(32, bold=True)
        self.f_text = font(28)
        self.f_small = font(24)
        self.f_smallb = font(24, bold=True)
        self.f_log = font(23)
        self.f_logb = font(23, bold=True)
        self.f_label = font(22)

    def items(self, frame: int, mode: str) -> list[str]:
        """The steps to show: the whole plan (G5), or the skills picked so far (G2 picks the next one only then)."""
        if self.split or mode == "finished":
            return self.plan
        return [sub["task"] for sub in self.subs if sub["start_step"] <= frame]

    def chips(self, draw, x: int, y: int, mode: str, k: int | None, frame: int) -> None:
        done = len(self.plan) if mode == "finished" else (k + 1 if mode == "home" else k)
        for i, task in enumerate(self.items(frame, mode)):
            label = f"{i + 1} {SKILL_ZH.get(task, task)}"
            if i < done:
                color, fill, label = DONE, (30, 60, 44), "✓ " + label
            elif mode == "run" and i == k:
                color, fill, label = (20, 20, 24), ACCENT, "▶ " + label
            else:
                color, fill = MUTED, PANEL
            width = int(draw.textlength(label, font=self.f_small)) + 28
            if x + width > W - 24:
                break
            draw.rounded_rectangle((x, y, x + width, y + 40), radius=20, fill=fill)
            draw.text((x + 14, y + 5), label, font=self.f_small, fill=color)
            x += width + 10

    def frame(self, src: np.ndarray, frame: int, *, hold: bool = False) -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        mode, k = ("finished", None) if hold else state_at(frame, self.subs)
        # top: the sub-task running now
        n = len(self.plan)
        total = f" / {n}" if self.split else ""
        if mode == "run":
            now, color = f"正在执行：{SKILL_ZH.get(self.subs[k]['task'], self.subs[k]['task'])}", ACCENT
            counter = f"第 {k + 1}{total} 步"
        elif mode == "home":
            now, color = "回起始姿态，准备下一步", HOME
            counter = f"已完成 {k + 1}{total} 步"
        else:
            now, color, counter = "全部完成", DONE, f"{n} / {n} 步"
        draw.text((32, 18), now, font=self.f_big, fill=color)
        cw = draw.textlength(counter, font=self.f_mid)
        draw.text((W - 32 - cw, 30), counter, font=self.f_mid, fill=TEXT)
        goal = f"目标（一句话）：{self.goal}"
        draw.text((32, 96), goal if draw.textlength(goal, font=self.f_text) < W - 64 else goal[:60] + "…",
                  font=self.f_text, fill=TEXT)  # fmt: skip
        self.chips(draw, 32, 142, mode, k, frame)
        draw.line((24, 196, W - 24, 196), fill=LINE, width=2)
        # cameras: top camera large, the two wrists beside it
        top = Image.fromarray(src[:, 0:640]).resize((1040, 780), Image.BILINEAR)
        img.paste(top, (24, 214))
        draw.text((36, 222), "顶部相机", font=self.f_label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        for i, (x0, name) in enumerate(((640, "右手腕相机"), (1280, "左手腕相机"))):
            cam = Image.fromarray(src[:, x0 : x0 + 640]).resize((360, 270), Image.BILINEAR)
            y = 214 + i * 286
            img.paste(cam, (1080, y))
            draw.text(
                (1090, y + 6), name, font=self.f_label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0)
            )
        # how it works, under the wrists
        box = (1080, 800, 1440, 994)
        draw.rounded_rectangle(box, radius=12, fill=PANEL)
        how = "拆计划，判断每一步做完没有" if self.split else "选下一个技能，判断做完没有"
        lines = ["高层调度器（视觉语言模型）", how, "底层世界模型", "看画面出动作，一步一步做"]
        for i, text in enumerate(lines):
            draw.text((1096, 812 + i * 45), text, font=self.f_smallb if i % 2 == 0 else self.f_small,
                      fill=TEXT if i % 2 == 0 else MUTED)  # fmt: skip
        # right: the planner's split (11-10, user: "调度器拆分也应该写到右边吧"), then its log
        x0, y0, x1, y1 = 1460, 214, W - 24, 994
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=PANEL)
        draw.text((x0 + 18, y0 + 10), "调度器拆分" if self.split else "调度器选技能", font=self.f_mid, fill=TEXT)
        y = y0 + 58
        goal_lines = wrap(draw, f"「{self.goal}」", self.f_small, x1 - x0 - 36)
        if len(goal_lines) > 3:
            goal_lines = [*goal_lines[:2], goal_lines[2][:-1] + "…"]
        for text in goal_lines:
            draw.text((x0 + 18, y), text, font=self.f_small, fill=MUTED)
            y += 32
        draw.text((x0 + 18, y), "↓ 拆成" if self.split else "↓ 一步一步选", font=self.f_small, fill=MUTED)
        y += 36
        done = len(self.plan) if mode == "finished" else (k + 1 if mode == "home" else k)
        for i, task in enumerate(self.items(frame, mode)):
            if i < done:
                mark, color, fnt = "✓", DONE, self.f_log
            elif mode == "run" and i == k:
                mark, color, fnt = "▶", ACCENT, self.f_logb
            else:
                mark, color, fnt = "·", MUTED, self.f_log
            draw.text((x0 + 18, y), f"{mark} {i + 1}. {SKILL_ZH.get(task, task)}", font=fnt, fill=color)
            y += 36
        y += 8
        draw.line((x0 + 18, y, x1 - 18, y), fill=LINE, width=2)
        draw.text((x0 + 18, y + 10), "调度器日志", font=self.f_mid, fill=TEXT)
        log_top = y + 58
        shown: list[tuple[int, str, str]] = []
        for at, kind, text in self.events:
            if at > frame and not hold:
                break
            if kind == "keep" and shown and shown[-1][1] == "keep":
                shown[-1] = (at, kind, text)  # "still on step k": one line, kept up to date
            else:
                shown.append((at, kind, text))
        rows: list[tuple[str, tuple, bool]] = []
        for at, kind, text in shown:
            color = {"plan": TEXT, "start": ACCENT, "vote": DONE, "keep": MUTED, "switch": HOME, "finish": DONE}[kind]
            body = wrap(draw, text, self.f_logb, x1 - x0 - 36 - 70)
            rows.append((f"{mmss(at)}  {body[0]}", color, kind in ("start", "switch", "finish", "plan")))
            rows += [(" " * 7 + more, color, False) for more in body[1:]]
        line_h = 32
        fit = max(1, (y1 - 12 - log_top) // line_h)
        for i, (text, color, bold) in enumerate(rows[-fit:]):
            draw.text((x0 + 18, log_top + i * line_h), text, font=self.f_logb if bold else self.f_log, fill=color)
        # bottom note
        note = "真机，两只机械臂 · 每一步做完回起始姿态再做下一步 · 视频按控制步 10 帧/秒播放（不含模型推理停顿）"
        draw.text((32, 1014), note, font=self.f_small, fill=MUTED)
        if self.title:
            tw = draw.textlength(self.title, font=self.f_small)
            draw.text((W - 32 - tw, 1014), self.title, font=self.f_small, fill=MUTED)
        return img


def render(run: pathlib.Path, episode_index: int, video: pathlib.Path | None, out: pathlib.Path, title: str | None,
           hold_s: float = 2.0) -> None:  # fmt: skip
    episode = json.loads((run / "results.json").read_text())["episodes"][episode_index]
    steps = int(episode["steps"])
    if video is None:  # the episode's video is the one with as many frames as it had steps (retries add _2, _3)
        for candidate in sorted(run.glob(f"episode_{episode_index:03d}*.mp4")):
            count = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                                    "stream=nb_read_packets", "-of", "csv=p=0", str(candidate)],
                                   capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
            if count == str(steps):
                video = candidate
        if video is None:
            raise SystemExit(f"no episode_{episode_index:03d}*.mp4 in {run} with {steps} frames")
    renderer = Renderer(episode, title)
    out.parent.mkdir(parents=True, exist_ok=True)
    decode = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                              stdout=subprocess.PIPE)  # fmt: skip
    encode = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", str(FPS), "-i", "-", "-vf", "fps=30", "-c:v", "libx264", "-preset", "medium",
                               "-crf", "18", "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                              stdin=subprocess.PIPE)  # fmt: skip
    size = SRC_W * SRC_H * 3
    frame, last = 0, None
    while True:
        raw = decode.stdout.read(size)
        if len(raw) < size:
            break
        last = np.frombuffer(raw, np.uint8).reshape(SRC_H, SRC_W, 3)
        encode.stdin.write(renderer.frame(last, frame).tobytes())
        frame += 1
    if last is not None:
        held = renderer.frame(last, frame, hold=True).tobytes()
        for _ in range(int(hold_s * FPS)):
            encode.stdin.write(held)
    encode.stdin.close()
    encode.wait()
    decode.wait()
    print(f"{out}: {frame} frames from {video.name} (+{hold_s:g} s held)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="eval_result_real/long_horizon/<run> (results.json and videos)")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--video", default=None, help="default: the run's episode video with as many frames as steps")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None, help="small text at the bottom right")
    args = parser.parse_args()
    render(pathlib.Path(args.run), args.episode, pathlib.Path(args.video) if args.video else None,
           pathlib.Path(args.out), args.title)  # fmt: skip


if __name__ == "__main__":
    main()
