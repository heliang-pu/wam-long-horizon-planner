"""The high-level planner model under its own interpreter (venvs/planner: transformers >= 5.2, peft): loading
RynnBrain1.1-2B with or without a train_planner.py adapter, and one planning step. planner_worker.py (the robot),
eval_planner.py and train_planner.py all go through here, so they build the input the same way.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.longhorizon import planner_prompt as pp  # noqa: E402

BASE = Path.home() / "workspace/models/internw0/RynnBrain1.1-2B"


def prompt_version(*, state: bool = False, progress: bool = False, past=pp.PAST_S) -> str:
    """planner_prompt.version(): a planner is asked with the prompt it was trained on."""
    return pp.version(state=state, progress=progress, past=past)


def planner_image(frame: np.ndarray):
    """An HWC uint8 RGB camera frame as the planner was trained on it: 320x240, through JPEG at quality 90."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame)).resize(pp.IMAGE_SIZE, Image.BILINEAR).save(buffer, "JPEG", quality=90)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def encode_prompt(
    processor,
    images,
    goal: str,
    running: str | None,
    memory: str,
    state: str | None = None,
    *,
    progress: bool = False,
    past=pp.PAST_S,
) -> dict:
    """The planner's input: the six images, then planner_prompt.prompt_text, through the chat template with the
    generation prompt (no thinking)."""
    content = [*({"type": "image", "image": image} for image in images)]
    text = pp.prompt_text(goal, running, memory, state=state, progress=progress, past=past)
    content.append({"type": "text", "text": text})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


class Planner:
    """RynnBrain1.1-2B, zero-shot (adapter None or "base") or with a train_planner.py adapter merged in."""

    def __init__(self, adapter: str | None = None, base: str | Path = BASE, device: str = "cuda:0"):
        from transformers import AutoModelForImageTextToText
        from transformers import AutoProcessor

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(base)
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
        self.adapter, self.meta, self.uses_state, self.uses_progress = None, {}, False, False
        self.past = pp.PAST_S
        if adapter not in (None, "", "base"):
            from peft import PeftModel

            self.adapter = Path(adapter).expanduser()
            self.meta = json.loads((self.adapter / "planner.json").read_text())
            self.uses_state = bool(self.meta.get("state"))
            self.uses_progress = bool(self.meta.get("progress"))
            self.past = tuple(self.meta.get("past_s") or pp.PAST_S)
            expected = prompt_version(state=self.uses_state, progress=self.uses_progress, past=self.past)
            if self.meta.get("prompt_version") != expected:
                raise ValueError(
                    f"{self.adapter} was trained on prompt {self.meta.get('prompt_version')}, planner_prompt.py is now "
                    f"{expected}: train it again (or check out that prompt)"
                )
            model = PeftModel.from_pretrained(model, str(self.adapter)).merge_and_unload()
        self.model = model.to(self.device).eval()

    def info(self) -> dict:
        return {"adapter": str(self.adapter) if self.adapter else "base",
                "prompt_version": prompt_version(state=self.uses_state, progress=self.uses_progress, past=self.past),
                "step": self.meta.get("step"), "state": self.uses_state, "progress": self.uses_progress,
                "past_s": list(self.past)}  # fmt: skip

    @torch.no_grad()
    def plan(
        self,
        images,
        goal: str,
        running: str | None,
        memory: str,
        skills=tuple(pp.SKILLS),
        max_new_tokens=96,
        state=None,
    ):
        """images: the six planner images (PIL, planner_image) -> (skill or None, memory or None, reply, seconds).
        state: planner_prompt.hands_text, for a planner trained with the hands (unknown if not given)."""
        result = self.plan_scored(images, goal, running, memory, skills, max_new_tokens, state)
        return result["skill"], result["memory"], result["reply"], result["seconds"]

    def skill_tokens(self, skills) -> tuple[int, dict[str, int]]:
        """(the number of tokens of "skill:", each skill's (and done's) first token after it): the first token tells
        the skills apart (checked for RynnBrain's tokenizer: all eleven differ)."""
        tok = self.processor.tokenizer
        prefix = tok("skill:", add_special_tokens=False)["input_ids"]
        first = {}
        for name in [*skills, pp.DONE]:
            ids = tok(f"skill: {name}", add_special_tokens=False)["input_ids"]
            if ids[: len(prefix)] != prefix or len(ids) <= len(prefix):
                raise ValueError(f"{name!r} does not tokenize after the prefix")
            first[name] = ids[len(prefix)]
        if len(set(first.values())) != len(first):
            raise ValueError("two skills share their first token: the switch probability is undefined")
        return len(prefix), first

    @torch.no_grad()
    def plan_scored(
        self,
        images,
        goal: str,
        running: str | None,
        memory: str,
        skills=tuple(pp.SKILLS),
        max_new_tokens=96,
        state=None,
    ) -> dict:
        """plan() and the probability of each skill's (and done's) first token where the reply names the skill
        (probs; None if the reply does not start with "skill:"): p of the running skill below a threshold is a
        switch whatever the greedy reply says, a calibrated alternative to replies in a row."""
        started = time.monotonic()
        state = (state or pp.STATE_UNKNOWN) if self.uses_state else None
        inputs = encode_prompt(
            self.processor, images, goal, running, memory, state, progress=self.uses_progress, past=self.past
        ).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True,
                                  return_dict_in_generate=True)  # fmt: skip
        generated = out.sequences[0, inputs["input_ids"].shape[1] :]
        reply = self.processor.decode(generated, skip_special_tokens=True).strip()
        skill, new_memory = pp.parse_reply(reply, skills)
        n_prefix, first = self.skill_tokens(skills)
        prefix = self.processor.tokenizer("skill:", add_special_tokens=False)["input_ids"]
        probs = None
        if generated[:n_prefix].tolist() == prefix and len(out.scores) > n_prefix:
            p = torch.softmax(out.scores[n_prefix][0].float(), -1)
            probs = {name: float(p[token]) for name, token in first.items()}
        return {"skill": skill, "memory": new_memory, "reply": reply, "seconds": time.monotonic() - started,
                "probs": probs}  # fmt: skip
