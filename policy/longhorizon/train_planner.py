"""LoRA fine-tuning of the high-level planner (RynnBrain1.1-2B) on build_planner_data.py samples.

Each sample is the prompt the robot asks (planner_prompt.prompt_text with its six images, through the model's chat
template with the generation prompt, exactly as planner_worker.py builds it) followed by the target reply; the loss is
on the reply only. LoRA on the language model's projections (the vision tower and everything else frozen).

    ../venvs/planner/bin/python policy/longhorizon/train_planner.py \
        --data ~/workspace/models/longhorizon/planner_data_<date> --out ~/workspace/models/longhorizon/planner_<date>

Writes <out>/adapter (peft), <out>/planner.json (base model, data, prompt version) and <out>/step_<n>/ on the way.
Several GPUs: start it with torchrun --nproc_per_node=<n>; --accumulate stays the samples per optimizer step in all (a
multiple of n), each GPU takes its share of them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.longhorizon import planner_prompt as pp  # noqa: E402
from policy.longhorizon import upstream_planner as up  # noqa: E402
from policy.longhorizon.planner_runtime import BASE  # noqa: E402
from policy.longhorizon.planner_runtime import encode_prompt  # noqa: E402
from policy.longhorizon.planner_runtime import prompt_version  # noqa: E402

LORA_TARGETS = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$"
)


class Samples(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Path,
        processor,
        samples: list[dict],
        states=None,
        state_dropout=0.0,
        seed=0,
        *,
        progress=False,
        lerobot=False,
        past=pp.PAST_S,
    ):
        self.data, self.processor, self.samples, self.progress = data, processor, samples, progress
        self.lerobot, self.past = lerobot, past
        self.states, self.state_dropout, self.seed = states, state_dropout, seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image

        s = self.samples[i]
        if self.lerobot:  # upstream_planner.py build: LeRobot's request and JSON reply
            inputs, reply = up.encode_lerobot(self.processor, self.data, s)
            reply += "<|im_end|>"
        else:
            images = [Image.open(self.data / path).convert("RGB") for path in s["images"]]
            state = None
            if self.states is not None and s.get("episode"):  # dropped as a whole: the planner reads the images too
                rng = random.Random(f"{self.seed}/{i}")
                rows = self.states.get(s["episode"])
                keep = rows is not None and rng.random() >= self.state_dropout
                state = pp.hands_text(rows, s["frame"], rng) if keep else pp.STATE_UNKNOWN
            if s.get("kind"):  # G5 (build_step_samples.py): a plan question (text only) or a step question
                inputs = encode_prompt(self.processor, images, "", None, "",
                                       text=pp.g5_prompt(s, state if s["kind"] == "step" else None, self.past))  # fmt: skip
                reply = pp.g5_target(s) + "<|im_end|>"
            else:
                inputs = encode_prompt(
                    self.processor, images, s["goal"], s["running"], s["memory"], state, progress=self.progress,
                    past=self.past, listing=s.get("listing"),
                )  # fmt: skip
                reply = pp.target_text(s["target_skill"], s["target_memory"]) + "<|im_end|>"
        target = self.processor.tokenizer(reply, add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = torch.cat([inputs["input_ids"], target], 1)
        labels = torch.full_like(ids, -100)
        labels[:, inputs["input_ids"].shape[1] :] = target
        batch = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
        batch.update(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
        if "mm_token_type_ids" in inputs:
            batch["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(target)], 1)
        return batch


PAD = {"attention_mask": 0, "labels": -100, "mm_token_type_ids": 0}  # input_ids: the tokenizer's pad id


def collate(items: list[dict], pad_id: int) -> dict:
    """Samples' items as one batch: token rows right-padded to the longest (masked, no loss), the images' patches
    (pixel_values) and grids (image_grid_thw) one after the other, in the order of the rows."""
    length = max(item["input_ids"].shape[1] for item in items)
    batch = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if key in ("input_ids", *PAD):
            fill = pad_id if key == "input_ids" else PAD[key]
            batch[key] = torch.cat(
                [torch.nn.functional.pad(v, (0, length - v.shape[1]), value=fill) for v in values], 0
            )
        else:
            batch[key] = torch.cat(values, 0)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--samples", default="samples.jsonl", help="under --data (add_hard_negatives.py: samples_hard.jsonl)"
    )
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=8, help="samples per optimizer step, all GPUs together")
    parser.add_argument("--batch", type=int, default=1, help="samples per forward pass on each GPU (padded together)")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1000, help="optimizer steps between saved adapters")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state", action="store_true", help="the hands in the prompt (<data>/states.json)")
    parser.add_argument("--state-dropout", type=float, default=0.5, help="share of samples with the hands unknown")
    parser.add_argument("--progress", action="store_true", help="a memory planner (build_mem_samples.py samples)")
    parser.add_argument("--listing", action="store_true", help="G4: samples with a skill list (build_list_samples.py)")
    parser.add_argument("--hints", action="store_true", help="G4: the list lines carry done hints (build_list_samples --hints)")
    parser.add_argument("--steps", action="store_true", help="G5: plan + step samples (build_step_samples.py)")
    parser.add_argument("--past", default="3,2,1", help="seconds before of the top-camera history images, as the "
                        "samples were built (build_mem_samples.py --past)")  # fmt: skip
    parser.add_argument("--format", choices=("ours", "lerobot"), default="ours", help="lerobot: LeRobot's planner "
                        "request and reply (upstream_planner.py build, --samples samples_lerobot.jsonl)")  # fmt: skip
    args = parser.parse_args()

    from peft import LoraConfig
    from peft import get_peft_model
    from transformers import AutoModelForImageTextToText
    from transformers import AutoProcessor

    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if args.accumulate % (world * args.batch):
        raise ValueError(f"--accumulate {args.accumulate} is not a multiple of {world} GPUs x --batch {args.batch}")
    device = f"cuda:{os.environ.get('LOCAL_RANK', '0')}"
    if world > 1:
        torch.distributed.init_process_group("nccl")

    torch.manual_seed(args.seed)
    data, out = Path(args.data).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in (data / args.samples).read_text().splitlines() if line.strip()]
    if args.max_samples:
        samples = samples[: args.max_samples]
    processor = AutoProcessor.from_pretrained(args.base)
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS))  # fmt: skip
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()  # LoRA weights in float32, the frozen model in bfloat16
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    total = max(1, math.ceil(len(samples) * args.epochs / args.accumulate))
    warmup = max(1, total // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    )
    past = tuple(int(v) for v in args.past.split(","))
    meta = {
        "base": str(Path(args.base).resolve()),
        "data": str(data.resolve()),
        "prompt_version": up.version()
        if args.format == "lerobot"
        else prompt_version(state=args.state, progress=args.progress, past=past, listing=args.listing, steps=args.steps),
        "format": args.format,
        "list": args.listing,
        "hints": args.hints,
        "steps": args.steps,
        "past_s": list(past),
        "state": args.state,
        "progress": args.progress,
        "lora_targets": LORA_TARGETS,
        "args": vars(args),
        "samples": len(samples),
        "optimizer_steps": total,
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if rank == 0:
        print(f"{len(samples)} samples, {total} optimizer steps of {args.accumulate}, {world} GPU(s)", flush=True)

    def save(path: Path, step: int) -> None:
        if rank:
            return
        model.save_pretrained(path)
        (path / "planner.json").write_text(json.dumps({**meta, "step": step}, indent=2, ensure_ascii=False) + "\n")

    states = json.loads((data / "states.json").read_text()) if args.state else None
    order = []
    while len(order) < total * args.accumulate:
        epoch = list(range(len(samples)))
        random.Random(args.seed + len(order)).shuffle(epoch)
        order += epoch
    order = order[: total * args.accumulate][rank::world]
    local_accumulate = args.accumulate // (world * args.batch)  # forward passes per optimizer step on this GPU
    pad_id = processor.tokenizer.pad_token_id
    pad_id = processor.tokenizer.eos_token_id if pad_id is None else pad_id
    loader = torch.utils.data.DataLoader(
        Samples(
            data,
            processor,
            samples,
            states,
            args.state_dropout,
            args.seed,
            progress=args.progress,
            lerobot=args.format == "lerobot",
            past=past,
        ),  # fmt: skip
        batch_size=None if args.batch == 1 else args.batch,
        collate_fn=None if args.batch == 1 else functools.partial(collate, pad_id=pad_id),
        sampler=order,
        num_workers=args.workers,
        prefetch_factor=4 if args.workers else None,
    )
    started, running, step = time.monotonic(), 0.0, 0
    for i, sample in enumerate(loader, 1):
        inputs = {k: v.to(device, non_blocking=True) for k, v in sample.items()}
        loss = model(**inputs).loss / local_accumulate
        loss.backward()
        running += float(loss)
        if i % local_accumulate:
            continue
        if world > 1:
            grads = torch.cat([p.grad.flatten() for p in trainable])
            torch.distributed.all_reduce(grads)
            grads /= world
            offset = 0
            for p in trainable:
                p.grad.copy_(grads[offset : offset + p.numel()].view_as(p))
                offset += p.numel()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if rank == 0 and (step % 20 == 0 or step == total):
            rate = i * args.batch * world / (time.monotonic() - started)
            print(f"step {step}/{total}: loss {running / 20:.4f}, lr {scheduler.get_last_lr()[0]:.2e}, "
                  f"{rate:.2f} samples/s", flush=True)  # fmt: skip
            running = 0.0
        if step % args.save_every == 0:
            save(out / f"step_{step:06d}", step)
    save(out / "adapter", step)
    if rank == 0:
        print(f"-> {out / 'adapter'}", flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
