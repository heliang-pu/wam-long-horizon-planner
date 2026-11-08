# WAM 长任务高层规划器

让一个**冻结的多任务世界动作模型（WAM）**连续完成多个任务：给一句目标（"先按铃，再倒水，最后再按一次铃"），由一个微调过的
小视觉语言模型当"高层"，看着相机判断当前技能做完没有、下一个做什么，并维护一段记忆；底层 WAM 不改、不训练，切任务只换提示词。

在双臂 CobotMagic（PiPER）平台的十个桌面任务上开发和评测，底层用十任务 Phoenix 模型。

- 训练、评测、推理的全部代码都在 `policy/longhorizon/`。
- 全部实验的配置、评测口径、结果和结论在 [policy/longhorizon/实验记录.md](policy/longhorizon/实验记录.md)。
- 数据和模型权重不在仓库里。

---

## 目录

1. [要解决的问题](#1-要解决的问题)
2. [架构](#2-架构)
3. [仓库结构](#3-仓库结构)
4. [环境](#4-环境)
5. [数据准备](#5-数据准备)
6. [训练](#6-训练)
7. [评测](#7-评测)
8. [推理与部署](#8-推理与部署)
9. [主要结果](#9-主要结果)
10. [局限与待做](#10-局限与待做)
11. [参考](#11-参考)

---

## 1. 要解决的问题

- **底层：** 一个十任务 WAM，按提示词做单个任务（按铃、倒水、开抽屉放番茄……），每个任务都从一个起始姿态开始。
- **目标：** 十个任务任意组合成一条长任务，可以任意顺序、任意长度，也可以重复，几个任务的物品摆在同一张桌上。
- **难点：** 什么时候算"当前任务做完了"。
  - 示范录像在成功后还会继续录很久，所以"最后一帧 = 成功"不成立。
  - 按铃、按搅拌器开关、移液枪按压这类任务，做完后的画面和开始时几乎一样。
  - 手臂做完后停住时，模型容易误以为"还没开始"。
- **约束：** 高层要在真机上跟得上，每 2–3 秒问一次；底层不重训。

## 2. 架构

```
目标句（任意组合、可重复）
        │
高层规划器：RynnBrain1.1-2B + LoRA（每 2–3 秒问一次，异步，不阻塞底层）
  输入：3 个相机的当前画面 + 顶部相机 3/2/1 秒前（G2 另加 20/10 秒前）
        目标句 + 正在做的技能 + 记忆（+ 可选的手部位置文字）
  输出：skill: <下一个技能或 done>
        memory: 已完成：A；B。当前进度：k/4。
        │  连续 2 次说切换才切换（或：当前技能的概率低于阈值，见 2.3）
        ▼
切换：手臂回到起始姿态 → 底层重置 → 换成下一个技能的提示词
        │
底层：十任务 WAM（冻结，不训练）
```

### 2.1 规划器的输入输出（`planner_prompt.py`）

- **图像：** `cam_high`、`cam_left_wrist`、`cam_right_wrist` 的当前画面，再加顶部相机若干秒前的画面（默认 3/2/1 秒前），
  全部缩到 320×240、JPEG 质量 90。训练、评测和真机用同一套处理（`planner_runtime.planner_image`）。
- **提示词：** 目标句、正在做的技能、记忆、十个技能的英文描述、回答格式。提示词内容取哈希作为"版本号"，存进 adapter 的
  `planner.json`；加载时版本号对不上就拒绝，防止训练和推理用的提示词不一致。
- **回复：** 两行，`skill: <技能名或 done>` 和 `memory: <记忆>`。

### 2.2 记忆（MEM 式语言记忆 + 阶段跟踪）

- **普通版（A/B/F）：** 记忆是"已完成：按铃；倒水。"，只在一个技能结束时更新。
- **记忆版（G/G2，`--progress`）：** 记忆写成"已完成：按铃；倒水。当前进度：2/4。"
  - 进度 k 是当前技能做到第几段，按到成功帧的时间比例分成 4 段，标签自动生成。
  - 规划器每次回答都会写进度；机器人这边用 `planner_prompt.MemoryTracker` 保存：最近 3 次回答里有 2 次到了某一段才前进，
    而且只前进不后退。这个投票规则参考了 BEHAVIOR 2025 冠军方案的阶段跟踪。
  - "已完成"列表只在真正切换后才加一项，由机器人记录，不信任模型凭空写出来的完成。
  - 作用：做完前后画面一样的任务，靠记忆里"已经做到 3/4"来判断做完了。
- **短期视觉记忆：** 顶部相机若干秒前的画面（`planner.FrameMemory`）。每个技能开始时清空，和训练时回合开头的情况一致。

### 2.3 切换规则

- **看文字（默认）：** 回复里的技能名和当前技能不同，就算一票切换；连续 `confirm` 票（默认 2）才切换。
- **按概率（`planner_runtime.plan_scored`）：** 读出回复中技能名第一个 token 的概率。RynnBrain 的分词器里，十个技能加
  done 的第一个 token 互不相同，所以每个技能都有一个明确的概率。当前技能的概率低于阈值 t 就算一票切换，切到概率最高的
  其他选项。t 越高越灵敏，误切也越多。评测里已支持（`--threshold`），真机运行时还没接上（见第 10 节）。
- **提问频率：** `long_horizon_planner_every_s`，默认 3 秒。每个技能开始时立即问一次。

### 2.4 三种用法

| 用法 | 说明 | 代码 |
|---|---|---|
| 规划器（主推） | 一句目标；规划器自己决定顺序、维护记忆、判断做完 | `long_horizon.py` 规划模式，`planner_worker.py` |
| 任务清单 + 完成判断 | 顺序在开始时定好（任务清单）；规划器只判断"当前做完没有" | 目前只在评测里用重新评分估计过（第 9 节），运行时还没实现 |
| 固定顺序 + 小判别器 | 顺序人给定；一个 GRU 小判别器读 Phoenix 自己的隐状态（或冻结的 DINOv3）判断做完 | `done_detector.py`、`train_done_detector.py`、`wam_features.py` |

## 3. 仓库结构

```
policy/longhorizon/
  planner_prompt.py        提示词、技能表、目标句模板、记忆格式、MemoryTracker、版本号
  planner_runtime.py       加载基座 + LoRA；plan()（贪心回复）和 plan_scored()（加技能概率）
  planner_worker.py        规划器子进程（规划器自己的 venv），stdin/stdout 一行一个 JSON
  planner.py               机器人这边的客户端（PlannerClient）和短期视觉记忆（FrameMemory）
  long_horizon.py          包在任意 WAM 适配器外面的长任务运行时（规划模式 / 固定顺序模式）
  deploy_policy.py/.yml    元适配器和全部 long_horizon_* 设置的默认值
  success_moments.py       用关节角 + 正运动学找每个成功回合的成功帧（FAQ 6 规则）
  build_done_features.py   数据集遍历、解码、成功帧读取（数据构造的公共部分）
  build_planner_data.py    规划器数据：每秒 1 帧抽图、帧标签、随机任务链、samples.jsonl
  build_planner_states.py  每帧的手部位置（states.json）和接管时刻（takeovers.json）
  add_hard_negatives.py    失败结尾、接管前、手停住样本 -> samples_hard.jsonl
  build_mem_samples.py     记忆版样本（链长 1-10、可重复、进度）-> samples_mem.jsonl / samples_mem2.jsonl
  train_planner.py         LoRA 微调（torchrun 多卡；--state --progress --past --format lerobot --batch）
  eval_planner.py          单回合回放、sweep（离线扫提问间隔、连续次数、阈值）、整链闭环（--chains）
  upstream_planner.py      LeRobot 外部规划器写法：零样本回放，以及按该格式构造样本、微调后回放
  claude_judge.py          用 Claude 不微调直接判断，在同一批回合上对比
  done_detector.py, train_done_detector.py, wam_features.py, export_encoder.py, phoenix_tap_worker
                           固定顺序模式的小判别器（读 WAM 隐状态或冻结图像编码器）
  setup_planner_env.sh     规划器 venv
  tests/test_long_horizon.py   单元测试（不需要 GPU、硬件和数据）
  README.md                模块内部说明（固定顺序模式、和 Phoenix 的接法等实现细节）
  实验记录.md               全部实验记录
requirements-planner.txt   规划器 venv 的依赖（和 setup_planner_env.sh 一致）
requirements-data.txt      数据构造、正运动学、运行时和单元测试的依赖
scripts/
  sim2real/ee_pose.py      PiPER 正运动学（pyAgxArm），成功帧规则用
  train_planner_8gpu.sh    训练示例
  eval_planner_8gpu.sh     评测示例（sweep / 整链，每卡一个分片）
```

## 4. 环境

需要两个 Python 环境，外加一个可选的：

| 环境 | 用途 | 关键依赖 |
|---|---|---|
| 规划器 venv | 训练、评测、规划器子进程 | Python 3.11、torch 2.7.1 + CUDA 12.8、transformers 5.20.2、peft、flash-linear-attention、pillow、av、pandas、pyarrow |
| 数据 / 运行时环境 | 读 LeRobot 数据、正运动学、长任务运行时、单元测试 | numpy、pandas、pyarrow、av、pyyaml、torch、pyAgxArm（正运动学） |
| （可选）Claude 对比 | `claude_judge.py` | anthropic、pillow；key 放在 `~/.config/claude_judge/key` |

```bash
bash policy/longhorizon/setup_planner_env.sh            # 规划器 venv，默认装到 <工作目录>/venvs/planner（PLANNER_VENV 可改）
# 国内镜像：
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
    bash policy/longhorizon/setup_planner_env.sh
```

- 基座模型：RynnBrain1.1-2B（默认路径见 `planner_runtime.BASE`，可用 `--base` 指定）。对比实验还用到 Qwen3.5-2B、
  Cosmos-Reason2-2B、Qwen3-VL-8B-Instruct、RynnBrain-8B。
- 跑单元测试：`CUDA_VISIBLE_DEVICES= python -m pytest -q policy/longhorizon/tests`（在仓库根目录，用数据 / 运行时环境）。

## 5. 数据准备

### 5.1 数据的位置和格式

数据构造脚本默认从仓库根目录下读两类数据，用软链接放过去即可：

```
Real/cobotmagic_Real_<task>/          官方真机数据（LeRobot v2.1，32 维状态，取 [0:7]+[16:23] 变成 14 维）
real_rollouts/<task>/<recording>/     自己的录像（LeRobot v2.1，14 维；episode_success.json 记录每回合的成功、失败、接管）
```

- 14 维状态：0–5 / 7–12 是左 / 右臂关节（弧度），6 / 13 是夹爪张开比例。
- 相机：`cam_high`、`cam_left_wrist`、`cam_right_wrist`，480×640。
- 帧率：真机数据都是 10 fps。规划器按每秒 1 帧使用。

### 5.2 一步一步构造（命令都在仓库根目录执行）

```bash
D=~/models/longhorizon/planner_data      # 规划器数据目录（几 GB，主要是 JPEG）

# 1) 成功帧：关节角 + 正运动学，按 FAQ 6 的成功条件（规则见实验记录 2.2）
python policy/longhorizon/success_moments.py --out audit_reports/success_moments_20261106/成功时刻.csv

# 2) 每秒 1 帧抽图、帧标签、随机任务链（留出 15% 的回合做测试，按回合哈希固定）
python policy/longhorizon/build_planner_data.py --out $D --workers 6

# 3) 手部位置（用于 --state）和接管时刻（评测时把"接管前的切换"算误切）
python policy/longhorizon/build_planner_states.py --data $D

# 4) 困难样本：失败结尾、接管前 6 秒、手停住 1-3 秒 -> samples_hard.jsonl（F）
python policy/longhorizon/add_hard_negatives.py --data $D

# 5) 记忆版样本：链长 1-10、可重复、记忆带进度 -> samples_mem.jsonl（G）
python policy/longhorizon/build_mem_samples.py --data $D
#    G2：加顶部相机 20/10 秒前、记忆滞后 2-8 秒、手停住 10-20 秒（会从录像里补抽缺的帧）
python policy/longhorizon/build_mem_samples.py --data $D --past 20,10,3,2,1 --lag 2,3,4,5,6,7,8 --long-still 0.4 \
    --out samples_mem2.jsonl

# 6) （可选）LeRobot 外部规划器格式的样本 -> samples_lerobot.jsonl（V）
python policy/longhorizon/upstream_planner.py build --data $D
```

### 5.3 帧标签（`train_done_detector.frame_labels`）

- 成功回合、找到了成功帧：从成功帧起标"做完了"；成功帧之前 1 秒内不使用；再往前标"没做完"。
- 成功回合、没找到成功帧：最后 1.5 秒标"做完了"，前 40% 标"没做完"，中间不使用。
- 失败回合：全部标"没做完"。

## 6. 训练

`train_planner.py`：LoRA（r=16，alpha=32，dropout 0.05）只加在语言模型的投影层上，视觉部分冻结；冻结部分用 bf16，LoRA 用
fp32，开梯度检查点；AdamW、学习率 1e-4、前 5% 预热后余弦衰减，1 个 epoch；只在回复部分算 loss。多卡用 torchrun，梯度手动
all-reduce。

```bash
V=<规划器 venv>/bin/python
RUN="$V -m torch.distributed.run --nproc_per_node=8 policy/longhorizon/train_planner.py --data $D"

$RUN --out out/A                                                                   # A：只看图
$RUN --out out/F --state --samples samples_hard.jsonl                              # F：+ 手部位置 + 困难样本
$RUN --out out/G --state --progress --samples samples_mem.jsonl                    # G：记忆版
$RUN --out out/G2 --state --progress --samples samples_mem2.jsonl --past 20,10,3,2,1   # G2：+ 长历史画面
$RUN --out out/G8 --state --progress --samples samples_mem.jsonl --base <Qwen3-VL-8B-Instruct>   # 换底座
$RUN --out out/V --format lerobot --samples samples_lerobot.jsonl --base <Qwen3-VL-8B-Instruct>  # LeRobot 写法
```

- `--accumulate`：每个优化步的样本总数（所有卡一共），必须是"卡数 × `--batch`"的倍数。
- `--batch`：每张卡一次前向的样本数，长度不同的样本补齐后拼成一批。实测 8B 模型上提速不明显，单条样本已经把 GPU 算满了。
- 输出：`<out>/adapter/`（LoRA 权重和 `planner.json`，里面有基座路径、提示词版本、是否用手部位置 / 进度 / 历史画面设置），
  训练中每 `--save-every` 步另存一份 `step_<n>/`。
- 用时参考（8×A100-80G，2B，约 8 万条样本）：约 1 小时；8B 约 2 小时。

## 7. 评测

全部用留出集的真实录像回放，不是真机闭环。示例脚本 `scripts/eval_planner_8gpu.sh` 每张卡跑一个分片，最后合并。

### 7.1 单回合 sweep（`eval_planner.py --sweep`）

```bash
for k in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$k $V policy/longhorizon/eval_planner.py --data $D --adapter out/G/adapter --sweep --ask-every 3 \
      --per-task 200 --out out/G/adapter/sweep_every3 --shard $k/8 &
done; wait
$V policy/longhorizon/eval_planner.py --data $D --adapter out/G/adapter --sweep --ask-every 3 --per-task 200 \
    --out out/G/adapter/sweep_every3 --merge 8      # -> sweep.md、sweep_thresholds.md
```

- 每个留出回合当作一条随机任务链中的一环；每 `--ask-every` 秒问一次，从头问到录像结束，不在切换处停下。
- 录像结束后图像停在最后一帧，模拟手臂停住，一直问到输入不再变化为止。
- 用保存下来的答案离线计算：提问间隔取 `--ask-every` 的倍数，连续 1–6 次才切，以及各个阈值（`sweep_thresholds.md`）。
- 判定：
  - 成功回合：抓到（在做完的帧切换）、提前（在没做完的帧切换）、漏掉；
  - 有人工接管的回合：在接管之前就切了，算误切；
  - 失败回合：切了就算误切。
  - "模型没做成时误切"的分母 = 失败回合数 + 接管回合数。

### 7.2 整链闭环（`eval_planner.py --chains`）

```bash
... eval_planner.py --data $D --adapter out/G/adapter --chains 60 --every 3 --confirm 2 [--threshold 0.8] --out ... --shard k/8
```

- 随机生成链长 2、3、4、6、10 的任务链，每种 60 条（可以重复，但不连着两次同一个），每个技能用一个留出的成功回合，首尾相接。
- 规划器用它自己维护的记忆往下走，和真机运行时用的是同一个 `MemoryTracker`。
- 每次切换判为抓到 / 提前 / 切错下一个 / 漏掉。出错后按正确技能继续，这样每次切换都能算到；只有全部正确才算整链做完。

### 7.3 基线

- **LeRobot 外部规划器**（`upstream_planner.py`）：照搬 LeRobot PR #4675 的提示词、规则、历史 4 次、JSON 回复和只能从固定
  列表里选技能；可以零样本回放（`replay --base <模型>`），也可以按这个格式微调后回放（`replay --adapter ... --frames frames
  --ask-every 3`）。
- **Claude 不微调直接判断**（`claude_judge.py`）：输入和提示词与我们的规划器相同，另外写明每个任务的成功条件。
  ```bash
  python policy/longhorizon/claude_judge.py run --data $D --out out/claude --model claude-opus-5-5 --effort low --per-task 12
  python policy/longhorizon/claude_judge.py report --data $D --out out/claude --compare F=out/F/adapter/sweep_every3/answers.jsonl \
      "G@0.8=out/G/adapter/sweep_every3/answers.jsonl"
  ```

## 8. 推理与部署

### 8.1 直接调用规划器（Python）

```python
from policy.longhorizon import planner_prompt as pp
from policy.longhorizon.planner_runtime import Planner, planner_image

planner = Planner("out/G/adapter", device="cuda:0")          # 基座路径从 planner.json 读，也可传 base=
memory = pp.MemoryTracker(progress=planner.uses_progress)
images = [planner_image(frame) for frame in six_frames]       # 3 个相机当前 + 顶部相机 3/2/1 秒前（G2：20/10/3/2/1）
result = planner.plan_scored(images, goal="先按铃，然后倒水。", running="click_bell", memory=memory.text())
memory.observe(result["memory"])
p_continue = result["probs"]["click_bell"]                    # 当前技能的概率；低于阈值算一票切换
```

### 8.2 规划器子进程（`planner_worker.py`）

规划器在自己的 venv 里以子进程运行，避免和底层 WAM 的依赖冲突。协议是 stdin/stdout 每行一个 JSON：

```
{"cmd": "info"}  -> {"ok": true, "info": {"adapter": ..., "prompt_version": ..., "progress": true, "past_s": [3, 2, 1]}}
{"cmd": "plan", "images_path": <npz: image_0..image_5>, "goal": ..., "running": ..., "memory": ..., "skills": [...]}
                 -> {"ok": true, "skill": ..., "memory": ..., "reply": ..., "infer_s": ...}
```

### 8.3 长任务运行时（`long_horizon.py`）

`LongHorizonPolicy` 包在任意提供 `get_model / eval / reset_model` 接口的 WAM 适配器外面（Phoenix、InternW0、OpenWAM），
在 RoboSynChallenge 代码库里通过 `run_real` 在真机上运行（仿真用 `scripts/eval_policy.py --config policy/longhorizon/deploy_policy.yml`）。
所有设置都是 `long_horizon_*`，默认值和说明在 `deploy_policy.yml`：

| 设置 | 默认 | 含义 |
|---|---|---|
| `long_horizon_goal` | "" | 规划模式的目标句 |
| `long_horizon_planner` | "" | adapter 目录（或 base 表示零样本） |
| `long_horizon_planner_every_s` | 3.0 | 两次提问之间至少隔几秒（0 = 每个动作块都问） |
| `long_horizon_planner_confirm` | 2 | 连续几次回答切换才切换 |
| `long_horizon_return_home` | true | 两个技能之间手臂回到起始姿态 |
| `long_horizon_max_steps` | 1000 | 每个技能的步数上限，超时按 `long_horizon_on_timeout` 处理 |

在真机上（RoboSynChallenge 的 `run_real`，底层以 Phoenix 为例）：

```bash
python policy/fmc3_no1/real_robot/run_real.py click_bell --policy phoenix --checkpoint_dir <Phoenix 权重目录> \
    --goal "先按铃，然后倒水，最后再按一次铃" --planner out/G/adapter --control_hz 10 \
    --overrides long_horizon_planner_every_s 3 long_horizon_planner_confirm 2
```

`--goal` 和 `--planner` 必须一起给；底层只能是世界模型（`--policy phoenix / internw0 / openwam`）。

运行时流程：
1. 开始时先问一次（没有在做的技能），选出第一个技能。
2. 每个动作块之后异步提问，读的是上一问的回复。
3. 记忆版规划器每次回复都更新进度。
4. 连续 `confirm` 次说切换后：结束当前技能，手臂回位，底层重置，开始下一个技能。
5. 操作员也可以手动结束当前技能（`n`），或者在超时后让规划器重新选。

## 9. 主要结果

详细表格、口径和局限见 [实验记录.md](policy/longhorizon/实验记录.md)。以下都是留出集录像回放，不是真机闭环。

**单回合 sweep，每 3 秒问、连续 2 次（抓到 / 提前切 / 模型没做成时误切）：**

| 规划器 | 结果 |
|---|---|
| F：RynnBrain-2B，困难样本 | 89% / 2% / 6% |
| G：RynnBrain-2B，记忆版，看文字 | 82% / 1% / 4% |
| G，阈值 0.8 | 91% / 1% / 11% |
| V：Qwen3-VL-8B，LeRobot 写法微调 | 90% / 5% / 13% |
| Claude Opus 5.5，不微调（80 个回合） | 33% / 5% / 9%（同一批回合上 F 是 93% / 0% / 4%） |
| RynnBrain-2B，不微调 | 0 |

**整链闭环，每 2 秒问、连续 2 次（链长 2 / 3 / 4 / 6 / 10 的整链做完比例）：**

| 规划器 | 结果 |
|---|---|
| A（只训练过 1–4 个、不重复的链） | 83 / 87 / 53 / 17 / 0%（1500 次切换里切错下一个 280 次） |
| G（记忆版，链长 1–10、可重复） | 90 / 77 / 53 / 52 / 28%（切错下一个 2 次） |
| F + 任务清单（顺序和"做完没有"拆开） | 93 / 92 / 68 / 70 / 47% |

**换底座（同一个记忆版配方，每 3 秒问，4 个任务的整链做完比例）：** RynnBrain1.1-2B 30%，Qwen3.5-2B 5%，Cosmos-Reason2-2B 2%。

**主要发现：**
1. 必须在自己的数据上微调。通用大模型不微调做不了，具身预训练过的小模型微调后效果最好。
2. 成功帧要准：示范在成功后还要录很久，用关节规则标成功帧，比"最后一帧"可靠得多。
3. 失败结尾、接管前、手停住这些困难样本能明显减少误切和"手停住就改口"。
4. 记忆里的阶段进度解决了做完前后画面一样的问题：手停住时判断做完 82% → 91%。
5. 长链必须用长链训练，再加上记忆，或者用显式的任务清单管顺序。
6. 每次切换的误差会连乘：每次约 89% 时，4 个任务的整链约 53%。
7. 按概率切换（阈值）是一个不用重训、可以随时调的灵敏度旋钮。

## 10. 局限与待做

- 评测都是录像回放；"漏掉"按录像结束后手臂停住仍不切换计算。同一张桌上放多个任务物品的画面还没有测过，需要录同桌回合。
- 连着两次同一个任务（例如插两次试管）还不支持：需要把"记忆里把当前技能记成已完成"也算作切换，并重训。
- 某个技能失败后没有"重来"的路径。
- 按概率切换只在评测里有，真机运行时还没接上：需要子进程把概率传回来，再加一个 `planner_threshold` 设置。
- 离心管对接（item_assembly）的成功帧标注可能有问题，它是每次切换误差最大的来源。
- 消融还不完整：F 的两类困难样本没有分开测；G 的长链训练和进度记忆也没有分开测。

## 11. 参考

- π0.5（arXiv 2504.16054）：同一个模型先预测语义子任务，再出动作。
- MEM（arXiv 2603.03596）：短期视频加语言记忆的多尺度记忆。
- BEHAVIOR Challenge 2025 冠军方案（arXiv 2512.06951）：π0.5 加阶段跟踪和投票去噪。
- SARM（arXiv 2509.25358）：阶段加进度的奖励模型。
- LeRobot PR #4675（`src/lerobot/rollout/planner.py`）：外部视觉语言模型规划器，参考 Steerable VLA Policies（arXiv 2602.13193）。
- RynnBrain（arXiv 2602.14979）：阿里达摩院的具身基础模型。
