# 长任务模块（WAM 即插即用，10-30 起，分支 `claude/longhorizon`）

把十个任务里的几个串成一回合连着做（例如按铃 → 递交物品 → 倒水）。世界动作模型（Phoenix-MWA、InternW0-Δ、OpenWAM）
一行不改、**不再训练**：本模块包在它们的统一适配器接口（`get_model` / `eval` / `reset_model`）外面。两种用法：

- **规划模式（10-31，主推，仿 π）：** 总目标给一句话（"先按铃，再把笔放进笔筒，最后倒水"），由一个视觉语言模型做高层，
  像 π0.5 那样每步说出下一个子任务，像 π0.6 的 MEM 那样带两层记忆（几秒的画面 + 一段文字记忆），低层是原样的 WAM。见下一节。
- **固定顺序 + 完成判别器（10-30）：** 顺序人给定，一个小判别器（默认读 Phoenix 自己的特征）判断当前子任务做完没有。见「怎么跑一回合」以后各节。

仿真（`scripts/eval_policy.py`）和真机（`run_real --goal` / `--long_horizon`）用同一份代码。

规划器的全部实验（数据、配置、评测协议、各版结果、结论、路径）记在 [实验记录.md](实验记录.md)，每出一个结果就补进去。
记忆版规划器 G（11-07）：任务链 1–10 个、可重复，记忆里带当前技能的进度 k/4（`train_planner.py --progress`、
`build_mem_samples.py`）；整链闭环评测 `eval_planner.py --chains`。

## 规划模式（仿 π0.5 + MEM）

参考 Physical Intelligence 公开的两篇：π0.5 先由同一个模型预测一句语义子任务、再出动作（arXiv 2504.16054）；
MEM（arXiv 2603.03596）给 π0.6 加了两层记忆：短期是几秒到一分钟的画面（视频编码器压缩），长期是高层模型自己写的一段文字摘要，
子任务成功后才更新，训练用的摘要由子任务标注和成败标记生成。我们的对应做法：

```
高层（RynnBrain1.1-2B，planner venv 里单独一个 worker）
  输入：总目标一句话 + 正在跑的技能 + 文字记忆（"已完成：按铃；递笔放进笔筒。"）
        + 三路相机当前画面 + 顶视 3、2、1 秒前的画面（短期视觉记忆）
  输出：skill: <十个技能之一，或 done>   memory: <已完成的事>
低层（Phoenix 等 WAM，不训）：用该技能的训练提示词执行
```

- **什么时候换：** 每段动作执行完，在后台问一次高层（不阻塞机械臂，下一段动作结束时读回答）；连续 2 次（`long_horizon_planner_confirm`）
  说要换别的技能（或 done），当前技能结束，记忆换成高层写的，两臂回起始位，直接开始它说的下一个技能。
- **回合开始、按 n、超时：** 两臂回起始位后，用"没有在跑的技能"再问一次高层。按 n 表示操作员认定做完了，记忆加上这一项；
  超时表示没做完，记忆不变，高层多半会让它重做——这就是 π 那种失败后重试。
- **为什么选 RynnBrain1.1-2B：** 阿里达摩院的具身规划模型（基于 Qwen3.5-2B），本机已有（10-23 随 InternW0 权重下的；Phoenix/InternW0 上游有个可选的 understanding 分支指向它，但默认关着，我们的 Phoenix 没用它）；
  2B 和 Phoenix 一起放在 4090 上放得下（推理约 4–5 GB，LoRA 训练约 7 GB）。一次规划 0.3–0.6 秒（GPU 空时），GPU 忙时约 1 秒。
- **零样本不行（10-31 实测）：** 它能按目标和记忆往下排，但认不出"做完了"：按铃结束、递笔放好的画面上都还让继续；
  在 7 组留出数据上一次都没切对，回复格式也常不合规（127 次无效）。所以要微调。

### 训练数据（`build_planner_data.py`）

用现有的单任务真机录制拼出串联样本（和 MEM 一样，标签来自成败标记，不用新标注）：每组录制当作某条随机目标（1–4 个技能、随机顺序、
中英文多种说法）里的一个技能，记忆写它之前的技能都已完成；这一帧的标签沿用完成判别器的规则
（成功组结尾 / 审核过的成功帧以后算做完了；失败组全程没做完；中间说不准的不训练）：
- 没做完：回答当前技能、记忆不变；
- 做完了：回答目标里的下一个技能（或 done）、记忆加上这一项。

每秒 1 帧，图 320×240（推理时也经过同样的缩放和 JPEG 压缩）。每组录制取刚做完的那一帧和最多 3 个做完的帧、4–5 个没做完的帧，
每帧配 2 条随机目标；留出 15%（和判别器同一个哈希）的组保留每一帧，给 `eval_planner.py` 逐秒重放。

### 微调（`train_planner.py`）和评测（`eval_planner.py`）

- LoRA（秩 16）只加在语言模型的投影层上（视觉编码器冻结），可训参数 1560 万；损失只算回答部分；回答接在推理时同一个
  `<think></think>` 前缀后面。冒烟测试每步约 0.35 秒（单样本），GPU 和仿真共用时约 2 样本/秒。
- 评测按真机的规则逐秒重放留出组（连续 2 次才切），统计抓到 / 提前 / 漏掉 / 失败误切、切到的下一个技能对不对、记忆对不对、延迟；
  `--every 2` 对应每段 20 步。
- 提示词、记忆格式、技能表、图像设置算成一个版本号写进 adapter，推理时版本不同就拒绝加载（只改注释不影响）。

```bash
cd <RoboSynChallenge 仓库>
V=../../../venvs/planner/bin/python            # bash policy/longhorizon/setup_planner_env.sh 建的
D=~/workspace/models/longhorizon/planner_data_20261031
$V policy/longhorizon/build_planner_data.py --out $D                                  # CPU，解码全部录制
HF_HUB_OFFLINE=1 $V policy/longhorizon/train_planner.py --data $D --out ~/workspace/models/longhorizon/planner_20261031
HF_HUB_OFFLINE=1 $V policy/longhorizon/eval_planner.py --data $D --adapter base                       # 零样本对照
HF_HUB_OFFLINE=1 $V policy/longhorizon/eval_planner.py --data $D --adapter ~/workspace/models/longhorizon/planner_20261031/adapter
```

真机：

```bash
M=<Phoenix 权重目录>
PHOENIX_ROOT=<Phoenix-MWA 源码> \
bash scripts/sim2real/sim2real_4090.sh run click_bell --policy phoenix --checkpoint_dir $M \
    --control_hz 10 --interp_hz 50 --takeover follow --max_episode_steps 3000 \
    --goal "先按铃，再把笔放进笔筒，最后倒水" --planner ~/workspace/models/longhorizon/planner_20261031/adapter \
    --overrides phoenix_step 9429 phoenix_encoders $M/encoders phoenix_wan_base <InternW0 权重目录> \
        phoenix_steps 20 phoenix_stall_steps 30 phoenix_stall_fix cover \
        long_horizon_task_overrides "{'click_bell': {'phoenix_steps': 10}, 'manipulate_pipette': {'phoenix_steps': 10}}"
```

`click_bell`（`run` 后面的任务名）只是先加载的那个技能，第一个真正执行的技能由高层选。每组结果里的 `long_horizon` 记着目标、
最终记忆、每个技能怎么结束的（planner / operator / timeout）以及期间每次规划的回答和耗时。

## 固定顺序 + 完成判别器：怎么跑一回合

```
子任务 k：WAM 用这个任务训练时的提示词跑（每段动作照常）
          └─ 每段动作的那次推理里，顺手取出 WAM 对当前画面的特征，交给完成判别器（每段一个：10 步 = 1 秒，20 步 = 2 秒）
判别器：完成概率 ≥ 该任务阈值，连续 confirm（2）次，且子任务已走 min_steps（30）步  → 子任务结束
          也可以：操作员按 n（手动结束）；走满 max_steps（1000）步 → 放弃（on_timeout next）或整条结束（stop）
回起始位姿：两臂回到本回合开始时的姿态（关节每步最多 0.05 rad，夹爪同步，到了再停 5 步）——训练数据每组都从这里开始
重置 WAM，换下一个任务的提示词 → 子任务 k+1
最后一个结束后：回起始位姿并保持，回合照常由操作员 s / q 或步数上限结束
```

- **换子任务尽量不重载模型。** 同一份权重只换提示词；执行步数、去噪步数用模型自带的在线设置（Phoenix 的
  `set_execution_steps` / `set_denoise_steps`）。某个子任务要别的权重才重建模型；Phoenix 权重相同的话沿用正在跑的 worker。
  注意：本分支基于 1d917cb9，那时的 Phoenix 适配器还没有在线设置和 worker 复用（在 `claude/real-stage` 未提交的改动里），
  所以在本分支上，两个子任务执行步数不同（按铃 10、其余 20）时，换任务会重启 worker（约 30 秒）；合并后就是即时切换。
- **提示词。** 依次取 `long_horizon_prompts`、适配器自己的训练提示词表（Phoenix 的 `phoenix_prompts`，一字不差）、
  `task_models.<任务>.prompt`。同时写进 env 的 `_current_instruction`，只看 env 指令的 OpenWAM 也能换。
- **接管。** run_real 每次接管都调 `reset_model`：只重置 WAM 的这一段动作，子任务进度不变；回合重新开始（env 步数归零）才从第一个子任务开始。
- **回合结果。** run_real 每组结果里多一个 `long_horizon`：各子任务起止步、结束原因（detector / operator / timeout）、阈值、判别器每次的完成概率。

## 完成判别器

只用官方允许的输入：三路相机和 qpos。每个"帧"一组特征，再加"距上一帧几秒"（dt），进一个因果 GRU（从子任务第一帧起），
每个任务输出一个完成概率。用 GRU 是因为有的任务完成是"事件"：按铃按完手抬起来，画面和没按差不多；移液枪要按够次数。

### 特征来源一（默认）：Phoenix 自己的特征（`wam_features.py`）

- **取什么：** 当前帧在 Phoenix 视频专家第 10、20、28 层（共 30 层）的隐藏状态，三路相机各自对自己的 token 求平均，
  得到 3 层 × 3 路 × 3072 维。最后一层不取：当前帧在最后一层的输出不参与任何损失（只解码未来帧和动作）。
- **为什么可靠：** 上游 joint96 的注意力让当前帧保持因果（只看自己、提示词和拼在提示词后的 proprio，时间步固定为 0），
  所以这些特征只由三路图像、提示词和 qpos 决定，和噪声、去噪第几步无关；又因为看了提示词，是 WAM 结合任务对画面的理解。
  10-31 在 4090 上用 `phoenix_real10trim_20261027` 第 9429 步实测：

  | 检查 | 结果 |
  |---|---|
  | 只算到第一次 MoT 前向就停 vs 完整推理（5 步去噪） | 特征完全相同（最大差 0） |
  | 换噪声种子（动作因此差 0.39 rad） | 特征完全相同 |
  | 去噪 2 步 vs 5 步 | 特征完全相同 |
  | 换一帧画面（隔 10 秒） | 平均差 0.16（特征平均绝对值 0.39） |
  | 耗时 | 完整推理 444 ms；只取特征 169 ms；显存峰值 14 GB |

- **真机上零额外开销：** 每段动作本来就要推理一次，特征是那次推理里顺手取的。做法不改 `policy/phoenix` 和上游代码：
  长任务包装层把适配器的 `phoenix_python` 设成 `phoenix_tap_worker`（本目录的启动脚本），由它在 Phoenix 的解释器里经
  `wam_features.py worker` 照常运行 `phoenix_worker.py`，只是用上游 MoT 自带的 `capture_layers` 在第一次 MoT 前向时取出这几层，
  每次推理后写到一个文件（原子替换；带 worker 进程号和计数，换 worker 也不会认错）。worker 的协议和动作不变。
  顶部相机被卡住处理遮住或标缺失的那一段，特征不交给判别器。
- **离线提取：** 同一个 `Runtime`、同样的输入（图像、提示词、qpos），推理在第一次 MoT 前向后就停，所以和真机上取到的一模一样。
  每秒取 1 帧，实测 5.6 帧/秒（含解码）。
- **代价：** 特征属于某一份 Phoenix 权重。换权重（或换步数）要重新提取、重新训判别器；运行时会检查判别器的
  `features.json` 记的权重和步数，对不上就拒绝启动。提示词也要和提取时一样（会检查）。

### 特征来源二：冻结的图像编码器（任何 WAM 都能用）

DINOv3-S/16（timm `vit_small_patch16_dinov3.lvd1689m`，冻结），224×224（整幅画面压缩），每路取 CLS 和 patch 平均（768 维），
导出成 TorchScript（`export_encoder.py`），任何有 torch 的 venv 都能跑。每秒 2 帧，图像在 env 每一步旁路取。
不依赖具体 WAM 和权重，但看不到提示词，也不是 WAM 自己的理解。InternW0 / OpenWAM 目前只能用这个。

### 标签（`train_done_detector.frame_labels`）

| 回合 | 标"完成" | 标"未完成" | 不训练 |
|---|---|---|---|
| 成功（主办方示范、我们标成功的、接管后人做完的） | 成功帧（`success_moments.py`，见下）起；规则找不到时最后 1.5 秒 | 成功帧前 1 秒以前；规则找不到时只有前 40%（录制可能在做完后还录很久，官方数据常见） | 中间那段 |
| 失败 | — | 全部，包括结尾（停下来但任务没完成，这是最要紧的负样本） | — |

成功帧（11-06，`policy/longhorizon/success_moments.py`，只用关节角和正运动学，判定标准按决赛 FAQ 第 6 条）：
- 递笔、试管、提篮、摆餐具、倒水、抽屉：两臂中较晚的那次最后松手（手臂伸在桌面上方时；回家路上张爪不算）；
- 按铃、搅拌器：按压臂（右、左）第一次到达最低点；
- 移液枪：左臂伸出后的按压，抬起 1 cm 算松开（条件是按下并松开）；
- 组装：两法兰离得最近之后、相对位置不再变的第一帧（顶视看不到插深，只是估计）。

结果在 `audit_reports/success_moments_20261106/成功时刻.csv`，覆盖 4653/4697 条成功组（没找到的 44 条都是我们的倒水，录制结束时还没松手）。
和 10-29 人工核对的对比（2 秒内）：递笔 100%、摆餐具 98–100%、试管 83–100%、组装 62%（中位差 0.3 秒）；倒水差约 11 秒是有意的，
那次用"瓶子回正"，现在按 FAQ 要两样都放回。按压类三个任务抽帧看过。

数据：主办方 `Real/cobotmagic_Real_<任务>`（599 组）+ 我们的 `real_rollouts/<任务>/<录制>`（标记删除的跳过、正在录的跳过），
都是 10 fps 原始录制。重跑提取会按当天的 `episode_success.json` 刷新标签，组号被 purge 改过的重新提取。

### 阈值

每个录制按组号哈希留出 15%。在留出组上按部署时的规则逐组重放（阈值、连续 2 次、30 步之后）：成功组在"完成"或"不训练"处触发算抓到，
在"未完成"处触发算提前，没触发算漏掉；失败组触发了就是误触发。Phoenix 特征按每秒 1 帧和 0.5 帧（对应每段 10 步和 20 步）各重放一遍，
训练时也随机隔帧；每个任务取两种速率下「提前 + 误触发 ≤ 5%」里最低的阈值，写进 `config.json`。结果在输出目录的 `report.md`。

### 训练（本机 4090）

```bash
W=<RoboSynChallenge 仓库>   # 本分支的 worktree
M=<Phoenix 权重目录>      # 要跑的 Phoenix 权重
# 1. 提取 Phoenix 特征（Phoenix 的 venv；本 worktree 没检出上游子模块，用 PHOENIX_ROOT 指向主 worktree 的；约 14 GB 显存，
#    全部录制每秒 1 帧约 10 万帧、约 5 小时；可中断，重跑接着做）
cd $W && PHOENIX_ROOT=<Phoenix-MWA 源码> \
    ../../../venvs/phoenix/bin/python policy/longhorizon/wam_features.py extract \
    --run-dir $M --step 9429 --encoders $M/encoders --wan-base <InternW0 权重目录> \
    --out ~/workspace/models/longhorizon/phoenix_real10trim_9429_features
# 2. 训练判别头并选阈值（几分钟）
policy/fmc3_no1/.venv/bin/python policy/longhorizon/train_done_detector.py \
    --features ~/workspace/models/longhorizon/phoenix_real10trim_9429_features \
    --out ~/workspace/models/longhorizon/done_phoenix_real10trim_9429
```

图像来源：`export_encoder.py`（一次，OpenWAM 的 venv 里有 timm）→ `build_done_features.py --encoder ... --out ...` → 同样的
`train_done_detector.py`。

## 用法

真机（终端，`n` 手动结束当前子任务；一回合走几个任务，步数上限要放大或用 `--until_stopped`）：

```bash
M=<Phoenix 权重目录>
PHOENIX_ROOT=<Phoenix-MWA 源码> \
bash scripts/sim2real/sim2real_4090.sh run click_bell --policy phoenix --checkpoint_dir $M \
    --control_hz 10 --interp_hz 50 --takeover follow --max_episode_steps 3000 \
    --long_horizon items_handover,water_pouring --done_detector ~/workspace/models/longhorizon/done_phoenix_real10trim_9429 \
    --overrides phoenix_step 9429 phoenix_encoders $M/encoders phoenix_wan_base <InternW0 权重目录> \
        phoenix_steps 20 phoenix_stall_steps 30 phoenix_stall_fix cover \
        long_horizon_task_overrides "{'click_bell': {'phoenix_steps': 10}, 'manipulate_pipette': {'phoenix_steps': 10}}"
```

- `task_name`（这里 click_bell）是第一个子任务，`--long_horizon` 是后面的。
- `PHOENIX_ROOT` 只在本 worktree 里跑时需要（子模块没检出）；合并回主 worktree 后不用。
- 不加 `--done_detector` 也能跑：每个子任务只由 `n` 或步数上限结束，可以先用它把整条链路走通。
- `--long_horizon` 只接世界模型（`--policy phoenix / internw0 / openwam`），不录 v2.1（录制目录只放一个任务的组），
  不和 `--rlt_actor` / `--hil_run` / `--acp_tag` 一起用；回合视频照常在 run 目录里。
- 别的设置都是 `long_horizon_*`，默认值在 `deploy_policy.yml`，`--overrides` 可改。

仿真（一个场景只有一个任务，只能验证链路和判别器；环境变量同 `policy/phoenix/eval.sh`；**还没实际跑过**）：

```bash
M=<Phoenix 训练输出>
PYTHONPATH=$PWD:$PWD/policy:../EmbodiChain EMBODICHAIN_SIM_EXIT_PROCESS=0 \
policy/fmc3_no1/.venv/bin/python scripts/eval_policy.py --config policy/longhorizon/deploy_policy.yml --overrides \
    --task_name click_bell --setting clear --gpu_id 0 --inner_policy phoenix --checkpoint_dir $M \
    --phoenix_encoders $M/encoders --phoenix_wan_base <InternW0 权重目录> \
    --long_horizon_detector <判别器目录>
```

## 现状和局限（10-31）

- 规划模式：代码、真机接线（`run_real --goal --planner`）、假规划器上的测试、真规划器 worker 端到端（录制画面回放）都通了；
  训练和评测在 3 个任务、474 个样本上冒烟跑通。**全量造数据、微调和评测还没跑**（GPU 正被仿真采集占着，微调约 5 小时），
  所以现在还不知道它能不能可靠地认出"做完了"。**还没上过真机。**
- 判别器模式：`tests/test_long_horizon.py` 里的判别器部分（含假 Phoenix 模型上的取特征、worker 外壳端到端、运行时读特征），
  `real_robot_test.py` 2 项。真 Phoenix 上验证了：取到的特征和完整推理一致、与噪声无关；真适配器 → 启动脚本 → 带 hook 的
  worker → 包装层读特征 → 判别器 → 换子任务，整条链路在录制画面上跑通；提取命令和训练在 2 个任务、36 组上冒烟跑通。
  **全量提取和训练还没跑**（约 5 小时、14 GB 显存，等台架空、并确定用哪份 Phoenix 权重后跑）。**还没上过真机。**
- 测试：`tests/test_long_horizon.py` 24 项（两种模式），`real_robot_test.py` 3 项。
- 训练数据都是单任务场景；串联时桌上同时有几个任务的物体，WAM、判别器和高层都没见过，这是最大的风险。
  高层靠文字记忆知道前面做完了什么，所以前面任务的物体留在桌上本身不要求它"看懂"，但会是没见过的画面。
- 网页控制台还没有"下一个子任务"按钮，也还没在控制台里测过 `--long_horizon`；先用终端 `run`。
- 异步推理（`phoenix_async real`）时，特征对应的是后台那次推理开始时的画面，和"本段开头"差几步，没有专门处理。

## 文件

| 文件 | 作用 |
|---|---|
| `long_horizon.py` | 子任务链：`LongHorizonPolicy`（包任意适配器模块）、`LongHorizonModel`（调度、回起始位姿、换任务、读 Phoenix 特征、规划模式、回合摘要）、`StepTap` |
| `planner_prompt.py` | 高层的提示词、技能表、目标和记忆模板、回答解析、版本号（造数据、训练、推理、评测共用） |
| `planner_runtime.py` / `planner_worker.py` | 高层模型的加载和一次规划 / 真机上的 worker（planner venv） |
| `planner.py` | 部署侧：`PlannerClient`（后台提问）、`FrameMemory`（顶视每秒一帧的短期记忆） |
| `build_planner_data.py` / `train_planner.py` / `eval_planner.py` | 高层的训练数据、LoRA 微调、留出逐秒重放评测 |
| `setup_planner_env.sh` | 建 planner venv（transformers 5.20、peft、flash-linear-attention；torch 同 Phoenix） |
| `done_detector.py` | 判别器：`DoneHead`（特征 + qpos + dt → GRU）、部署用的 `DoneDetector` / `DoneTracker`；图像来源的编码器加载和预处理 |
| `wam_features.py` | Phoenix 特征：按相机池化、`PhoenixFeatureTap`、worker 外壳（`worker`）、离线提取（`extract`） |
| `phoenix_tap_worker` | 给适配器当 `phoenix_python` 用的启动脚本 |
| `deploy_policy.py` / `.yml` | 给 `scripts/eval_policy.py` 用的元适配器（`inner_policy` 选 WAM）；`long_horizon_*` 设置的默认值（run_real 也读） |
| `build_done_features.py` | 录制清单、标签字段、解码；图像来源的编码；两种来源共用的提取循环和 `index.jsonl` |
| `export_encoder.py` | 导出图像来源的 TorchScript 编码器 |
| `train_done_detector.py` | 训练判别头、按留出组选阈值，输出判别器目录（`head.pt`、`config.json`、`features.json` 或 `encoder.pt`、`report.md`） |
| `tests/` | 不需要模型、机器人和显卡 |
