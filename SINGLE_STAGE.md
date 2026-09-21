# 端到端单阶段对照（R2-2）操作说明

本文件说明如何用本仓库完成审稿意见 **R2-2** 要求的第二个匹配对照：
**在本数据集上端到端训练的单阶段模型**。

---

## 1. 这项对照要回答什么

本文的两阶段流程（D2R）中，第一阶段已经在青铜镜数据上做了 LoRA 微调，而用于比较的若干
基线方法是直接使用现成模型的。因此观察到的差距可能同时包含两种来源，评审时会被要求拆开：

* **架构解耦的收益** —— 把修复拆成"扩散粗修 + GAN 细化"两阶段，本身带来多少提升；
* **域适配的收益** —— 仅仅因为模型见过青铜镜这个域，带来多少提升。

要区分二者，至少需要两个匹配对照：

| 对照 | 目的 |
|---|---|
| (i) 用**完全相同**的 LoRA 配置与训练预算微调的 SD-Inpainting | 隔离"域适配"的贡献 |
| (ii) 在本数据集上**端到端训练的单阶段模型** | 隔离"两阶段解耦"的贡献 |

单阶段对照必须**独立训练**，并显式说明其**架构、损失、可训练参数与训练预算**。把
Stage-1-only 换个名字不构成对照；未说明预算的全量微调也不能自动视为同预算对照。

本仓库的分工：

| 对照 | 由什么满足 | 状态 |
|---|---|---|
| (i) 同 LoRA 配置与预算微调的 SD-Inpainting | 现有 Stage-1 检查点本身（"仅第一阶段"行） | 已有产物 |
| (ii) 端到端训练的单阶段模型 | **本文件 + `training/single_stage.py`** | 本仓库新增 |

---

## 2. 单阶段对照是什么（必须原样写进论文的定义）

**`SingleStageInpaintingGenerator`**（`models/single_stage.py`）：

- **输入**：4 通道 = 扣洞后的 masked RGB(3) + mask(1)。
  **没有** Stage-1 输出这一路，**没有**文本提示，**没有**扩散采样。
- **输出**：对受损区域内容的**直接预测**（tanh，[-1, 1]）；
  掩膜外区域在合成时严格保留原图（逐字节不变）。
- **初始化**：随机初始化，**不用任何预训练权重**（不加载 SD，不加载 LoRA）。
- **骨干**：与 D2R 第二阶段（残差细化生成器）**逐层完全同构** —— 同一套编解码块、SE 通道注意力、
  深层自注意力、12 通道多滤波器纹理编码器（3 RGB + 3 Canny + 1 Sobel + 1 Laplacian + 4 Gabor）
  与纹理注意力门控。v4 起两侧输入通道数都是 4，因此**唯一差别是残差缩放参数**（单阶段为直接
  预测，无 `learnable_scale`）与输出头语义（直接预测 vs 残差）。
  这一点是**故意**的：骨干容量与输入维度都相同，比较才干净 —— 差异只剩下"有没有扩散先验 /
  拆不拆两阶段"。
- **输入语义与 D2R 的对应**：D2R 的细化输入是 `[I_s; M]`，其中 `I_s` 在洞外保留观测像素、
  洞内为扩散预测；本对照的输入是 `[X_obs; M]`，其中 `X_obs` 在洞外是观测像素、洞内置零。
  两侧同为 4 通道，逐位对应。
- **判别器**：与 Stage-2 相同的 `SimpleUNetDiscriminator`，逐像素 logits，只看受损区域。
- **损失**：与 Stage-2 完全相同的三项（实现同源，均来自 `training/common.py`）：
  hinge 对抗（λ=0.1）+ 掩膜内 L1（λ=50）+ 掩膜内归一化 Sobel 纹理（λ=10）。
  感知损失默认关闭，与论文一致。
- **不读取 Stage-1 缓存**：这是完全独立的一次训练，与 D2R 的任何检查点无关。

**与 D2R 对齐的其他协议**：同一 train/val 划分、512×512、同一同步几何增强（hflip + ±5° 旋转）、
batch size 1、梯度累积 1、AdamW（lr = 1e-4，betas = (0.5, 0.999)，与 Stage-2 相同）、
早停 patience 20、**选模规则相同**（验证集生成器损失最小者为最佳检查点）。

**默认规模**（`--single_stage_base_channels 64`）：生成器 26,177,298 个可训练参数，
判别器 7,085,505 个；D2R 的对应量为 LoRA 可训练参数 + Stage-2 生成器与判别器。
实际数值一律以 `budget_report.json` 为准，不要手抄正文里的旧估计值。

---

## 3. 预算匹配（审稿人最在意的一点）

单阶段对照默认按**更新步数**匹配参考 D2R 运行：

```
目标步数 = Stage-1 实际优化步数 + Stage-2 实际优化步数
```

推导优先级（写在 `budget_report.json` 的 `budget_match.reference` 里，可审计）：

1. `training_state.pt` 的 `global_step`（本版本起记录，最准确）；
2. 最新 `checkpoint-epoch-N` 的 N × 每 epoch 步数（旧检查点的回退口径，会在报告中标注"估算"）；
3. `run_config.json` 的 `max_train_steps`（**计划值**，不是实际值，会显式标注）。

三种口径：

| `--single_stage_budget_match` | 停止条件 | 适用场景 |
|---|---|---|
| `steps`（默认） | 达到目标更新步数 | 与 Table 4 的 "Update count" 列直接对应 |
| `gpu_hours` | 达到目标 GPU·h | 想按挂机时间对齐时使用（需参考运行记录过 `train_seconds`，即本版本之后重跑的 D2R） |
| `none` | 只按 `--single_stage_epochs` | 仅用于调试；报告中会标注"未匹配预算"，**不可用于投稿** |

**GPU·h 口径**（写在报告的 `budget.gpu_hours_definition` 中）：

```
gpu_hours = wall_clock_seconds × world_size / 3600
```

即"所有参与进程的 GPU 占用时间之和"。单卡单进程时等于挂机小时数。

**不要说**"两个对照的 FLOPs 完全相同"：步数匹配 ≠ 计算量匹配（单阶段每步比 Stage-1 便宜得多）。
正确写法是**同时给出两者的实际 GPU·h**，并说明匹配的是更新步数。这一点审稿人明确要求
"budget stated explicitly"，如实报告比含糊其辞安全。

---

## 4. 怎么跑

### 4.1 三个 seed（与 D2R 的 2026 / 2027 / 2028 对应）

先确认 D2R 的三个 seed 已经用**本版本代码**重跑（旧检查点没有计时，也没有 `global_step`）：

```bash
python train.py --mode full --seed 2026 \
  --stage1_output_dir stage1_results_seed2026 \
  --stage2_output_dir stage2_results_seed2026 \
  --cache_dir stage2_results_seed2026/stage1_cache \
  --resume_from_checkpoint none
```

再跑对应的单阶段对照（预算自动取自上面那次运行）：

```bash
python train.py --mode single_stage --seed 2026 \
  --single_stage_output_dir single_stage_results_seed2026 \
  --single_stage_budget_ref_stage1 stage1_results_seed2026 \
  --single_stage_budget_ref_stage2 stage2_results_seed2026 \
  --resume_from_checkpoint none
```

`--seed` 换成 2027 / 2028 重复。**每次用独立输出目录，不要续训旧目录。**

### 4.2 一条命令跑完（推荐）

```bash
python train.py --mode all --seed 2026 \
  --stage1_output_dir stage1_results_seed2026 \
  --stage2_output_dir stage2_results_seed2026 \
  --single_stage_output_dir single_stage_results_seed2026 \
  --resume_from_checkpoint none
```

`--mode all` = 两阶段 + 单阶段，单阶段预算自动取刚跑完的两阶段。

### 4.3 双卡

与两阶段一致，`--train_batch_size` 是**每卡**批大小：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 train.py \
  --mode single_stage --seed 2026 \
  --single_stage_output_dir single_stage_results_seed2026 \
  --single_stage_budget_ref_stage1 stage1_results_seed2026 \
  --single_stage_budget_ref_stage2 stage2_results_seed2026 \
  --resume_from_checkpoint none
```

注意：单阶段对照**不需要** Stable Diffusion 权重，所以不设 `--model_name` 也能跑；
它比 Stage-1 快很多（没有 VAE 与扩散 UNet）。若两卡运行，请在论文中说明全局有效批大小。

### 4.4 意见里的第一个对照（可选：独立重训的 SD-Inpainting-LoRA）

R2-2 的第 (i) 项要求"用完全相同的 LoRA 配置与预算微调的 SD-Inpainting"。
Stage-1-only 行直接用 D2R 自己的 Stage-1 检查点即可满足（同模型、同 LoRA、同预算、
同推理设置）。如果审稿人要求一个**独立训练**的副本，用现有入口单独跑一次 Stage-1：

```bash
python train.py --mode stage1 --seed 2026 \
  --stage1_output_dir stage1_results_lora_baseline_seed2026 \
  --resume_from_checkpoint none
```

该命令同样会写出 `budget_report.json`（LoRA 可训练参数、优化步数、GPU·h），
可直接作为 Table 4 中 Stage-1-only 行的独立证据。

---

## 5. 产出物（每个 run 目录）

```
single_stage_results_seed2026/
├── run_config.json            启动参数 + 预算匹配决策（含参考运行来源）
├── budget_report.json         ★ Table 4 所需：模型定义 / 参数量 / 更新步数 / GPU·h
├── epoch_metrics.jsonl        每 epoch 的训练与验证指标（可审计选模过程）
├── checkpoint-best/
│   ├── generator.pth          ★ 推理与评估加载这个文件
│   ├── discriminator.pth
│   ├── training_state.pt      epoch / global_step / 优化器 / protocol_id
│   ├── BEST_MODEL
│   └── validation_metrics.txt
└── checkpoint-epoch-*/
```

`training_state.pt` 带 `protocol_id = d2r-single-stage-paper-v1-direct-4ch-masked-input`，
用于阻止误续训不同协议的检查点。

---

## 6. 怎么评估（与其他方法完全同一协议）

单阶段对照**不需要** `--stage1_checkpoint` / `--stage2_checkpoint`，直接指到运行目录即可：

```bash
python infer.py --image_dir datasets/val/img --mask_dir datasets/val/mask \
  --manifest manifests/val.csv \
  --single_stage_checkpoint single_stage_results_seed2026 \
  --out evaluation_single_stage_seed2026 --compute_metrics
```

D2R 与 Stage-1-only 用同一条命令、同一 test manifest、同一 `--steps/--guidance_scale` 设置：

```bash
# D2R
python infer.py --image_dir datasets/val/img --mask_dir datasets/val/mask --manifest manifests/val.csv \
  --stage1_checkpoint stage1_results_seed2026/checkpoint-best \
  --stage2_checkpoint stage2_results_seed2026/checkpoint-best \
  --out evaluation_d2r_seed2026 --compute_metrics

# Stage-1-only（同 Stage-1 检查点，不加 Stage-2）
python infer.py --image_dir datasets/val/img --mask_dir datasets/val/mask --manifest manifests/val.csv \
  --stage1_checkpoint stage1_results_seed2026/checkpoint-best \
  --out evaluation_stage1_only_seed2026 --compute_metrics
```

论文的标量指标取 `batch_metrics.json` 的 `mask_region`（PSNR_M / SSIM_M / LPIPS_M），
与 Table 3 / Table 4 的口径一致。**不要**用训练期的验证代理指标填表。

掩膜外像素在两条推理路径上都已逐字节等于原图（uint8 空间二次合成），
因此"非 mask 区域严格不变"的说法在实现层面成立。

---

## 7. 生成 Table 4 素材

```bash
python scripts/control_budget_report.py \
  --stage1_dir stage1_results_seed2026 \
  --stage2_dir stage2_results_seed2026 \
  --single_stage_dir single_stage_results_seed2026 \
  --d2r_metrics evaluation_d2r_seed2026/batch_metrics.json \
  --stage1_only_metrics evaluation_stage1_only_seed2026/batch_metrics.json \
  --single_stage_metrics evaluation_single_stage_seed2026/batch_metrics.json \
  --out TABLE4_seed2026.md --json_out TABLE4_seed2026.json
```

脚本只读取训练产物，**不会编造数字**：缺失的量保留 `[[ MISSING ]]` 占位符，
并在"缺失项"一节列出原因。投稿前必须把所有占位符清零。

---

## 8. 论文里要写的话

### 8.1 Table 4 两个空槽位

| 槽位 | 填什么 | 来源 |
|---|---|---|
| `[[SINGLE_STAGE_MODEL_AND_TRAINING]]` | 见 §2 的架构 + 损失 + 随机初始化 + 同骨干说明（可直接用 `budget_report.json` 的 `model_definition` 与 `losses` 拼接） | `budget_report.json` |
| `[[SINGLE_PARAMS]]` | 生成器可训练参数（默认 26,177,298）；若同时给出判别器请注明 | `budget_report.json` |
| `[[SINGLE_BUDGET]]` | 更新步数 + GPU·h（例如 `27,279 updates, 6.4 GPU·h`） | `budget_report.json` |
| `[[SINGLE_PSNR]] / [[SINGLE_SSIM]] / [[SINGLE_LPIPS]]` | 掩膜区域三项指标 | `batch_metrics.json` |

### 8.2 3.2.3 结果段（模板，`[[ ]]` 填实测值）

> The end-to-end single-stage control obtained a masked PSNR of `[[SINGLE_PSNR]]` dB,
> SSIM of `[[SINGLE_SSIM]]` and LPIPS of `[[SINGLE_LPIPS]]`, compared with `[[D2R_PSNR]]` dB,
> `[[D2R_SSIM]]` and `[[D2R_LPIPS]]` for D2R; the paired artefact-equal differences were
> `[[DELTA_PSNR]]` dB, `[[DELTA_SSIM]]` and `[[DELTA_LPIPS]]` (Table 8).
> This comparison tests whether two-stage decoupling adds anything over training a single
> network end to end on the same corpus under the same update budget.

### 8.3 §4 Discussion（按实测结论二选一）

- **D2R 明显领先**：
  > The end-to-end single-stage control trained under the same protocol is a closer alternative,
  > and D2R remains ahead of it by `[[DELTA]]` (Table 8); the residual difference is interpreted
  > within the documented budget and pretraining constraints rather than attributed to decoupling alone.
- **两者不可分辨**：
  > The end-to-end single-stage control is not distinguishable from D2R on these endpoints at this
  > sample size, so the present experiments do not establish an advantage for two-stage decoupling
  > over a single-stage model trained on the same corpus; the contribution of the decoupled design
  > therefore rests on modularity and on the surface properties that remain unmeasured here.

### 8.4 不要写的话

- ❌ "预算完全匹配" / "FLOPs 完全相同" → 只匹配更新步数，GPU·h 如实并列报告。
- ❌ "Stage-1-only 就是单阶段对照" → 审稿人已明确否掉这个说法。
- ❌ 把单阶段对照写进 Table 7（结构标注）→ 该表仍只含六个方法，正文已保留
  "no structural annotation was performed for it" 的限制说明。
- ❌ 用未重跑的旧 D2R 数字做比较 → 旧检查点与修复后的训练/推理协议不兼容。

---

## 9. 已知限制（建议主动写进 Limitations）

1. **步数匹配 ≠ 算力匹配**：单阶段每步比 Stage-1 便宜（无 VAE、无扩散 UNet），
   因此"同等更新次数"下它的 GPU·h 更低；文中同时给出两者 GPU·h。
2. **无预训练**：单阶段对照从随机初始化开始，D2R 的 Stage-1 站在 SD 预训练上。
   因此差异混合了"两阶段解耦"与"预训练先验"两个因素 —— 本对照的作用是给出**下界式**
   的单阶段基线，而不是把两个因素彻底分离；正文按 §8.3 的口径措辞。
3. **单一骨干族**：单阶段对照使用与 Stage-2 相同的骨干，因此结论只适用于
   "这一族单阶段模型"，不能推广到所有单阶段架构（正文模板已含 "These results apply to the
   stated configurations"）。
4. 三 seed 只提供描述性的训练波动，不构成显著性检验（与 Table 9 的既有口径一致）。

---

## 10. 自检

```bash
# 语法与协议测试（含单阶段对照的 F 节）
python -m compileall -q dataset models training metrics inference utils train.py infer.py scripts
python test_paper_params.py --quick

# 单阶段训练器最小冒烟（CPU，几秒）
python train.py --mode single_stage --allow_cpu --no-augment --resolution 64 \
  --train_image_dir datasets/train/img --train_mask_dir datasets/train/mask \
  --single_stage_epochs 1 --single_stage_budget_match none \
  --single_stage_output_dir /tmp/ss_smoke --resume_from_checkpoint none
```

服务器正式跑之前先执行 `python scripts/check_server.py`（单阶段对照本身不需要 SD 权重，
但同一台机器上的 D2R 需要）。
