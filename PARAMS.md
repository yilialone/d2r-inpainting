# D2R-inpainting — 论文参数版代码

本目录以 `early-internal-implementation` 为基体，将全部可配置参数改为论文（npj Heritage Science 投稿稿）中的数值。
**训练数据保持原样**（`dataset/` 目录结构与路径未改动）；其余参数以论文为准。

## 参数对照表（论文 ↔ 代码）

| 参数 | 论文值 | 代码位置 | 状态 |
|---|---|---|---|
| Stage-1 学习率 | 5×10⁻⁴ | `train.py --learning_rate`（默认 5e-4） | ✅ |
| Stage-2 学习率 | 1×10⁻⁴ | `train.py --stage2_lr`（默认 1e-4，**新增**，与原版共用 lr 的 bug 已修复） | ✅ |
| 批大小 | 1 | `train.py --train_batch_size`（默认 1，原版 2） | ✅ |
| 最大 epoch | 100 | `--stage1_epochs` / `--stage2_epochs`（默认 100） | ✅ |
| 早停 patience | 20 | `train.py --early_stopping_patience`（默认 20，原版 30）；`training/stage1.py`、`training/stage2.py` 默认值同步为 20 | ✅ |
| 输入分辨率 | 512×512 | `--resolution`（默认 512） | ✅ |
| 混合精度 | fp16 | 两阶段 Accelerator 均 fp16 | ✅ |
| LoRA r / α / dropout | 32 / 64 / 0.05 | `--lora_rank` / `--lora_alpha`；dropout 0.05 固定 | ✅ |
| λ_GAN / λ_L1 / λ_tex | 0.1 / 50.0 / 10.0 | `--lambda_gan` / `--lambda_l1` / `--lambda_texture` | ✅ |
| 对抗损失 | Hinge | `--use_hinge_loss`（默认 True） | ✅ |
| 残差缩放初值 | 0.3（可学习） | `--residual_scale`（默认 0.3）；`training/stage2.py` 默认值同步 0.05→0.3 | ✅ |
| VGG 感知损失 | 论文未采用 | `--use_perceptual_loss`（默认 False） | ✅ |
| 12 通道纹理描述子 | 3RGB+3Canny+1Sobel+1Laplacian+4Gabor | `models/generator.py`（未改动） | ✅ |
| 判别器 | U-Net 逐像素 logits（限于受损区域） | `models/discriminator.py`（无末端 sigmoid） | ✅ |
| 随机种子 | 论文未指定 | `--seed`（默认 42，沿用原版） | — |
| 梯度累积 | 有效 batch size 必须为 1 | `--gradient_accumulation_steps`（默认 1） | ✅ |
| 训练数据 | 待 manifest 核验 | 支持 `--train_manifest` / `--val_manifest` | ⚠️ |

## 端到端单阶段对照（审稿意见 R2-2，新增）

| 参数 | 取值 | 代码位置 | 说明 |
|---|---|---|---|
| 训练模式 | `--mode single_stage` / `all` | `train.py` | `all` = 两阶段 + 单阶段，一条命令 |
| 输入通道 | 4（masked RGB 3 + mask 1） | `models/single_stage.py` | 无 Stage-1 输出、无文本、无扩散 |
| 输出 | 受损区域直接预测（tanh） | 同上 | 掩膜外逐字节保留原图 |
| 初始化 | 随机（无预训练） | 同上 | 不加载 SD / LoRA |
| 骨干 | 与 Stage-2 生成器逐层同构 | 同上 | 仅 enc1 首层通道与输出头不同 |
| 生成器可训练参数 | 26,177,298（base_channels=64） | `budget_report.json` | 判别器 7,085,505 |
| 学习率 | 1×10⁻⁴ | `--single_stage_lr` | 与 Stage-2 相同 |
| 损失权重 | 0.1 / 50.0 / 10.0 | 复用 `--lambda_*` | 与 Stage-2 同源实现（`training/common.py`） |
| 选模规则 | 验证集生成器损失最小 | `training/single_stage.py` | 与 Stage-2 相同，早停 patience 20 |
| 预算匹配 | 默认 `steps` | `--single_stage_budget_match` | 目标 = Stage-1 + Stage-2 实际优化步数 |
| 预算参考目录 | 两阶段输出目录 | `--single_stage_budget_ref_stage1/2` | 结果写入 `budget_report.json` |

## 两张 GPU 训练

两阶段训练器现支持 `Accelerator` 多进程 DDP。使用 `accelerate launch` 或
`torchrun` 启动两个进程；每个进程/每张卡读取自己的数据分片，验证损失在所有
进程上聚合，Stage-1 缓存也按进程分片生成。`--train_batch_size` 是每卡批大小，
因此两卡运行时全局有效批大小为 `train_batch_size × 2 × gradient_accumulation_steps`，
不再等同于论文单卡 batch size=1 的严格设置；最终论文结果应在方法部分明确硬件与
全局 batch size。

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 train.py \
  --mode full --resume_from_checkpoint none \
  --stage1_output_dir stage1_results_2gpu --stage2_output_dir stage2_results_2gpu \
  --cache_dir stage2_results_2gpu/stage1_cache
```

## 相对原版的其他修复

1. **`train.py` import bug**：原版 `from data import create_dataloaders` 指向不存在的模块，已改为 `from dataset.dataset import create_dataloaders`（原版一运行即 ImportError）。
2. **Stage-2 学习率接线**：原版 `train.py` 把同一个 `--learning_rate`（5e-4）传给两阶段，导致 Stage-2 实际用 5e-4 而非论文的 1e-4；现拆分为 `--learning_rate`（Stage-1）与 `--stage2_lr`（Stage-2，默认 1e-4）。
3. **`build_parser()`**：参数解析器独立成函数，供测试脚本校验默认值。
4. **扣洞修复（代码审计 🔴-1）→ 通道精简（v4）**：
   - 第一步（🔴-1）：`training/stage2.py` `_refine()` 把生成器输入的原图通道由"未扣洞原图"改为 **`orig_img×(1−mask) + stage1_out×mask`**，消除掩膜区域内真值像素对生成器与纹理编码器的泄漏；`inference/pipeline.py` `refine_with_stage2()` 做相同处理。
   - 第二步（v4）：该修复落地后，那个填充通道在洞内恒等于 `stage1_out`、洞外只是已知观测像素的拷贝，**不携带任何独立信息**，于是整个"原图"三通道路由被删除，生成器与纹理编码器输入统一为 **4 通道 `[I_S1; M]`**。这同时消除了与端到端单阶段对照（`[masked; M]`，同为 4 通道）之间的输入维度差异，使 R2-2 的匹配对照只剩"有无扩散先验 / 拆不拆两阶段"这一个差别。
   - 涉及文件：`models/generator.py`（`in_channels` 默认 4，删除 `x[:, 3:6]` 旧路径，构造非 4 通道直接报错）、`training/stage2.py`、`inference/pipeline.py`、`test_paper_params.py`。
   - ⚠️ **首层卷积形状随之改变，所有旧 7 通道 Stage-2 检查点不再适用**。`PROTOCOL_ID` 已升为 `d2r-stage2-paper-v4-4ch-I_S1-M-leakage-free-sample-id-logits-hinge`，带旧 ID 的 `training_state.pt` 会拒绝自动续训；`load_state_dict(strict=True)` 也会拒绝形状不匹配的 `generator.pth`。必须按 v4 重新训练 Stage-2 并重新评估全部指标。
5. **Stage-2 优化器修复**：原训练循环仅反向传播而未调用 G/D 的 `optimizer.step()`；现已补齐真实更新并加入回归测试。
6. **缓存与增强修复**：缓存由 loader 顺序改为 `sample_id` 绑定；Stage-2 的图像、mask 与 Stage-1 输出执行同参数几何增强。
7. **对抗目标一致化**：判别器输出 raw logits，默认使用 hinge loss；BCE 备选路径使用 `BCEWithLogitsLoss`。
8. **可复现性**：每次训练自动写出 `run_config.json`；自动恢复优先选择最大数字 epoch；早停后的最终 checkpoint 使用真实末轮编号。
9. **指标语义**：不再把单图 Inception 特征距离称为 FID，也不把遮罩合成 LPIPS 称为严格 mask-only LPIPS；增加集合级 KID 和 bootstrap 置信区间。
10. **掩膜外逐字节保真（本轮修复）**：`inference/pipeline.py` 的 `refine_with_stage2()` 与新增的
    `single_stage_inference()` 在 uint8 空间做二次合成。此前 uint8→float32→uint8 的往返会让部分
    掩膜外像素产生 1/255 的截断误差（例如 128→127），与"非 mask 区域严格不变"的表述不符。
    掩膜区域内的指标（PSNR_M/SSIM_M/LPIPS_M）不受影响；**全图诊断指标需用同一版代码重算后再入表**。
11. **预算记账（本轮新增）**：`Stage1DiffusionTrainer` / `Stage2GANTrainer` / `SingleStageGANTrainer`
    都会写出 `budget_report.json`（可训练参数量、实际优化步数、墙钟小时、GPU·h、峰值显存），
    并在 `training_state.pt` 中记录 `global_step` 与 `train_seconds`。这是论文 Table 4
    "Trainable parameters" 与 "Update count and GPU hours" 两列的**唯一**数据来源；
    旧检查点没有这些字段，必须重跑后才有 GPU·h。
12. **损失实现同源（本轮重构）**：对抗/纹理/感知损失与验证指标统一收敛到 `training/common.py`，
    Stage-2 与单阶段对照共用同一份实现，避免两个对照因实现差异而产生不可解释的差距。

## 已知说明（与 07 号代码审计一致，未改动）

- LoRA 已按原稿限定到注意力投影 `to_q/to_k/to_v/to_out.0`，r=32、α=64、dropout=0.05。原稿“约 120,000”不能由这些参数直接保证，最终准确数量必须从实际实例化模型导出后写回论文。
- 旧 checkpoint 与已修复的训练协议不兼容，不得用于最终论文结果。

## 测试

```bash
# 在本仓库所用环境运行（示例环境名，按需替换）
conda activate <your-env>
python test_paper_params.py
```

测试分 A–F 六节：A 参数默认值、B 模型前向、C 损失与掩膜保真、D 一步 G/D 更新、
E 真实数据端到端推理、F 单阶段对照（模型定义 / 一步训练 / 预算匹配 / 推理合成 /
Table 4 报告脚本）。缺少可选依赖 `lpips` 时 F 节会跳过推理子项而不算失败。

单阶段对照的完整操作说明见 `SINGLE_STAGE.md`。

---

## 附录：可选优化开关（opt-in，默认全部关闭）

上面的对照表描述的是**默认路径**，即论文协议。以下开关默认关闭，只有显式启用时才会
改变训练行为；`tools/check_protocol.py` 会断言它们的默认值未被改动。

| 开关 | 默认 | 作用 | 实测影响 |
|---|---|---|---|
| `--stage1_lora_ff` | 关闭 | LoRA 额外覆盖前馈层 `ff.net.0.proj` | Stage-1 可训练参数变为 **9,971,712**；协议号追加 `-ff`，旧检查点拒绝续训 |
| `--augment_photometric <f>` | `0.0` | 亮度/对比度/饱和度抖动（`saturation = f/2`），**只作用于图像** | 光度变换不移动像素，mask 无需同步；几何增强仍严格同参数作用于 image 与 mask |
| `--gradient_accumulation_steps` | `1` | 提高有效批大小 | 8 GB 显存下 batch 2 会换页（13.8 s/step），应改用梯度累积 |
| `--learning_rate` | `5e-4` | Stage-1 学习率 | ⚠️ 有效批大小为 8 时 5e-4 会退化，实测 2e-4 可用 |

**关于学习率的重要提醒。** 论文协议的有效批大小为 1、学习率 5e-4。把有效批大小提到 8
之后，按线性缩放应取 8e-4，但实测该量级过大：5e-4 在 epoch 0 即低于未微调基座
（18.966 vs 18.995 dB），epoch 1 掉到 18.512 dB。改用 2e-4 后最优达到 **19.237 dB**
（+0.242 vs 基座），优于论文协议的 +0.075 dB。逐 epoch 记录见 `docs/RESULTS.md`。

**关于评测侧。** 训练使用一段固定的青铜镜提示词，而 `evaluate.py` 默认用空 prompt。
这**不是**配置失误：12 张先导对比显示空 prompt（17.72 dB）优于训练提示词（17.43 dB）；
并且 `compute_adaptive_cfg` 在本数据集的 51 张掩膜上恒返回 8.0（全部为小洞，面积均
< 25%），等价于固定 CFG 8.0，并非逐图自适应。

## 关于本发布版

本目录是从作者工作区整理出的**精简快照**：只包含主体代码、必要脚本与文档，不含任何
模型权重或检查点。工作区中原有的约 250 个一次性分析/审计脚本、中间 PNG、日志与 docx
稿件均未包含。

**本仓库尚未完成，仍在维护中。** 已知缺陷、粗糙之处、API 稳定性说明与维护承诺见
`docs/STATUS.md`。其中最需要留意的一条：Stage-2 的 `budget_report.json` 里
`model_definition` 仍写成 7 通道输入，而实际协议已是 4 通道 —— 该字符串会被
`scripts/control_budget_report.py` 直接搬进论文 Table 4，**引用前必须核对**。

代码许可为 Apache-2.0（见 `LICENSE`），第三方组件与基座权重许可见 `NOTICE`。
数据许可是独立的：`data/public_subset/LICENSE` 只覆盖图像（CC0 1.0），不覆盖代码。

图像数据仅包含 `data/public_subset/` 下的 14 张博物馆开放获取图像（权利链清晰、无 GPS
元数据）；语料其余部分、损伤掩膜与 51 对评测划分均不分发。该子集是**持续增补**的集合，
取得新的授权后会继续添加，详见 `data/README.md`。
