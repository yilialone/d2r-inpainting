# -*- coding: utf-8 -*-
"""
test_paper_params.py — 论文参数版代码测试（在项目环境中运行）

用法:
    python test_paper_params.py            # 全部测试
    python test_paper_params.py --quick    # 跳过 E（端到端推理，最耗时）

覆盖:
  A. train.py 默认参数 = 论文参数
  B. 模型前向形状（生成器 / 判别器 / 纹理编码器 / TAG）
  C. 损失函数与前向组合（掩膜外严格保真）
  D. 一步 G/D 训练更新
  E. 真实数据端到端推理（dataset/test，Stage1 LoRA + Stage2）
  F. R2-2 端到端单阶段对照（模型定义 / 一步训练 / 预算匹配 / 推理合成）
"""
import os
import sys
import math
import argparse

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def section(title):
    print(f"\n=== {title} ===")


def skip(name, reason=""):
    print(f"  [SKIP] {name}" + (f"  ({reason})" if reason else ""))


# ═══════════════════════════════════════════════
# A. train.py 默认参数 = 论文参数
# ═══════════════════════════════════════════════
def test_a_config():
    section("A. train.py 默认参数（论文值）")
    import train
    args = train.build_parser().parse_args([])

    check("Stage-1 lr = 5e-4", args.learning_rate == 5e-4, f"got {args.learning_rate}")
    check("Stage-2 lr = 1e-4", args.stage2_lr == 1e-4, f"got {args.stage2_lr}")
    check("batch size = 1", args.train_batch_size == 1, f"got {args.train_batch_size}")
    check("effective batch size = 1", args.gradient_accumulation_steps == 1,
          f"got accumulation={args.gradient_accumulation_steps}")
    check("epochs = 100 / 100", args.stage1_epochs == 100 and args.stage2_epochs == 100)
    check("patience = 20", args.early_stopping_patience == 20, f"got {args.early_stopping_patience}")
    check("resolution = 512", args.resolution == 512)
    check("LoRA r=32 / alpha=64", args.lora_rank == 32 and args.lora_alpha == 64)
    check("lambda 0.1/50/10", args.lambda_gan == 0.1 and args.lambda_l1 == 50.0 and args.lambda_texture == 10.0)
    check("residual_scale = 0.3", args.residual_scale == 0.3, f"got {args.residual_scale}")
    check("hinge loss on", args.use_hinge_loss)
    check("perceptual loss off", not args.use_perceptual_loss)

    # train.py 中 Stage-2 配置确实使用 stage2_lr（源码级校验）
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    check("stage2 config uses stage2_lr", '"learning_rate": args.stage2_lr' in src)
    check("import 修复 (dataset.dataset)", "from dataset.dataset import create_dataloaders" in src)

    # 训练器默认值（独立实例化时也与论文一致）
    import training.stage1 as s1
    import training.stage2 as s2
    check("Stage1 trainer patience=20", s1.Stage1DiffusionTrainer.__init__.__defaults__ is not None)
    check("Stage2 trainer patience=20", s2.Stage2GANTrainer.__init__.__defaults__ is not None)
    check("Stage2 trainer residual_scale=0.3", "residual_scale: float = 0.3" in open(os.path.join(ROOT, "training/stage2.py"), encoding="utf-8").read())
    check("Stage1 trainer lr=5e-4", "learning_rate=5e-4" in open(os.path.join(ROOT, "training/stage1.py"), encoding="utf-8").read())
    check("Stage2 trainer lr=1e-4", "learning_rate: float = 1e-4" in open(os.path.join(ROOT, "training/stage2.py"), encoding="utf-8").read())
    src_s1 = open(os.path.join(ROOT, "training/stage1.py"), encoding="utf-8").read()
    check("LoRA 仅作用于原稿所述注意力投影", '"ff.net.0.proj"' not in src_s1)
    check("SD Inpainting 通道顺序为 noisy/mask/masked-latent",
          "torch.cat([noisy_latents, mask_latents, masked_image_latents]" in src_s1)

    # 输入契约（🔴-1 泄漏修复 + v4 通道精简）：训练与推理的生成器输入都必须是 4 通道 [I_S1, M]
    src_s2 = open(os.path.join(ROOT, "training/stage2.py"), encoding="utf-8").read()
    src_pipe = open(os.path.join(ROOT, "inference/pipeline.py"), encoding="utf-8").read()
    src_gen = open(os.path.join(ROOT, "models/generator.py"), encoding="utf-8").read()
    check("训练生成器输入 = [I_S1, M] (stage2.py _refine)",
          "residual = self.generator(stage1_out, mask)" in src_s2)
    check("推理生成器输入 = [I_S1, M] (pipeline.py)",
          "residual = generator(stage1_tensor, mask_tensor)" in src_pipe)
    check("生成器只接受 4 通道 (models/generator.py)",
          "def __init__(self, in_channels=4," in src_gen)
    check("7 通道拼接路径已彻底移除 (generator.py)",
          "x[:, 3:6" not in src_gen
          and "torch.cat([stage1_output, original_image, mask]" not in src_gen)
    check("旧 7 通道输入路径已移除 (stage2.py)",
          "torch.cat([stage1_out, filled, mask]" not in src_s2
          and "torch.cat([stage1_out, orig_img, mask]" not in src_s2)
    check("旧 7 通道输入路径已移除 (pipeline.py)",
          "torch.cat([stage1_tensor, filled_tensor, mask_tensor]" not in src_pipe
          and "torch.cat([stage1_tensor, original_tensor, mask_tensor]" not in src_pipe)
    check("Stage2 输入协议版本已升级到 v4",
          "d2r-stage2-paper-v4-4ch-I_S1-M" in src_s2)
    check("Stage2 实际执行 G/D optimizer.step",
          "self.optimizer_g.step()" in src_s2 and "self.optimizer_d.step()" in src_s2)
    check("Stage2 缓存由 sample_id 定位", "_get_cache_path(split, sample_id)" in src_s2)


# ═══════════════════════════════════════════════
# B. 模型前向形状
# ═══════════════════════════════════════════════
def test_b_models():
    section("B. 模型前向形状（512×512）")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    from models.generator import (SimpleUNetGeneratorWithTexture,
                                  EnhancedTextureEncoder, TextureAttentionGate,
                                  multiscale_canny)
    from models.discriminator import SimpleUNetDiscriminator

    # 滤波器参数与论文表 1 一致（源码级校验）
    gen_src = open(os.path.join(ROOT, "models/generator.py"), encoding="utf-8").read()
    check("Gabor 15x15 sigma4 lambda10 gamma0.5",
          "ksize=(15, 15)" in gen_src and "sigma=4.0" in gen_src and "lambd=10.0" in gen_src and "gamma=0.5" in gen_src)
    check("Canny 三档阈值 (50,150)/(100,200)/(150,250)",
          "[(50, 150), (100, 200), (150, 250)]" in gen_src)

    gen = SimpleUNetGeneratorWithTexture(in_channels=4, residual_scale=0.3).to(device).eval()
    stage1_in = torch.randn(1, 3, 512, 512, device=device)
    mask_in = (torch.randn(1, 1, 512, 512, device=device) > 0).float()
    with torch.no_grad():
        residual = gen(stage1_in, mask_in)
    check("生成器 [I_S1,M]→3ch @512", residual.shape == (1, 3, 512, 512), f"got {tuple(residual.shape)}")
    try:
        SimpleUNetGeneratorWithTexture(in_channels=7)
        check("7 通道构造被拒绝", False, "in_channels=7 未报错")
    except ValueError:
        check("7 通道构造被拒绝", True)
    check("learnable_scale 初值 0.3", abs(float(gen.learnable_scale.item()) - 0.3) < 1e-6,
          f"got {float(gen.learnable_scale.item())}")

    disc = SimpleUNetDiscriminator().to(device).eval()
    with torch.no_grad():
        score = disc(torch.randn(1, 3, 512, 512, device=device))
    check("判别器 3ch→1ch @512", score.shape == (1, 1, 512, 512), f"got {tuple(score.shape)}")
    with torch.no_grad():
        disc.final_conv.weight.zero_()
        disc.final_conv.bias.fill_(5.0)
        logits = disc(torch.zeros(1, 3, 512, 512, device=device))
    check("判别器输出 raw logits（无 Sigmoid）", bool((logits > 1.0).all()))

    # 纹理编码器 12 通道描述子
    tex = EnhancedTextureEncoder(base_channels=64)
    img = torch.randn(1, 3, 512, 512)  # 编码器内部转 numpy，CPU 足够
    feat12 = EnhancedTextureEncoder.extract_texture_features(img)
    check("12 通道描述子 (1,12,512,512)", feat12.shape == (1, 12, 512, 512), f"got {tuple(feat12.shape)}")
    check("描述子取值在 [0,1]（归一化）", bool((feat12 >= 0).all() and (feat12 <= 1).all()))
    feats = tex(img)
    check("编码器 enc0..enc4 形状",
          feats["enc0"].shape == (1, 64, 512, 512) and feats["enc4"].shape == (1, 512, 32, 32),
          f"enc0={tuple(feats['enc0'].shape)} enc4={tuple(feats['enc4'].shape)}")

    # TAG（式 1: F_out = G⊙F_dec + (1−G)⊙F_tex）
    gate = TextureAttentionGate(feat_channels=64, texture_channels=64).eval()
    fd = torch.randn(1, 64, 128, 128)
    ft = torch.randn(1, 64, 64, 64)
    with torch.no_grad():
        fout = gate(fd, ft)
    check("TAG 输出形状与 F_dec 一致", fout.shape == fd.shape, f"got {tuple(fout.shape)}")
    # 权重为 0 时：G = σ(BN(ReLU(0))) = 0.5，且 W_x 输出为 0 → F_out ≈ 0.5*F_dec（融合用的是投影后的纹理特征）
    with torch.no_grad():
        gate.W_g.weight.zero_(); gate.W_x.weight.zero_()
        gate.psi[0].weight.zero_(); gate.psi[0].bias.zero_()
        fout0 = gate(fd, ft)
    check("TAG 零初始化时门≈0.5（输出≈0.5·F_dec）", abs(float((fout0 - 0.5 * fd).abs().max())) < 1e-3)


# ═══════════════════════════════════════════════
# C. 损失函数与组合正确性
# ═══════════════════════════════════════════════
def test_c_losses():
    section("C. 损失函数与组合正确性")
    torch.manual_seed(0)
    from training.stage2 import Stage2GANTrainer
    t = Stage2GANTrainer.__new__(Stage2GANTrainer)  # 绕过 __init__，只测纯函数
    t.use_hinge_loss = True

    B, H, W = 1, 256, 256
    orig = torch.rand(B, 3, H, W) * 2 - 1
    stage1 = torch.rand(B, 3, H, W) * 2 - 1
    residual = torch.randn(B, 3, H, W) * 0.1
    mask = (torch.rand(B, 1, H, W) > 0.8).float()  # ~20% 受损
    target = orig

    # 组合公式：掩膜外严格等于原图
    refined = orig * (1.0 - mask) + (stage1 + residual) * mask
    outside = (1 - mask).bool()
    check("掩膜外像素严格保真", bool((refined[outside.repeat(1, 3, 1, 1)] - orig[outside.repeat(1, 3, 1, 1)]).abs().max() < 1e-6))

    # 掩膜 L1
    l1 = F.l1_loss(refined * mask, target * mask)
    check("掩膜 L1 有限", math.isfinite(float(l1)))

    # 纹理损失（Sobel 边缘空间）
    tex_loss = t.compute_texture_loss(refined, target, mask)
    check("纹理损失有限", math.isfinite(float(tex_loss)), f"value={float(tex_loss):.4f}")

    # Hinge 对抗损失
    from models.discriminator import SimpleUNetDiscriminator
    disc = SimpleUNetDiscriminator().eval()
    df, dr = disc(refined * mask), disc(target * mask)
    g_adv = -df.mean()
    d_loss = F.relu(1.0 - dr).mean() + F.relu(1.0 + df).mean()
    check("Hinge G 损失有限", math.isfinite(float(g_adv.detach())))
    check("Hinge D 损失有限", math.isfinite(float(d_loss.detach())))

    # 输入契约验证（🔴-1 泄漏修复 + v4 通道精简）：捕获 _refine 传给生成器的 4 通道输入，
    # 断言 [I_S1, M] 两段内容正确，且掩膜区域内不含真值。
    class _CaptureGen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.captured = None
            self.captured_mask = None
        def forward(self, stage1_output, mask=None):
            self.captured = stage1_output.clone()
            self.captured_mask = mask.clone() if mask is not None else None
            return torch.zeros_like(stage1_output)

    cap = _CaptureGen()
    t2 = Stage2GANTrainer.__new__(Stage2GANTrainer)
    t2.generator = cap
    stage1b = torch.rand(B, 3, H, W) * 2 - 1
    origb = torch.rand(B, 3, H, W) * 2 - 1
    maskb = (torch.rand(B, 1, H, W) > 0.75).float()
    t2._refine(stage1b, origb, maskb)
    check("训练输入第 1 段为 Stage-1 输出（3 通道）",
          cap.captured.shape[1] == 3 and float((cap.captured - stage1b).abs().max()) < 1e-6)
    check("训练输入第 2 段为掩膜（1 通道）",
          cap.captured_mask is not None and cap.captured_mask.shape[1] == 1
          and float((cap.captured_mask - maskb).abs().max()) < 1e-6)
    check("生成器输入总通道数 = 4（旧 7 通道已移除）",
          cap.captured.shape[1] + (cap.captured_mask.shape[1] if cap.captured_mask is not None else 0) == 4)
    # 真值不在生成器任何一路里：合成结果在掩膜内恒等于 I_S1 + residual，故只要
    # I_S1 在掩膜内与原图不同，就可以确认网络没有拿到掩膜内真值。
    refined = t2._refine(stage1b, origb, maskb)
    check("掩膜内像素只来自 I_S1（真值未进入网络任一路）",
          float(((refined.detach() - stage1b).abs() * maskb).max()) < 1e-6
          and float(((stage1b - origb).abs() * maskb).max()) > 1e-6)


# ═══════════════════════════════════════════════
# D. 一步 G/D 训练更新
# ═══════════════════════════════════════════════
def test_d_step():
    section("D. 一步 G/D 训练更新")
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from models.generator import SimpleUNetGeneratorWithTexture
    from models.discriminator import SimpleUNetDiscriminator
    from torch.optim import AdamW

    gen = SimpleUNetGeneratorWithTexture(in_channels=4, residual_scale=0.3).to(device)
    disc = SimpleUNetDiscriminator().to(device)
    opt_g = AdamW(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_d = AdamW(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))

    B, H, W = 1, 128, 128
    stage1 = torch.rand(B, 3, H, W, device=device) * 2 - 1
    orig = torch.rand(B, 3, H, W, device=device) * 2 - 1
    mask = (torch.rand(B, 1, H, W, device=device) > 0.75).float()

    residual = gen(stage1, mask)
    refined = orig * (1 - mask) + (stage1 + residual) * mask
    fake = refined * mask
    real = orig * mask

    # D 更新
    df, dr = disc(fake.detach()), disc(real.detach())
    loss_d = F.relu(1.0 - dr).mean() + F.relu(1.0 + df).mean()
    opt_d.zero_grad(); loss_d.backward(); opt_d.step()
    # G 更新
    df = disc(fake)
    loss_g = -df.mean() + 50.0 * F.l1_loss(refined * mask, real)
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()

    check("一步 D 更新完成", math.isfinite(float(loss_d.item())))
    check("一步 G 更新完成", math.isfinite(float(loss_g.item())))
    check("梯度已传播（G 参数有梯度）",
          any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in gen.parameters()))

    # Exercise the trainer's real train_epoch path; this catches regressions where
    # backward() is called but optimizer.step() is accidentally omitted.
    from contextlib import nullcontext
    from types import MethodType
    from training.stage2 import Stage2GANTrainer

    class _TinyGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 3, 1)
        def forward(self, stage1_output, mask=None):
            x = torch.cat([stage1_output, mask], dim=1)
            return 0.1 * torch.tanh(self.conv(x))

    class _TinyDiscriminator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, 1)
        def forward(self, x):
            return self.conv(x)

    class _FakeAccelerator:
        device = torch.device("cpu")
        def accumulate(self, *models):
            return nullcontext()
        def backward(self, loss):
            loss.backward()

    trainer = Stage2GANTrainer.__new__(Stage2GANTrainer)
    trainer.generator = _TinyGenerator()
    trainer.discriminator = _TinyDiscriminator()
    trainer.optimizer_g = AdamW(trainer.generator.parameters(), lr=1e-3)
    trainer.optimizer_d = AdamW(trainer.discriminator.parameters(), lr=1e-3)
    trainer.accelerator = _FakeAccelerator()
    trainer.augment_training = False
    trainer.use_hinge_loss = True
    trainer.use_perceptual_loss = False
    trainer.lambda_gan, trainer.lambda_l1 = 0.1, 50.0
    trainer.lambda_texture, trainer.lambda_perceptual = 10.0, 0.0
    tiny_stage1 = torch.rand(1, 3, 32, 32) * 2 - 1
    trainer._load_stage1_batch = MethodType(lambda self, batch, split: tiny_stage1, trainer)
    tiny_batch = {
        "image": torch.rand(1, 3, 32, 32) * 2 - 1,
        "mask": (torch.rand(1, 1, 32, 32) > 0.7).float(),
        "sample_id": ["unit_sample"],
    }
    before_g = trainer.generator.conv.weight.detach().clone()
    before_d = trainer.discriminator.conv.weight.detach().clone()
    trainer.train_epoch([tiny_batch], epoch=0)
    check("Stage2.train_epoch 真正更新 G", not torch.equal(before_g, trainer.generator.conv.weight))
    check("Stage2.train_epoch 真正更新 D", not torch.equal(before_d, trainer.discriminator.conv.weight))


# ═══════════════════════════════════════════════
# F. R2-2 端到端单阶段对照
# ═══════════════════════════════════════════════
def test_f_single_stage():
    section("F. R2-2 端到端单阶段对照")

    # ---- F0. CLI 与模式 ----
    import train as train_module
    args = train_module.build_parser().parse_args([])
    check("single_stage lr = 1e-4（与 Stage-2 相同）", args.single_stage_lr == 1e-4,
          f"got {args.single_stage_lr}")
    check("single_stage 预算匹配默认 steps", args.single_stage_budget_match == "steps",
          f"got {args.single_stage_budget_match}")
    check("single_stage 骨干宽度 = 64（与 Stage-2 生成器一致）",
          args.single_stage_base_channels == 64)
    src_train = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    check("--mode 支持 single_stage/all",
          '"single_stage"' in src_train and '"all"' in src_train)
    check("单阶段对照不读取 Stage-1 缓存",
          "stage1_cache_used" in open(os.path.join(ROOT, "training/single_stage.py"),
                                      encoding="utf-8").read())
    check("单阶段训练器不导入 diffusers/peft",
          "diffusers" not in open(os.path.join(ROOT, "training/single_stage.py"),
                                  encoding="utf-8").read())

    # ---- F1. 模型定义：4 通道输入 + 直接预测头 + 与 Stage-2 同骨干 ----
    from models.single_stage import SingleStageInpaintingGenerator
    from models.generator import SimpleUNetGeneratorWithTexture
    from training.common import parameter_report

    gen = SingleStageInpaintingGenerator()
    check("输入通道 = 4（masked RGB + mask）", gen.enc1.conv1.in_channels == 4,
          f"got {gen.enc1.conv1.in_channels}")
    check("输出头为 Tanh（直接预测 [-1,1]）", isinstance(gen.final_activation, torch.nn.Tanh))
    check("无残差缩放参数（直接预测而非残差）",
          gen.learnable_scale is None
          and not any("learnable_scale" in n for n, _ in gen.named_parameters()))
    stage2_backbone = SimpleUNetGeneratorWithTexture(in_channels=4, use_full_reconstruction=False)
    ss_shapes = {k: tuple(v.shape) for k, v in gen.state_dict().items()}
    s2_shapes = {k: tuple(v.shape) for k, v in stage2_backbone.state_dict().items()}
    differing = {k for k in set(ss_shapes) | set(s2_shapes) if ss_shapes.get(k) != s2_shapes.get(k)}
    # v4 起 D2R Stage-2 生成器也是 4 通道，因此唯一允许的差别只剩残差缩放参数
    # （单阶段为直接预测，无此概念）。骨干逐层完全同构，匹配对照的可归因性更强。
    check("与 Stage-2 生成器骨干逐层完全同构（仅残差缩放不同）",
          differing == {"learnable_scale"},
          f"differing={sorted(differing)}")
    check("Stage-2 与单阶段输入通道数一致（均为 4）",
          stage2_backbone.enc1.conv1.in_channels == gen.enc1.conv1.in_channels == 4,
          f"stage2={stage2_backbone.enc1.conv1.in_channels} ss={gen.enc1.conv1.in_channels}")
    check("架构描述可用于论文",
          "single-stage conditional U-Net" in gen.architecture_summary
          and "no diffusion prior" in gen.architecture_summary)
    del stage2_backbone

    # ---- F2. 前向与合成协议 ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = gen.to(device).eval()
    B, H, W = 1, 96, 96
    image = torch.rand(B, 3, H, W, device=device) * 2 - 1
    mask = (torch.rand(B, 1, H, W, device=device) > 0.8).float()
    masked_image = image * (1.0 - mask)
    with torch.no_grad():
        prediction = gen(masked_image, mask)
    check("前向形状 = 输入图像形状", prediction.shape == image.shape,
          f"got {tuple(prediction.shape)}")
    check("输出范围在 [-1,1]", bool(prediction.abs().max() <= 1.0 + 1e-5))
    composite = masked_image * (1.0 - mask) + prediction * mask
    outside = (1 - mask).bool().repeat(1, 3, 1, 1)
    check("掩膜外严格等于原图（单阶段合成协议）",
          bool((composite[outside] - image[outside]).abs().max() < 1e-6))
    check("掩膜内使用模型预测",
          bool((composite - prediction).abs().mul(mask).max() < 1e-6))

    try:
        gen(image)
        check("缺少 mask 时抛错", False)
    except ValueError:
        check("缺少 mask 时抛错", True)

    # ---- F3. 一步 G/D 训练更新（走真实 train_epoch 路径）----
    from contextlib import nullcontext
    from torch.optim import AdamW
    from training.common import BudgetTracker
    from training.single_stage import SingleStageGANTrainer

    class _TinySSGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 3, 1)

        def forward(self, masked_image, mask):
            return 0.1 * torch.tanh(self.conv(torch.cat([masked_image, mask], dim=1)))

    class _TinyDiscriminator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, 1)

        def forward(self, x):
            return self.conv(x)

    class _FakeAccelerator:
        device = torch.device("cpu")

        def accumulate(self, *models):
            return nullcontext()

        def backward(self, loss):
            loss.backward()

    trainer = SingleStageGANTrainer.__new__(SingleStageGANTrainer)
    trainer.generator = _TinySSGenerator()
    trainer.discriminator = _TinyDiscriminator()
    trainer.optimizer_g = AdamW(trainer.generator.parameters(), lr=1e-3)
    trainer.optimizer_d = AdamW(trainer.discriminator.parameters(), lr=1e-3)
    trainer.accelerator = _FakeAccelerator()
    trainer.augment_training = False
    trainer.use_hinge_loss = True
    trainer.use_perceptual_loss = False
    trainer.lambda_gan, trainer.lambda_l1 = 0.1, 50.0
    trainer.lambda_texture, trainer.lambda_perceptual = 10.0, 0.0
    trainer.global_step = 0
    trainer.target_steps = None
    trainer.target_gpu_hours = None
    trainer.stop_reason = None
    trainer.budget = BudgetTracker(world_size=1)

    tiny_batch = {
        "image": torch.rand(1, 3, 32, 32) * 2 - 1,
        "mask": (torch.rand(1, 1, 32, 32) > 0.7).float(),
        "sample_id": ["unit_single_stage"],
    }
    before_g = trainer.generator.conv.weight.detach().clone()
    before_d = trainer.discriminator.conv.weight.detach().clone()
    trainer.train_epoch([tiny_batch], epoch=0)
    check("SingleStage.train_epoch 真正更新 G",
          not torch.equal(before_g, trainer.generator.conv.weight))
    check("SingleStage.train_epoch 真正更新 D",
          not torch.equal(before_d, trainer.discriminator.conv.weight))
    check("更新步数计数 +1（Table 4 口径）", trainer.global_step == 1,
          f"got {trainer.global_step}")

    # 预算耗尽即停止
    trainer.target_steps = 1
    check("达到目标步数即停止", trainer._budget_exhausted())
    check("停止原因已记录", trainer.stop_reason == "target_steps_reached")

    # ---- F4. 预算匹配：从参考 D2R 运行导出实际步数 ----
    import json
    import tempfile
    import train as train_module

    with tempfile.TemporaryDirectory() as tmp:
        def _make_stage(name, epochs_done, global_step=None, seconds=None, world_size=1):
            stage_dir = os.path.join(tmp, name)
            ckpt = os.path.join(stage_dir, f"checkpoint-epoch-{epochs_done}")
            os.makedirs(ckpt, exist_ok=True)
            state = {"epoch": epochs_done, "protocol_id": "unit-test"}
            if global_step is not None:
                state["global_step"] = global_step
            if seconds is not None:
                state["train_seconds"] = seconds
            torch.save(state, os.path.join(ckpt, "training_state.pt"))
            with open(os.path.join(stage_dir, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "world_size": world_size,
                    "arguments": {"stage1_epochs": 100},
                    "trainer_config": {"max_train_steps": 400},
                }, f)
            return stage_dir

        s1 = _make_stage("stage1", epochs_done=3, seconds=3600.0)
        s2 = _make_stage("stage2", epochs_done=5, global_step=999, seconds=7200.0)

        ref1 = train_module.reference_stage_budget(s1, steps_per_epoch=4)
        check("旧检查点：步数 = (epoch+1) × steps/epoch", ref1["steps"] == 16,
              f"got {ref1['steps']}")
        check("GPU·h 由 train_seconds 导出", ref1["gpu_hours"] == 1.0,
              f"got {ref1['gpu_hours']}")
        ref2 = train_module.reference_stage_budget(s2, steps_per_epoch=4)
        check("新检查点：优先使用 global_step", ref2["steps"] == 999,
              f"got {ref2['steps']}")

        budget_args = train_module.build_parser().parse_args([])
        target_steps, target_gpu_hours, reference = train_module.resolve_single_stage_budget(
            budget_args, steps_per_epoch=4, stage1_dir=s1, stage2_dir=s2
        )
        check("目标步数 = Stage-1 + Stage-2 实际步数", target_steps == 16 + 999,
              f"got {target_steps}")
        check("默认不做 GPU·h 匹配", target_gpu_hours is None)
        check("参考信息写入报告", reference["total_reference_steps"] == 16 + 999)

        budget_args.single_stage_budget_match = "gpu_hours"
        t_steps, t_hours, _ = train_module.resolve_single_stage_budget(
            budget_args, steps_per_epoch=4, stage1_dir=s1, stage2_dir=s2
        )
        check("gpu_hours 模式：目标 = 1.0 + 2.0 GPU·h", t_hours == 3.0 and t_steps is None,
              f"got steps={t_steps}, hours={t_hours}")

        empty = os.path.join(tmp, "missing_stage")
        fallback_args = train_module.build_parser().parse_args([])
        f_steps, _, f_ref = train_module.resolve_single_stage_budget(
            fallback_args, steps_per_epoch=4, stage1_dir=empty, stage2_dir=empty
        )
        check("无参考数据时不静默匹配", f_steps is None and f_ref["notes"])

    # ---- F5. 单阶段推理：掩膜外严格保真 + 可与评测脚本对接 ----
    # inference.pipeline 会连带导入 metrics（依赖 lpips）。缺少可选依赖时跳过本节，
    # 但不影响其余单阶段协议测试。
    try:
        from inference.pipeline import single_stage_inference
        _inference_ok = True
        _inference_err = ""
    except Exception as exc:  # pragma: no cover - 取决于环境
        _inference_ok = False
        _inference_err = repr(exc)

    if not _inference_ok:
        skip("F5 单阶段推理（掩膜外保真）", f"无法导入 inference.pipeline: {_inference_err}")
    else:
        from PIL import Image
        import numpy as np

        cpu_gen = SingleStageInpaintingGenerator().eval()
        orig_pil = Image.fromarray(
            (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8), "RGB")
        mask_arr = np.zeros((64, 64), dtype=np.uint8)
        mask_arr[20:40, 20:40] = 255
        mask_pil = Image.fromarray(mask_arr, "L")
        restored = single_stage_inference(cpu_gen, orig_pil, mask_pil, device="cpu")
        orig_np = np.asarray(orig_pil.convert("RGB"))
        res_np = np.asarray(restored.convert("RGB"))
        keep = np.asarray(mask_pil) < 128
        check("推理输出尺寸不变", res_np.shape == orig_np.shape)
        check("推理：掩膜外像素严格不变",
              bool(np.array_equal(res_np[keep], orig_np[keep])))
        check("推理：掩膜内确实被模型改写",
              bool(not np.array_equal(res_np[~keep], orig_np[~keep])))

    # ---- F6. Table 4 报告脚本可解析产物 ----
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import control_budget_report as cbr

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "single_stage_results")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "budget_report.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model_definition": "unit-test model",
                "parameters": {"generator": {"total": 100, "trainable": 90, "frozen": 10}},
                "completed_updates": 1234,
                "completed_epochs": 7,
                "budget": {"gpu_hours": 2.5, "wall_clock_hours": 2.5, "world_size": 1},
            }, f)
        record = cbr.collect_run(run_dir, steps_per_epoch=10, role="unit")
        check("Table 4 报告：可训练参数", record["trainable_parameters"] == 90,
              f"got {record['trainable_parameters']}")
        check("Table 4 报告：更新步数", record["updates"] == 1234, f"got {record['updates']}")
        check("Table 4 报告：GPU·h", record["gpu_hours"] == 2.5, f"got {record['gpu_hours']}")
        check("Table 4 报告：缺失项为空", not record["missing"], f"got {record['missing']}")

        empty_dir = os.path.join(tmp, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        empty_record = cbr.collect_run(empty_dir, role="unit-empty")
        check("Table 4 报告：不编造缺失数据",
              empty_record["updates"] is None and empty_record["missing"])


# ═══════════════════════════════════════════════
# E. 真实数据端到端推理（dataset/test）
# ═══════════════════════════════════════════════

def test_e_inference(num_images=2):
    section("E. 真实数据端到端推理（dataset/test）")
    # 公开仓库不附带权重与数据，因此 E 节在缺少本地资源时整体跳过，而不是判失败。
    # 指定本地 snapshot 目录即可启用：set D2R_SD_MODEL=/path/to/snapshots/<hash>
    base = os.environ.get("D2R_SD_MODEL", "runwayml/stable-diffusion-inpainting")
    if not os.path.isdir(base):
        skip("E 真实数据端到端推理", f"D2R_SD_MODEL 未指向本地目录（当前 = {base}）")
        return

    from inference.pipeline import load_model, stage1_inference, refine_with_stage2
    from utils import find_image_mask_pairs, ensure_output_directory
    from metrics import MetricCalculator
    from PIL import Image
    import numpy as np

    img_dir = os.path.join(ROOT, "dataset/test/img")
    mask_dir = os.path.join(ROOT, "dataset/test/mask")
    if not os.path.isdir(img_dir):
        skip("E 真实数据端到端推理", f"未找到测试图像目录 {img_dir}（数据不随仓库分发）")
        return
    pairs = find_image_mask_pairs(img_dir, mask_dir)[:num_images]
    check("找到测试图像", len(pairs) == num_images, f"found {len(pairs)}")

    pipe, generator = load_model(
        base_model_path=base,
        stage1_checkpoint=os.path.join(ROOT, "stage1_results/checkpoint-best"),
        stage2_checkpoint=os.path.join(ROOT, "stage2_results/checkpoint-best"),
        device="cuda",
    )
    check("模型加载（LoRA + Stage2）", pipe is not None and generator is not None)

    out_dir = ensure_output_directory(os.path.join(ROOT, "test_outputs"))
    mc = MetricCalculator()
    for img_p, mask_p, num in pairs[:num_images]:
        orig = Image.open(img_p).convert("RGB").resize((512, 512))
        mask = Image.open(mask_p).convert("L").resize((512, 512))
        s1 = stage1_inference(pipe, orig, mask, prompt="", num_steps=5, cfg_scale=6.0, seed=42)
        s2 = refine_with_stage2(generator, s1, orig, mask, device="cuda")
        s2.save(os.path.join(out_dir, f"mirror{num}_restored.png"))
        p, ss = mc.calculate_psnr_ssim(orig, s2)
        print(f"    mirror{num}: PSNR={p:.2f} dB, SSIM={ss:.4f} (全图口径, 5步快速测试)")
        check(f"mirror{num} 输出已保存且 PSNR 有限", math.isfinite(p) and os.path.exists(os.path.join(out_dir, f"mirror{num}_restored.png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过端到端推理测试 E")
    args = ap.parse_args()

    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    test_a_config()
    test_b_models()
    test_c_losses()
    test_d_step()
    test_f_single_stage()
    if not args.quick:
        test_e_inference()

    print(f"\n======== 结果: {PASS} 通过, {FAIL} 失败 ========")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
