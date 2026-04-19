#!/usr/bin/env python3
"""
Full experiment v3: oriented EER metrics, larger FF++ sampling, optional EfficientNet,
optional MediaPipe face crops, optional K-fold visual evaluation, CLI overrides.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from src.data.asvspoof_loader import ASVspoofDataset, asvspoof_kwargs_from_config
from src.data.ff_c23_loader import (
    FF_C23_Dataset,
    balanced_sample_pool,
    load_ff_c23_metadata,
    train_val_test_split_indices,
)
from src.data.sampling import load_data_config
from src.evaluation.calibration import best_threshold_min_max_frr_far
from src.evaluation.metrics import compute_all_metrics, metrics_to_jsonable, orient_scores_for_spoof
from src.models.audio_backbone import AudioBackbone
from src.models.fusion import ContrastiveMetricFusion, build_fusion
from src.models.visual_backbone import build_visual_backbone
from src.training.trainer import Trainer
from src.utils.torch_device import get_torch_device


# ── CONFIG (defaults; main() overrides via argparse) ─────────────────────────
DEVICE = get_torch_device()
OUTPUT_DIR = Path("outputs/experiment_v3")
CHARTS_DIR = OUTPUT_DIR / "charts"
RESULTS_DIR = OUTPUT_DIR / "results"

N_SAMPLES = 800
NUM_FRAMES = None
VISUAL_EPOCHS = 20
AUDIO_EPOCHS = 20
FUSION_EPOCHS = 30
BATCH_SIZE = 8
NUM_WORKERS = 0
FUSION_BATCH = 16
# Used when --preset colab or COLAB_RELEASE_TAG is set (unless CLI overrides).
COLAB_DEFAULT_BATCH = 32
COLAB_DEFAULT_NUM_WORKERS = 6
LR = 3e-4
LABEL_SMOOTHING = 0.1
VISUAL_BACKBONE = "resnet18"
VISUAL_LABEL = "Visual (ResNet18)"
FACE_DETECTOR = "haar"
KFOLD = 0
KFOLD_EPOCHS = 10
MODEL_CONFIG_PATH = "config/model_config.yaml"
USE_AMP = False
FUSION_METHOD = "attention"
FUSION_CONTRASTIVE_WEIGHT = 0.3
# cross_dataset: legacy FF++ vs ASVspoof pairing + synthetic mismatches; ff_av: same FaceForensics++ clip
PAIRED_SOURCE = "cross_dataset"


plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11, "figure.facecolor": "white",
})
sns.set_theme(style="whitegrid", palette="muted")


def _loader_kw() -> dict:
    """DataLoader options for throughput: workers + prefetch (Colab/Linux) and pin_memory (CUDA)."""
    kw: dict = {"num_workers": int(NUM_WORKERS)}
    if NUM_WORKERS > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = max(2, min(8, NUM_WORKERS))
    if DEVICE.type == "cuda":
        kw["pin_memory"] = True
    return kw


def _num_frames_for_run() -> int:
    data_cfg = load_data_config()
    return NUM_FRAMES if NUM_FRAMES is not None else int(
        data_cfg["ff_c23"]["frame_extraction"]["num_frames"]
    )


def _apply_colab_loader_defaults(args: argparse.Namespace, preset: str) -> None:
    """Raise batch/workers on Colab when preset is colab or COLAB_RELEASE_TAG is set."""
    global BATCH_SIZE, NUM_WORKERS, FUSION_BATCH

    bs = args.batch_size if args.batch_size is not None else BATCH_SIZE
    nw = args.num_workers if args.num_workers is not None else NUM_WORKERS
    fb = args.fusion_batch_size if args.fusion_batch_size is not None else FUSION_BATCH

    on_colab_vm = bool(os.environ.get("COLAB_RELEASE_TAG"))
    use_colab = preset == "colab" or on_colab_vm
    if use_colab:
        if args.batch_size is None:
            bs = COLAB_DEFAULT_BATCH
        if args.num_workers is None:
            nw = COLAB_DEFAULT_NUM_WORKERS
        if args.fusion_batch_size is None:
            fb = bs

    BATCH_SIZE = max(1, int(bs))
    NUM_WORKERS = max(0, int(nw))
    FUSION_BATCH = max(1, int(fb))


# ── AUDIO ────────────────────────────────────────────────────────────────────
def train_audio_model():
    print("\n" + "=" * 70)
    print("  AUDIO BACKBONE — LCNN on ASVspoof 2019 LA")
    print("=" * 70)

    data_cfg = load_data_config()
    root = data_cfg["paths"]["asvspoof2019"]
    acfg = data_cfg["asvspoof2019"]["audio"]

    n_eff = N_SAMPLES if N_SAMPLES and N_SAMPLES > 0 else None
    ak = asvspoof_kwargs_from_config(acfg)
    train_ds = ASVspoofDataset(
        root=root,
        split="train",
        n_samples=n_eff,
        seed=42,
        **ak,
    )
    dev_ds = ASVspoofDataset(
        root=root,
        split="dev",
        n_samples=n_eff,
        seed=42,
        **ak,
    )
    eval_ds = ASVspoofDataset(
        root=root,
        split="eval",
        n_samples=n_eff,
        seed=42,
        **ak,
    )

    print(f"  Train: {len(train_ds)} | Dev: {len(dev_ds)} | Eval: {len(eval_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **_loader_kw()
    )
    val_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, **_loader_kw())
    test_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, **_loader_kw())

    model = AudioBackbone(embedding_dim=512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=AUDIO_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    out = OUTPUT_DIR / "audio"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        output_dir=str(out),
        scheduler=scheduler,
        use_amp=USE_AMP,
    )

    print(f"\nTraining {AUDIO_EPOCHS} epochs...")
    history = trainer.train(epochs=AUDIO_EPOCHS, patience=12)

    ckpt = torch.load(out / "best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    test_scores, test_labels, test_preds = collect_predictions(model, test_loader)
    test_metrics = compute_all_metrics(test_labels, test_scores)

    return {"history": history, "test_metrics": test_metrics,
            "test_scores": test_scores, "test_labels": test_labels, "test_preds": test_preds}


# ── VISUAL ───────────────────────────────────────────────────────────────────
def train_visual_model():
    print("\n" + "=" * 70)
    print(f"  VISUAL BACKBONE — {VISUAL_LABEL} (frozen layers + augmentation)")
    print("=" * 70)

    data_cfg = load_data_config()
    root = data_cfg["paths"]["ff_c23"]
    csv_dir = data_cfg["ff_c23"]["csv_dir"]
    face_size = data_cfg["ff_c23"]["frame_extraction"]["face_size"]
    nf = _num_frames_for_run()
    n_eff = N_SAMPLES if N_SAMPLES and N_SAMPLES > 0 else None

    print(
        f"Loading data ({nf} frames, face detector={FACE_DETECTOR}, "
        f"n_samples={n_eff or 'all'}, augmentation ON)..."
    )
    train_ds = FF_C23_Dataset(
        root=root,
        csv_dir=csv_dir,
        split="train",
        num_frames=nf,
        face_size=face_size,
        n_samples=n_eff,
        seed=42,
        use_face_detection=True,
        face_detector=FACE_DETECTOR,
    )
    val_ds = FF_C23_Dataset(
        root=root,
        csv_dir=csv_dir,
        split="val",
        num_frames=nf,
        face_size=face_size,
        n_samples=n_eff,
        seed=42,
        use_face_detection=True,
        face_detector=FACE_DETECTOR,
    )
    test_ds = FF_C23_Dataset(
        root=root,
        csv_dir=csv_dir,
        split="test",
        num_frames=nf,
        face_size=face_size,
        n_samples=n_eff,
        seed=42,
        use_face_detection=True,
        face_detector=FACE_DETECTOR,
    )

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **_loader_kw()
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **_loader_kw())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **_loader_kw())

    freeze_vis = 5 if VISUAL_BACKBONE == "efficientnet_b0" else 6
    model = build_visual_backbone(
        backbone=VISUAL_BACKBONE,
        embedding_dim=512,
        pretrained=True,
        freeze_layers=freeze_vis,
        dropout=0.5,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=VISUAL_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    out = OUTPUT_DIR / "visual"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        output_dir=str(out),
        scheduler=scheduler,
        use_amp=USE_AMP,
    )

    print(f"\nTraining {VISUAL_EPOCHS} epochs...")
    history = trainer.train(epochs=VISUAL_EPOCHS, patience=12)

    ckpt = torch.load(out / "best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    test_scores, test_labels, test_preds = collect_predictions(model, test_loader)
    test_metrics = compute_all_metrics(test_labels, test_scores)

    return {
        "history": history,
        "test_metrics": test_metrics,
        "test_scores": test_scores,
        "test_labels": test_labels,
        "test_preds": test_preds,
        "model": model,
        "test_loader": test_loader,
    }


# ── FUSION (real embedding-based) ────────────────────────────────────────────
@torch.no_grad()
def _validate_fusion_contrastive(model, loader, criterion, cw: float):
    model.eval()
    all_scores, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    for v_emb, a_emb, labels in loader:
        v_emb, a_emb, labels = v_emb.to(DEVICE), a_emb.to(DEVICE), labels.to(DEVICE)
        logits, dist = model(v_emb, a_emb)
        loss = criterion(logits, labels) + cw * ContrastiveMetricFusion.contrastive_loss(dist, labels)
        total_loss += float(loss.item())
        n_batches += 1
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        all_scores.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    metrics = compute_all_metrics(all_labels, all_scores)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def _train_contrastive_fusion_loop(model, train_loader, val_loader, out_dir: Path, epochs: int, patience: int, cw: float):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    history = {"train_loss": [], "val_loss": [], "val_eer": [], "val_accuracy": [], "val_auc": []}
    best_eer = 1.0
    no_improve = 0
    last_metrics: dict = {}
    epoch = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tl = 0.0
        nb = 0
        for v_emb, a_emb, labels in train_loader:
            v_emb, a_emb, labels = v_emb.to(DEVICE), a_emb.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits, dist = model(v_emb, a_emb)
            loss = criterion(logits, labels) + cw * ContrastiveMetricFusion.contrastive_loss(dist, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tl += float(loss.item())
            nb += 1
        train_loss = tl / max(nb, 1)
        val_metrics = _validate_fusion_contrastive(model, val_loader, criterion, cw)
        sched.step()
        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_eer"].append(val_metrics["eer"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])
        fn = " [scores↔]" if val_metrics.get("scores_flipped") else ""
        print(
            f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"EER: {val_metrics['eer']:.4f} | AUC: {val_metrics['auc']:.4f} | Acc: {val_metrics['accuracy']:.4f}{fn} | "
            f"{elapsed:.1f}s"
        )
        last_metrics = val_metrics
        if val_metrics["eer"] < best_eer:
            best_eer = val_metrics["eer"]
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "metrics": val_metrics,
                },
                out_dir / "best_model.pt",
            )
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "metrics": last_metrics,
        },
        out_dir / "last_model.pt",
    )
    return history


def train_fusion_model(visual_results, audio_results):
    fusion_method = (FUSION_METHOD or "attention").lower()
    with open(MODEL_CONFIG_PATH) as f:
        mcfg = yaml.safe_load(f)
    fcw = float(mcfg.get("fusion", {}).get("contrastive_weight", FUSION_CONTRASTIVE_WEIGHT))
    if fusion_method not in ("concat", "attention", "contrastive"):
        fusion_method = "attention"

    print("\n" + "=" * 70)
    paired = (PAIRED_SOURCE or "cross_dataset").lower()
    print(f"  FUSION — {fusion_method} + embedding-based training | pairing={paired}")
    print("=" * 70)

    # --- Load pre-trained backbones ---
    visual_model = build_visual_backbone(
        backbone=VISUAL_BACKBONE,
        embedding_dim=512,
        pretrained=False,
        freeze_layers=0,
        dropout=0.5,
    )
    v_ckpt = torch.load(OUTPUT_DIR / "visual" / "best_model.pt",
                         map_location=DEVICE, weights_only=False)
    visual_model.load_state_dict(v_ckpt["model_state_dict"])
    visual_model.to(DEVICE).eval()

    audio_model = AudioBackbone(embedding_dim=512)
    a_ckpt = torch.load(OUTPUT_DIR / "audio" / "best_model.pt",
                         map_location=DEVICE, weights_only=False)
    audio_model.load_state_dict(a_ckpt["model_state_dict"])
    audio_model.to(DEVICE).eval()

    # --- Extract real embeddings ---
    data_cfg = load_data_config()
    acfg = data_cfg["asvspoof2019"]["audio"]
    root_v = data_cfg["paths"]["ff_c23"]
    csv_dir = data_cfg["ff_c23"]["csv_dir"]
    face_size = data_cfg["ff_c23"]["frame_extraction"]["face_size"]

    nf = _num_frames_for_run()
    n_eff = N_SAMPLES if N_SAMPLES and N_SAMPLES > 0 else None

    if paired == "ff_av":
        print("  Same-clip pairing: extracting audio from each FF++ video (requires ffmpeg if torchaudio cannot decode).")
        v_train_ds = FF_C23_Dataset(
            root=root_v,
            csv_dir=csv_dir,
            split="train",
            num_frames=nf,
            face_size=face_size,
            n_samples=n_eff,
            seed=42,
            use_face_detection=True,
            face_detector=FACE_DETECTOR,
            include_audio=True,
            audio_cfg=acfg,
        )
        v_test_ds = FF_C23_Dataset(
            root=root_v,
            csv_dir=csv_dir,
            split="test",
            num_frames=nf,
            face_size=face_size,
            n_samples=n_eff,
            seed=42,
            use_face_detection=True,
            face_detector=FACE_DETECTOR,
            include_audio=True,
            audio_cfg=acfg,
        )
        v_train_loader = DataLoader(v_train_ds, batch_size=BATCH_SIZE, **_loader_kw())
        v_test_loader = DataLoader(v_test_ds, batch_size=BATCH_SIZE, **_loader_kw())
        print("  Extracting paired visual + audio embeddings (FaceForensics++ clips)...")
        train_v_emb, train_a_emb, train_labels = extract_paired_av_embeddings(
            visual_model, audio_model, v_train_loader
        )
        test_v_emb, test_a_emb, test_labels = extract_paired_av_embeddings(
            visual_model, audio_model, v_test_loader
        )
    else:
        print("  Extracting visual embeddings...")
        v_train_ds = FF_C23_Dataset(
            root=root_v,
            csv_dir=csv_dir,
            split="train",
            num_frames=nf,
            face_size=face_size,
            n_samples=n_eff,
            seed=42,
            use_face_detection=True,
            face_detector=FACE_DETECTOR,
        )
        v_test_ds = FF_C23_Dataset(
            root=root_v,
            csv_dir=csv_dir,
            split="test",
            num_frames=nf,
            face_size=face_size,
            n_samples=n_eff,
            seed=42,
            use_face_detection=True,
            face_detector=FACE_DETECTOR,
        )

        v_train_loader = DataLoader(v_train_ds, batch_size=BATCH_SIZE, **_loader_kw())
        v_test_loader = DataLoader(v_test_ds, batch_size=BATCH_SIZE, **_loader_kw())

        v_train_emb, v_train_lbl = extract_embeddings(visual_model, v_train_loader)
        v_test_emb, v_test_lbl = extract_embeddings(visual_model, v_test_loader)

        print("  Extracting audio embeddings (ASVspoof)...")
        root_a = data_cfg["paths"]["asvspoof2019"]
        ak = asvspoof_kwargs_from_config(acfg)
        a_train_ds = ASVspoofDataset(
            root=root_a,
            split="train",
            n_samples=n_eff,
            seed=42,
            **ak,
        )
        a_test_ds = ASVspoofDataset(
            root=root_a,
            split="eval",
            n_samples=n_eff,
            seed=42,
            **ak,
        )

        a_train_loader = DataLoader(a_train_ds, batch_size=BATCH_SIZE, **_loader_kw())
        a_test_loader = DataLoader(a_test_ds, batch_size=BATCH_SIZE, **_loader_kw())

        a_train_emb, a_train_lbl = extract_embeddings(audio_model, a_train_loader)
        a_test_emb, a_test_lbl = extract_embeddings(audio_model, a_test_loader)

        train_v_emb, train_a_emb, train_labels = build_pairs(
            v_train_emb, v_train_lbl, a_train_emb, a_train_lbl
        )
        test_v_emb, test_a_emb, test_labels = build_pairs(
            v_test_emb, v_test_lbl, a_test_emb, a_test_lbl
        )

    print(f"  Paired train: {len(train_labels)} | test: {len(test_labels)}")

    train_dataset = TensorDataset(train_v_emb, train_a_emb, train_labels)
    test_dataset = TensorDataset(test_v_emb, test_a_emb, test_labels)

    # Split train into train/val (80/20)
    n_val = max(int(len(train_dataset) * 0.2), 4)
    n_train = len(train_dataset) - n_val
    train_sub, val_sub = torch.utils.data.random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_sub, batch_size=FUSION_BATCH, shuffle=True, drop_last=True, **_loader_kw()
    )
    val_loader = DataLoader(val_sub, batch_size=FUSION_BATCH, shuffle=False, **_loader_kw())
    test_loader = DataLoader(test_dataset, batch_size=FUSION_BATCH, shuffle=False, **_loader_kw())

    fcfg = mcfg.get("fusion", {})
    hidden_dim = int(fcfg.get("hidden_dim", 256))
    dropout = float(fcfg.get("dropout", 0.3))

    out = OUTPUT_DIR / "fusion"
    out.mkdir(parents=True, exist_ok=True)

    if fusion_method == "contrastive":
        fusion_model = build_fusion(
            "contrastive",
            visual_dim=512,
            audio_dim=512,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(DEVICE)
        print(f"\nTraining contrastive fusion (λ={fcw}) for {FUSION_EPOCHS} epochs...")
        fusion_history = _train_contrastive_fusion_loop(
            fusion_model, train_loader, val_loader, out, FUSION_EPOCHS, 15, fcw
        )
    else:
        fusion_model = build_fusion(
            fusion_method if fusion_method in ("concat", "attention") else "attention",
            visual_dim=512,
            audio_dim=512,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        def fusion_forward(model, batch, device):
            v_emb, a_emb, labels = batch
            v_emb, a_emb, labels = v_emb.to(device), a_emb.to(device), labels.to(device)
            logits = model(v_emb, a_emb)
            return logits, labels

        optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FUSION_EPOCHS)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        trainer = Trainer(
            model=fusion_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            output_dir=str(out),
            scheduler=scheduler,
            forward_fn=fusion_forward,
            use_amp=USE_AMP,
        )

        print(f"\nTraining {fusion_method} fusion for {FUSION_EPOCHS} epochs...")
        fusion_history = trainer.train(epochs=FUSION_EPOCHS, patience=15)

    # --- Evaluate fusion ---
    ckpt = torch.load(out / "best_model.pt", map_location=DEVICE, weights_only=False)
    fusion_model.load_state_dict(ckpt["model_state_dict"])
    fusion_model.to(DEVICE).eval()

    test_scores, test_lbls, test_preds = [], [], []
    with torch.no_grad():
        for v_emb, a_emb, labels in test_loader:
            v_emb, a_emb = v_emb.to(DEVICE), a_emb.to(DEVICE)
            out = fusion_model(v_emb, a_emb)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits, dim=1)[:, 1]
            test_scores.append(probs.cpu().numpy())
            test_lbls.append(labels.numpy())

    test_scores = np.concatenate(test_scores)
    test_lbls = np.concatenate(test_lbls)
    test_metrics = compute_all_metrics(test_lbls, test_scores)
    oriented_f, _, _ = orient_scores_for_spoof(test_lbls, test_scores)
    test_preds = (oriented_f >= test_metrics["eer_threshold"]).astype(int)

    # --- Also compute late fusion for comparison ---
    v_sc = visual_results["test_scores"]
    a_sc = audio_results["test_scores"]
    n = min(len(v_sc), len(a_sc))

    late_results = {}
    best_late_eer = 1.0
    best_w = 0.5
    for w in np.arange(0.0, 1.05, 0.1):
        fused = w * v_sc[:n] + (1 - w) * a_sc[:n]
        lbl = np.asarray(visual_results["test_labels"])[:n]
        m = compute_all_metrics(lbl, fused)
        late_results[f"{w:.1f}"] = m
        if m["eer"] < best_late_eer:
            best_late_eer = m["eer"]
            best_w = w

    print(f"\n  Fusion ({fusion_method}) EER: {test_metrics['eer']:.4f}, AUC: {test_metrics['auc']:.4f}")
    print(f"  Late Fusion best EER: {best_late_eer:.4f} (w_visual={best_w:.1f})")

    return {
        "history": fusion_history,
        "test_metrics": test_metrics,
        "test_scores": test_scores,
        "test_labels": test_lbls,
        "test_preds": test_preds,
        "late_fusion": late_results,
        "late_best_weight": best_w,
        "late_best_metrics": late_results.get(f"{best_w:.1f}", {}),
    }


# ── HELPERS ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    all_scores, all_labels = [], []
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_scores.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    m = compute_all_metrics(labels, scores)
    oriented, _, _ = orient_scores_for_spoof(labels, scores)
    preds = (oriented >= m["eer_threshold"]).astype(int)
    return oriented, labels, preds


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    embs, labels = [], []
    for inputs, lbl in loader:
        inputs = inputs.to(DEVICE)
        emb = model.extract_features(inputs)
        embs.append(emb.cpu())
        labels.append(lbl)
    return torch.cat(embs), torch.cat(labels)


@torch.no_grad()
def extract_paired_av_embeddings(visual_model, audio_model, loader):
    """Visual + audio embeddings from the same batch (e.g. FF++ clip with ``include_audio``)."""
    visual_model.eval()
    audio_model.eval()
    v_embs, a_embs, labels = [], [], []
    for batch in loader:
        if len(batch) == 3:
            v_in, a_in, lbl = batch
        else:
            raise ValueError("Expected (video, audio, label) batches for paired AV extraction.")
        v_in = v_in.to(DEVICE)
        a_in = a_in.to(DEVICE)
        v_e = visual_model.extract_features(v_in)
        a_e = audio_model.extract_features(a_in)
        v_embs.append(v_e.cpu())
        a_embs.append(a_e.cpu())
        labels.append(lbl)
    return torch.cat(v_embs), torch.cat(a_embs), torch.cat(labels)


def build_pairs(v_emb, v_lbl, a_emb, a_lbl):
    """Pair visual and audio embeddings by label for fusion training."""
    v_real = v_emb[v_lbl == 0]
    v_fake = v_emb[v_lbl == 1]
    a_real = a_emb[a_lbl == 0]
    a_fake = a_emb[a_lbl == 1]

    pairs_v, pairs_a, pairs_lbl = [], [], []

    # Genuine pairs: real visual + bonafide audio -> label 0
    n_gen = min(len(v_real), len(a_real))
    pairs_v.append(v_real[:n_gen])
    pairs_a.append(a_real[:n_gen])
    pairs_lbl.append(torch.zeros(n_gen, dtype=torch.long))

    # Attack pairs: fake visual + spoof audio -> label 1
    n_atk = min(len(v_fake), len(a_fake))
    pairs_v.append(v_fake[:n_atk])
    pairs_a.append(a_fake[:n_atk])
    pairs_lbl.append(torch.ones(n_atk, dtype=torch.long))

    # Cross-modal mismatch: real visual + spoof audio -> label 1
    n_cross1 = min(len(v_real), len(a_fake), n_gen)
    if n_cross1 > 0:
        pairs_v.append(v_real[:n_cross1])
        pairs_a.append(a_fake[:n_cross1])
        pairs_lbl.append(torch.ones(n_cross1, dtype=torch.long))

    # Cross-modal mismatch: fake visual + bonafide audio -> label 1
    n_cross2 = min(len(v_fake), len(a_real), n_gen)
    if n_cross2 > 0:
        pairs_v.append(v_fake[:n_cross2])
        pairs_a.append(a_real[:n_cross2])
        pairs_lbl.append(torch.ones(n_cross2, dtype=torch.long))

    return torch.cat(pairs_v), torch.cat(pairs_a), torch.cat(pairs_lbl)


# ── CHARTS ───────────────────────────────────────────────────────────────────
def generate_charts(audio, visual, fusion):
    print("\n" + "=" * 70)
    print("  GENERATING CHARTS")
    print("=" * 70)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_training_curves(audio["history"], "Audio (LCNN)", CHARTS_DIR / "audio_training.png")
    plot_training_curves(visual["history"], VISUAL_LABEL, CHARTS_DIR / "visual_training.png")
    plot_training_curves(fusion["history"], "Attention Fusion", CHARTS_DIR / "fusion_training.png")
    plot_combined(audio["history"], visual["history"], CHARTS_DIR / "combined_training.png")
    plot_roc(audio, visual, fusion, CHARTS_DIR / "roc_curves.png")
    plot_confusion(audio, visual, fusion, CHARTS_DIR / "confusion_matrices.png")
    plot_metrics(audio, visual, fusion, CHARTS_DIR / "metric_comparison.png")
    plot_scores(audio, visual, CHARTS_DIR / "score_distributions.png")
    plot_eer(audio, visual, fusion, CHARTS_DIR / "eer_comparison.png")
    plot_dashboard(audio, visual, fusion, CHARTS_DIR / "summary_dashboard.png")

    # Comparison with v1 (if available)
    v1_path = Path("outputs/full_experiment/results/experiment_summary.json")
    if v1_path.exists():
        plot_v1_comparison(audio, visual, fusion, v1_path, CHARTS_DIR / "v1_vs_v2_comparison.png")

    print(f"  All charts saved to {CHARTS_DIR}/")


def plot_training_curves(h, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ep = range(1, len(h["train_loss"]) + 1)

    axes[0].plot(ep, h["train_loss"], "o-", label="Train", color="#2196F3", lw=2, ms=4)
    axes[0].plot(ep, h["val_loss"], "s-", label="Val", color="#F44336", lw=2, ms=4)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{title} — Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, [e*100 for e in h["val_eer"]], "D-", color="#FF9800", lw=2, ms=4)
    axes[1].set(xlabel="Epoch", ylabel="EER (%)", title=f"{title} — EER")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ep, [a*100 for a in h["val_accuracy"]], "^-", label="Acc", color="#4CAF50", lw=2, ms=4)
    axes[2].plot(ep, [a*100 for a in h["val_auc"]], "v-", label="AUC", color="#9C27B0", lw=2, ms=4)
    axes[2].set(xlabel="Epoch", ylabel="%", title=f"{title} — Accuracy & AUC")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Training Curves — {title}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_combined(ah, vh, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ae, ve = range(1, len(ah["train_loss"])+1), range(1, len(vh["train_loss"])+1)

    for ax, key, title in [
        (axes[0,0], "train_loss", "Training Loss"), (axes[0,1], "val_loss", "Validation Loss"),
        (axes[1,0], "val_eer", "Validation EER (%)"), (axes[1,1], "val_auc", "Validation AUC (%)"),
    ]:
        mul = 100 if "eer" in key or "auc" in key else 1
        ax.plot(ae, [v*mul for v in ah[key]], "o-", label="Audio", color="#2196F3", lw=2, ms=4)
        ax.plot(ve, [v*mul for v in vh[key]], "s-", label="Visual", color="#F44336", lw=2, ms=4)
        ax.set(title=title, xlabel="Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("Audio vs Visual Training", fontsize=15, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_roc(audio, visual, fusion, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    for r, name, color, ls in [
        (audio, "Audio (LCNN)", "#2196F3", "-"),
        (visual, VISUAL_LABEL, "#F44336", "--"),
        (fusion, "Attention Fusion", "#4CAF50", "-."),
    ]:
        fpr, tpr, _ = roc_curve(r["test_labels"], r["test_scores"])
        ax.plot(fpr, tpr, color=color, lw=2.5, ls=ls,
                label=f"{name} (AUC={r['test_metrics']['auc']:.3f})")
    ax.plot([0,1],[0,1], "k--", alpha=0.3)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curves", xlim=[0,1], ylim=[0,1.02])
    ax.legend(loc="lower right", fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_confusion(audio, visual, fusion, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = ["Genuine", "Spoof"]
    for ax, r, t in [(axes[0], audio, "Audio"), (axes[1], visual, "Visual"), (axes[2], fusion, "Att. Fusion")]:
        cm = confusion_matrix(r["test_labels"], r["test_preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=names, yticklabels=names, cbar_kws={"shrink": 0.8})
        ax.set(xlabel="Predicted", ylabel="Actual", title=t)
    plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_metrics(audio, visual, fusion, path):
    names = ["EER (%)", "Accuracy (%)", "AUC (%)", "min t-DCF"]
    vals = {}
    for r, lbl in [(audio, "Audio"), (visual, "Visual"), (fusion, "Fusion")]:
        m = r["test_metrics"]
        vals[lbl] = [m["eer"]*100, m["accuracy"]*100, m["auc"]*100, m["min_tdcf"]*100]

    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"Audio": "#2196F3", "Visual": "#F44336", "Fusion": "#4CAF50"}
    for i, (lbl, v) in enumerate(vals.items()):
        bars = ax.bar(x + (i-1)*w, v, w, label=lbl, color=colors[lbl], edgecolor="black", lw=0.5)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                    f"{b.get_height():.1f}", ha="center", fontsize=9, fontweight="bold")

    ax.set(ylabel="Value (%)", title="Metric Comparison"); ax.set_xticks(x); ax.set_xticklabels(names)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_scores(audio, visual, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, r, t, c in [
        (axes[0], audio, "Audio Scores", ("#2196F3", "#F44336")),
        (axes[1], visual, "Visual Scores", ("#4CAF50", "#FF9800")),
    ]:
        gen = r["test_scores"][r["test_labels"] == 0]
        spf = r["test_scores"][r["test_labels"] == 1]
        ax.hist(gen, bins=25, alpha=0.6, color=c[0], label="Genuine", edgecolor="black", lw=0.5)
        ax.hist(spf, bins=25, alpha=0.6, color=c[1], label="Spoof", edgecolor="black", lw=0.5)
        th = r["test_metrics"]["eer_threshold"]
        ax.axvline(th, color="black", ls="--", lw=2, label=f"Threshold={th:.3f}")
        ax.set(xlabel="Score P(spoof)", ylabel="Count", title=t); ax.legend()
    plt.suptitle("Score Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_eer(audio, visual, fusion, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    names = ["Audio\n(LCNN)", VISUAL_LABEL.replace(" ", "\n", 1), "Attention\nFusion"]
    eers = [r["test_metrics"]["eer"]*100 for r in [audio, visual, fusion]]
    colors = ["#2196F3", "#F44336", "#4CAF50"]
    bars = ax.bar(names, eers, color=colors, edgecolor="black", lw=0.5, width=0.5)
    for b, e in zip(bars, eers):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                f"{e:.2f}%", ha="center", fontsize=13, fontweight="bold")
    ax.set(ylabel="EER (%)", title="EER Comparison"); ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(eers)*1.3)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_dashboard(audio, visual, fusion, path):
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    ae = range(1, len(audio["history"]["train_loss"])+1)
    ve = range(1, len(visual["history"]["train_loss"])+1)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ae, audio["history"]["train_loss"], "o-", label="Audio", color="#2196F3", ms=4)
    ax.plot(ve, visual["history"]["train_loss"], "s-", label="Visual", color="#F44336", ms=4)
    ax.set_title("Training Loss"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ae, [e*100 for e in audio["history"]["val_eer"]], "o-", label="Audio", color="#2196F3", ms=4)
    ax.plot(ve, [e*100 for e in visual["history"]["val_eer"]], "s-", label="Visual", color="#F44336", ms=4)
    ax.set_title("Val EER (%)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    for r, n, c in [(audio, "Audio", "#2196F3"), (visual, "Visual", "#F44336"), (fusion, "Fusion", "#4CAF50")]:
        fpr, tpr, _ = roc_curve(r["test_labels"], r["test_scores"])
        ax.plot(fpr, tpr, color=c, lw=2, label=f'{n} ({r["test_metrics"]["auc"]:.3f})')
    ax.plot([0,1],[0,1], "k--", alpha=0.3); ax.set_title("ROC Curves"); ax.legend(fontsize=9)

    names = ["Genuine", "Spoof"]
    for i, (r, t) in enumerate([(audio, "Audio CM"), (visual, "Visual CM"), (fusion, "Fusion CM")]):
        ax = fig.add_subplot(gs[1, i])
        cm = confusion_matrix(r["test_labels"], r["test_preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=names, yticklabels=names)
        ax.set(title=t, xlabel="Predicted", ylabel="Actual")

    ax = fig.add_subplot(gs[2, 0:2])
    mnames = ["EER (%)", "Accuracy (%)", "AUC (%)"]
    x = np.arange(3); w = 0.25
    for off, (r, n, c) in enumerate([(audio,"Audio","#2196F3"),(visual,"Visual","#F44336"),(fusion,"Fusion","#4CAF50")]):
        v = [r["test_metrics"]["eer"]*100, r["test_metrics"]["accuracy"]*100, r["test_metrics"]["auc"]*100]
        bars = ax.bar(x+(off-1)*w, v, w, label=n, color=c, edgecolor="black", lw=0.5)
        for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{b.get_height():.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(mnames); ax.set_title("Metrics"); ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[2, 2])
    for r, n, c in [("audio","Audio","#2196F3"),("visual","Visual","#F44336")]:
        rr = audio if r=="audio" else visual
        ax.hist(rr["test_scores"][rr["test_labels"]==0], bins=20, alpha=0.4, color=c, label=f"{n} Gen")
        ax.hist(rr["test_scores"][rr["test_labels"]==1], bins=20, alpha=0.4, color=c, histtype="step", lw=2, ls="--", label=f"{n} Spf")
    ax.set_title("Score Distributions"); ax.legend(fontsize=8)

    fig.suptitle("Multi-Modal Liveness Detection — v3 Dashboard", fontsize=18, fontweight="bold", y=0.98)
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_v1_comparison(audio, visual, fusion, v1_path, path):
    """Compare v2 results with v1 baseline."""
    with open(v1_path) as f:
        v1 = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["eer", "accuracy", "auc"]
    titles = ["EER (%) ↓ better", "Accuracy (%) ↑ better", "AUC (%) ↑ better"]

    for ax, metric, title in zip(axes, metrics, titles):
        v1_vals = [v1[f"{m}_test_metrics"][metric]*100 for m in ["audio", "visual", "fusion"]]
        v2_vals = [r["test_metrics"][metric]*100 for r in [audio, visual, fusion]]
        names = ["Audio", "Visual", "Fusion"]
        x = np.arange(3); w = 0.35

        b1 = ax.bar(x - w/2, v1_vals, w, label="v1 (baseline)", color="#BBDEFB", edgecolor="black", lw=0.5)
        b2 = ax.bar(x + w/2, v2_vals, w, label="v2 (improved)", color="#1565C0", edgecolor="black", lw=0.5)

        for bars in [b1, b2]:
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                        f"{b.get_height():.1f}", ha="center", fontsize=9)

        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("v1 Baseline vs v2 Improved — Side-by-Side", fontsize=16, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def run_visual_kfold():
    """K-fold on train+val pool; reports mean ± std of best validation EER per fold."""
    print("\n" + "=" * 70)
    print(f"  VISUAL K-FOLD ({KFOLD} folds, {KFOLD_EPOCHS} epochs/fold)")
    print("=" * 70)

    data_cfg = load_data_config()
    root = data_cfg["paths"]["ff_c23"]
    csv_dir = data_cfg["ff_c23"]["csv_dir"]
    face_size = data_cfg["ff_c23"]["frame_extraction"]["face_size"]
    nf = _num_frames_for_run()
    n_eff = N_SAMPLES if N_SAMPLES and N_SAMPLES > 0 else None

    pool = balanced_sample_pool(
        load_ff_c23_metadata(Path(root), Path(csv_dir)), n_eff, seed=42
    )
    if len(pool) < KFOLD * 8:
        print(f"  Skipping K-fold: need at least {KFOLD * 8} samples, got {len(pool)}")
        return None

    train_idx, val_idx, _test_idx = train_val_test_split_indices(len(pool), 42, 0.6, 0.2)
    tv_samples = [pool[i] for i in train_idx + val_idx]
    if len(tv_samples) < KFOLD * 4:
        print("  Skipping K-fold: train+val pool too small")
        return None

    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
    fold_eers = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(len(tv_samples)))):
        train_s = [tv_samples[j] for j in tr_i]
        val_s = [tv_samples[j] for j in va_i]
        train_ds = FF_C23_Dataset(
            root=root,
            csv_dir=csv_dir,
            split="train",
            explicit_samples=train_s,
            num_frames=nf,
            face_size=face_size,
            use_face_detection=True,
            face_detector=FACE_DETECTOR,
        )
        val_ds = FF_C23_Dataset(
            root=root,
            csv_dir=csv_dir,
            split="val",
            explicit_samples=val_s,
            num_frames=nf,
            face_size=face_size,
            use_face_detection=True,
            face_detector=FACE_DETECTOR,
        )
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **_loader_kw()
        )
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **_loader_kw())

        freeze_vis = 5 if VISUAL_BACKBONE == "efficientnet_b0" else 6
        model = build_visual_backbone(
            backbone=VISUAL_BACKBONE,
            embedding_dim=512,
            pretrained=True,
            freeze_layers=freeze_vis,
            dropout=0.5,
        )
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            weight_decay=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=KFOLD_EPOCHS)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        out = OUTPUT_DIR / "visual_kfold" / f"fold_{fold}"
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            output_dir=str(out),
            scheduler=scheduler,
            use_amp=USE_AMP,
        )
        trainer.train(epochs=KFOLD_EPOCHS, patience=5)
        best_eer = min(trainer.history["val_eer"])
        fold_eers.append(best_eer)
        print(f"  Fold {fold + 1}/{KFOLD}: best val EER = {best_eer:.4f}")

    mean_eer = float(np.mean(fold_eers))
    std_eer = float(np.std(fold_eers))
    print(f"\n  K-fold val EER: {mean_eer:.4f} ± {std_eer:.4f}")
    return {"k": KFOLD, "epochs_per_fold": KFOLD_EPOCHS, "fold_best_val_eers": fold_eers,
            "mean_best_val_eer": mean_eer, "std_best_val_eer": std_eer}


# ── MAIN ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Full multi-modal liveness experiment (v3)")
    p.add_argument("--output-dir", default="outputs/experiment_v3", type=str)
    p.add_argument(
        "--n-samples",
        type=int,
        default=800,
        help="Per-modality cap (balanced). Use 0 for full available pool.",
    )
    p.add_argument("--num-frames", type=int, default=None, help="Override config num_frames")
    p.add_argument(
        "--visual-backbone",
        choices=["resnet18", "resnet34", "efficientnet_b0"],
        default=None,
        help="Default: config/model_config.yaml visual.backbone",
    )
    p.add_argument("--face-detector", choices=["haar", "mediapipe"], default="haar")
    p.add_argument("--model-config", default="config/model_config.yaml")
    p.add_argument("--kfold", type=int, default=0, help="If >0, run visual K-fold after main train")
    p.add_argument("--kfold-epochs", type=int, default=10)
    p.add_argument("--visual-epochs", type=int, default=None)
    p.add_argument("--audio-epochs", type=int, default=None)
    p.add_argument("--fusion-epochs", type=int, default=None)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Audio/visual DataLoader batch (default {BATCH_SIZE}, or {COLAB_DEFAULT_BATCH} on Colab when "
        "--preset colab or COLAB_RELEASE_TAG is set). Use --amp on CUDA; lower if OOM.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=f"DataLoader workers (default {NUM_WORKERS}, or {COLAB_DEFAULT_NUM_WORKERS} on Colab when "
        "--preset colab or COLAB_RELEASE_TAG is set).",
    )
    p.add_argument(
        "--fusion-batch-size",
        type=int,
        default=None,
        help=f"Fusion DataLoader batch (default {FUSION_BATCH}, or same as batch-size on Colab when "
        "--preset colab or COLAB_RELEASE_TAG is set).",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision (no-op on CPU)",
    )
    p.add_argument(
        "--preset",
        choices=["none", "strong", "production", "colab"],
        default="none",
        help="strong: full pool + MediaPipe + mild label smoothing + 12 frames if omitted; "
        "production: same + 50 epochs/modality if epochs not set (long FF++ run; use GPU + --amp on CUDA); "
        "colab: same as production + larger DataLoader defaults (also when COLAB_RELEASE_TAG is set)",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=None,
        help="Override CrossEntropy label smoothing (default module constant or preset)",
    )
    p.add_argument(
        "--fusion-method",
        choices=["attention", "contrastive", "concat"],
        default=None,
        help="Override config/model_config.yaml fusion.method",
    )
    p.add_argument(
        "--paired-source",
        choices=["cross_dataset", "ff_av"],
        default=None,
        help="Fusion embedding pairing: cross_dataset (FF++ vs ASVspoof + synthetic mismatches) or "
        "ff_av (same FaceForensics++ clip: face + soundtrack). Default: model_config fusion.paired_source",
    )
    return p.parse_args()


def main():
    global OUTPUT_DIR, CHARTS_DIR, RESULTS_DIR, N_SAMPLES, NUM_FRAMES
    global VISUAL_BACKBONE, VISUAL_LABEL, FACE_DETECTOR, KFOLD, KFOLD_EPOCHS
    global VISUAL_EPOCHS, AUDIO_EPOCHS, FUSION_EPOCHS, MODEL_CONFIG_PATH, USE_AMP, LABEL_SMOOTHING
    global FUSION_METHOD, FUSION_CONTRASTIVE_WEIGHT, PAIRED_SOURCE
    global BATCH_SIZE, NUM_WORKERS, FUSION_BATCH

    args = parse_args()
    num_frames_arg = args.num_frames
    OUTPUT_DIR = Path(args.output_dir)
    CHARTS_DIR = OUTPUT_DIR / "charts"
    RESULTS_DIR = OUTPUT_DIR / "results"
    N_SAMPLES = None if args.n_samples == 0 else args.n_samples
    NUM_FRAMES = num_frames_arg
    FACE_DETECTOR = args.face_detector
    KFOLD = args.kfold
    KFOLD_EPOCHS = args.kfold_epochs
    MODEL_CONFIG_PATH = args.model_config

    with open(MODEL_CONFIG_PATH) as f:
        mcfg = yaml.safe_load(f)
    vb = (args.visual_backbone or mcfg.get("visual", {}).get("backbone", "resnet18")).lower()
    VISUAL_BACKBONE = vb.replace("-", "_")
    if VISUAL_BACKBONE == "efficientnet_b0":
        VISUAL_LABEL = "Visual (EfficientNet-B0)"
    elif VISUAL_BACKBONE == "resnet34":
        VISUAL_LABEL = "Visual (ResNet34)"
    else:
        VISUAL_LABEL = "Visual (ResNet18)"
    FUSION_METHOD = args.fusion_method or mcfg.get("fusion", {}).get("method", "attention")
    FUSION_CONTRASTIVE_WEIGHT = float(mcfg.get("fusion", {}).get("contrastive_weight", 0.3))
    PAIRED_SOURCE = args.paired_source or mcfg.get("fusion", {}).get("paired_source", "cross_dataset")
    if args.visual_epochs:
        VISUAL_EPOCHS = args.visual_epochs
    if args.audio_epochs:
        AUDIO_EPOCHS = args.audio_epochs
    if args.fusion_epochs:
        FUSION_EPOCHS = args.fusion_epochs
    USE_AMP = bool(args.amp)

    preset = args.preset
    if preset == "strong":
        N_SAMPLES = None
        FACE_DETECTOR = "mediapipe"
        LABEL_SMOOTHING = 0.05
        if num_frames_arg is None:
            NUM_FRAMES = 12
    elif preset in ("production", "colab"):
        N_SAMPLES = None
        FACE_DETECTOR = "mediapipe"
        LABEL_SMOOTHING = 0.05
        if num_frames_arg is None:
            NUM_FRAMES = 12
        if args.visual_epochs is None:
            VISUAL_EPOCHS = 50
        if args.audio_epochs is None:
            AUDIO_EPOCHS = 50
        if args.fusion_epochs is None:
            FUSION_EPOCHS = 50
    if args.label_smoothing is not None:
        LABEL_SMOOTHING = float(args.label_smoothing)

    _apply_colab_loader_defaults(args, preset)

    print("=" * 70)
    print("  MULTI-MODAL LIVENESS DETECTION — EXPERIMENT v3")
    print(f"  Device: {DEVICE} | n_samples={N_SAMPLES or 'all'} | frames={_num_frames_for_run()}")
    print(
        f"  DataLoader: batch_size={BATCH_SIZE} | num_workers={NUM_WORKERS} | "
        f"fusion_batch={FUSION_BATCH}"
    )
    _on_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))
    if (preset == "colab" or _on_colab) and (
        args.batch_size is None or args.num_workers is None or args.fusion_batch_size is None
    ):
        print(
            "  Colab throughput: applied defaults for unset --batch-size / --num-workers / "
            "--fusion-batch-size (override any flag to replace)"
        )
    print(
        f"  Visual: {VISUAL_BACKBONE} | face_detector={FACE_DETECTOR} | "
        f"label_smoothing={LABEL_SMOOTHING} | preset={preset} | fusion_pairing={PAIRED_SOURCE} | AMP={USE_AMP}"
    )
    print("=" * 70)

    for d in [OUTPUT_DIR, CHARTS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    start = time.time()

    t0 = time.time()
    audio_results = train_audio_model()
    audio_time = time.time() - t0
    print(f"\n  Audio done in {audio_time:.1f}s")

    t0 = time.time()
    visual_results = train_visual_model()
    visual_time = time.time() - t0
    print(f"\n  Visual done in {visual_time:.1f}s")

    kfold_summary = None
    if KFOLD > 0:
        t0 = time.time()
        kfold_summary = run_visual_kfold()
        print(f"\n  K-fold done in {time.time() - t0:.1f}s")

    vis_for_fusion = {k: v for k, v in visual_results.items() if k not in ("model", "test_loader")}

    t0 = time.time()
    fusion_results = train_fusion_model(vis_for_fusion, audio_results)
    fusion_time = time.time() - t0
    print(f"\n  Fusion done in {fusion_time:.1f}s")

    generate_charts(audio_results, visual_results, fusion_results)

    total_time = time.time() - start

    _dc = load_data_config()
    _ac = _dc["asvspoof2019"]["audio"]
    summary = {
        "experiment": {
            "device": str(DEVICE),
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "fusion_batch_size": FUSION_BATCH,
            "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
            "n_samples": N_SAMPLES,
            "num_frames": _num_frames_for_run(),
            "visual_backbone": VISUAL_BACKBONE,
            "audio_feature": _ac.get("feature_type", "lfcc"),
            "fusion_method": FUSION_METHOD,
            "fusion_contrastive_weight": FUSION_CONTRASTIVE_WEIGHT,
            "paired_source": PAIRED_SOURCE,
            "face_detector": FACE_DETECTOR,
            "label_smoothing": LABEL_SMOOTHING,
            "preset": preset,
            "visual_epochs": VISUAL_EPOCHS,
            "audio_epochs": AUDIO_EPOCHS,
            "fusion_epochs": FUSION_EPOCHS,
            "total_time": round(total_time, 1),
            "oriented_eer_metrics": True,
            "use_amp": USE_AMP,
        },
        "audio_test_metrics": metrics_to_jsonable(audio_results["test_metrics"]),
        "visual_test_metrics": metrics_to_jsonable(visual_results["test_metrics"]),
        "fusion_test_metrics": metrics_to_jsonable(fusion_results["test_metrics"]),
    }
    if kfold_summary:
        summary["visual_kfold"] = kfold_summary

    thr_opt = {}
    for name, res in (
        ("audio", audio_results),
        ("visual", visual_results),
        ("fusion", fusion_results),
    ):
        row = best_threshold_min_max_frr_far(res["test_labels"], res["test_scores"])
        if row:
            thr_opt[name] = row
    if thr_opt:
        summary["threshold_minmax_frr_far"] = thr_opt

    with open(RESULTS_DIR / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  FINAL RESULTS (v3)")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Audio (LCNN)':<18} {VISUAL_LABEL:<22} {'Att. Fusion':<18}")
    print("-" * 80)
    for metric in ["eer", "accuracy", "auc", "min_tdcf"]:
        a = audio_results["test_metrics"][metric]
        v = visual_results["test_metrics"][metric]
        f_val = fusion_results["test_metrics"][metric]
        if metric != "min_tdcf":
            a_s, v_s, f_s = f"{a:.2%}", f"{v:.2%}", f"{f_val:.2%}"
        else:
            a_s, v_s, f_s = f"{a:.4f}", f"{v:.4f}", f"{f_val:.4f}"
        print(f"{metric.upper():<20} {a_s:<18} {v_s:<22} {f_s:<18}")

    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Charts: {CHARTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
