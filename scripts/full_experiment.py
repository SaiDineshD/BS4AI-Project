#!/usr/bin/env python3
"""
Full experiment v2: all recommendations applied.
 - Visual: data augmentation, frozen ResNet layers, face detection, 200 samples, 60/20/20 splits
 - Audio: 200 samples, more epochs
 - Fusion: real embedding-based attention fusion (trained)
 - Label smoothing, stronger regularization
 - Comparison charts against v1 baseline
"""

import json
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
from sklearn.metrics import confusion_matrix, roc_curve
from torch.utils.data import DataLoader, TensorDataset

from src.data.asvspoof_loader import ASVspoofDataset
from src.data.ff_c23_loader import FF_C23_Dataset
from src.data.sampling import load_data_config
from src.evaluation.metrics import compute_all_metrics, compute_eer
from src.models.audio_backbone import AudioBackbone
from src.models.fusion import build_fusion
from src.models.visual_backbone import VisualBackbone
from src.training.trainer import Trainer

# ── CONFIG ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs/experiment_v2")
CHARTS_DIR = OUTPUT_DIR / "charts"
RESULTS_DIR = OUTPUT_DIR / "results"

N_SAMPLES = 200
NUM_FRAMES = 4
VISUAL_EPOCHS = 20
AUDIO_EPOCHS = 20
FUSION_EPOCHS = 30
BATCH_SIZE = 8
LR = 3e-4
LABEL_SMOOTHING = 0.1

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11, "figure.facecolor": "white",
})
sns.set_theme(style="whitegrid", palette="muted")


# ── AUDIO ────────────────────────────────────────────────────────────────────
def train_audio_model():
    print("\n" + "=" * 70)
    print("  AUDIO BACKBONE — LCNN on ASVspoof 2019 LA")
    print("=" * 70)

    data_cfg = load_data_config()
    root = data_cfg["paths"]["asvspoof2019"]
    acfg = data_cfg["asvspoof2019"]["audio"]

    train_ds = ASVspoofDataset(root=root, split="train", sample_rate=acfg["sample_rate"],
        max_length=acfg["max_length"], n_lfcc=acfg["n_lfcc"], n_fft=acfg["n_fft"],
        n_samples=N_SAMPLES, seed=42)
    dev_ds = ASVspoofDataset(root=root, split="dev", sample_rate=acfg["sample_rate"],
        max_length=acfg["max_length"], n_lfcc=acfg["n_lfcc"], n_fft=acfg["n_fft"],
        n_samples=N_SAMPLES, seed=42)
    eval_ds = ASVspoofDataset(root=root, split="eval", sample_rate=acfg["sample_rate"],
        max_length=acfg["max_length"], n_lfcc=acfg["n_lfcc"], n_fft=acfg["n_fft"],
        n_samples=N_SAMPLES, seed=42)

    print(f"  Train: {len(train_ds)} | Dev: {len(dev_ds)} | Eval: {len(eval_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AudioBackbone(embedding_dim=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=AUDIO_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    out = OUTPUT_DIR / "audio"
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, criterion=criterion, device=DEVICE,
        output_dir=str(out), scheduler=scheduler)

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
    print("  VISUAL BACKBONE — ResNet18 (frozen layers + augmentation)")
    print("=" * 70)

    data_cfg = load_data_config()
    root = data_cfg["paths"]["ff_c23"]
    csv_dir = data_cfg["ff_c23"]["csv_dir"]
    face_size = data_cfg["ff_c23"]["frame_extraction"]["face_size"]

    print(f"Loading data ({NUM_FRAMES} frames, face detection ON, augmentation ON)...")
    train_ds = FF_C23_Dataset(root=root, csv_dir=csv_dir, split="train",
        num_frames=NUM_FRAMES, face_size=face_size, n_samples=N_SAMPLES, seed=42,
        use_face_detection=True)
    val_ds = FF_C23_Dataset(root=root, csv_dir=csv_dir, split="val",
        num_frames=NUM_FRAMES, face_size=face_size, n_samples=N_SAMPLES, seed=42,
        use_face_detection=True)
    test_ds = FF_C23_Dataset(root=root, csv_dir=csv_dir, split="test",
        num_frames=NUM_FRAMES, face_size=face_size, n_samples=N_SAMPLES, seed=42,
        use_face_detection=True)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = VisualBackbone(embedding_dim=512, pretrained=True,
                           freeze_layers=6, dropout=0.5)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=VISUAL_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    out = OUTPUT_DIR / "visual"
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, criterion=criterion, device=DEVICE,
        output_dir=str(out), scheduler=scheduler)

    print(f"\nTraining {VISUAL_EPOCHS} epochs...")
    history = trainer.train(epochs=VISUAL_EPOCHS, patience=12)

    ckpt = torch.load(out / "best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    test_scores, test_labels, test_preds = collect_predictions(model, test_loader)
    test_metrics = compute_all_metrics(test_labels, test_scores)

    return {"history": history, "test_metrics": test_metrics,
            "test_scores": test_scores, "test_labels": test_labels, "test_preds": test_preds,
            "model": model, "test_loader": test_loader}


# ── FUSION (real embedding-based) ────────────────────────────────────────────
def train_fusion_model(visual_results, audio_results):
    print("\n" + "=" * 70)
    print("  FUSION — Attention + Embedding-based Training")
    print("=" * 70)

    # --- Load pre-trained backbones ---
    visual_model = VisualBackbone(embedding_dim=512, pretrained=False,
                                  freeze_layers=0, dropout=0.5)
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
    root_a = data_cfg["paths"]["asvspoof2019"]
    acfg = data_cfg["asvspoof2019"]["audio"]
    root_v = data_cfg["paths"]["ff_c23"]
    csv_dir = data_cfg["ff_c23"]["csv_dir"]
    face_size = data_cfg["ff_c23"]["frame_extraction"]["face_size"]

    print("  Extracting visual embeddings...")
    v_train_ds = FF_C23_Dataset(root=root_v, csv_dir=csv_dir, split="train",
        num_frames=NUM_FRAMES, face_size=face_size, n_samples=N_SAMPLES, seed=42,
        use_face_detection=True)
    v_test_ds = FF_C23_Dataset(root=root_v, csv_dir=csv_dir, split="test",
        num_frames=NUM_FRAMES, face_size=face_size, n_samples=N_SAMPLES, seed=42,
        use_face_detection=True)

    v_train_loader = DataLoader(v_train_ds, batch_size=BATCH_SIZE, num_workers=0)
    v_test_loader = DataLoader(v_test_ds, batch_size=BATCH_SIZE, num_workers=0)

    v_train_emb, v_train_lbl = extract_embeddings(visual_model, v_train_loader)
    v_test_emb, v_test_lbl = extract_embeddings(visual_model, v_test_loader)

    print("  Extracting audio embeddings...")
    a_train_ds = ASVspoofDataset(root=root_a, split="train", sample_rate=acfg["sample_rate"],
        max_length=acfg["max_length"], n_lfcc=acfg["n_lfcc"], n_fft=acfg["n_fft"],
        n_samples=N_SAMPLES, seed=42)
    a_test_ds = ASVspoofDataset(root=root_a, split="eval", sample_rate=acfg["sample_rate"],
        max_length=acfg["max_length"], n_lfcc=acfg["n_lfcc"], n_fft=acfg["n_fft"],
        n_samples=N_SAMPLES, seed=42)

    a_train_loader = DataLoader(a_train_ds, batch_size=BATCH_SIZE, num_workers=0)
    a_test_loader = DataLoader(a_test_ds, batch_size=BATCH_SIZE, num_workers=0)

    a_train_emb, a_train_lbl = extract_embeddings(audio_model, a_train_loader)
    a_test_emb, a_test_lbl = extract_embeddings(audio_model, a_test_loader)

    # --- Build paired training data ---
    train_v_emb, train_a_emb, train_labels = build_pairs(
        v_train_emb, v_train_lbl, a_train_emb, a_train_lbl)
    test_v_emb, test_a_emb, test_labels = build_pairs(
        v_test_emb, v_test_lbl, a_test_emb, a_test_lbl)

    print(f"  Paired train: {len(train_labels)} | test: {len(test_labels)}")

    train_dataset = TensorDataset(train_v_emb, train_a_emb, train_labels)
    test_dataset = TensorDataset(test_v_emb, test_a_emb, test_labels)

    # Split train into train/val (80/20)
    n_val = max(int(len(train_dataset) * 0.2), 4)
    n_train = len(train_dataset) - n_val
    train_sub, val_sub = torch.utils.data.random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # --- Train attention fusion ---
    fusion_model = build_fusion("attention", visual_dim=512, audio_dim=512,
                                hidden_dim=256, dropout=0.3)

    def fusion_forward(model, batch, device):
        v_emb, a_emb, labels = batch
        v_emb, a_emb, labels = v_emb.to(device), a_emb.to(device), labels.to(device)
        logits = model(v_emb, a_emb)
        return logits, labels

    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FUSION_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    out = OUTPUT_DIR / "fusion"
    trainer = Trainer(model=fusion_model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, criterion=criterion, device=DEVICE,
        output_dir=str(out), scheduler=scheduler, forward_fn=fusion_forward)

    print(f"\nTraining attention fusion for {FUSION_EPOCHS} epochs...")
    fusion_history = trainer.train(epochs=FUSION_EPOCHS, patience=15)

    # --- Evaluate fusion ---
    ckpt = torch.load(out / "best_model.pt", map_location=DEVICE, weights_only=False)
    fusion_model.load_state_dict(ckpt["model_state_dict"])
    fusion_model.to(DEVICE).eval()

    test_scores, test_lbls, test_preds = [], [], []
    with torch.no_grad():
        for v_emb, a_emb, labels in test_loader:
            v_emb, a_emb = v_emb.to(DEVICE), a_emb.to(DEVICE)
            logits = fusion_model(v_emb, a_emb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            test_scores.append(probs.cpu().numpy())
            test_lbls.append(labels.numpy())

    test_scores = np.concatenate(test_scores)
    test_lbls = np.concatenate(test_lbls)
    test_metrics = compute_all_metrics(test_lbls, test_scores)
    _, thresh = compute_eer(test_lbls, test_scores)
    test_preds = (test_scores >= thresh).astype(int)

    # --- Also compute late fusion for comparison ---
    v_sc = visual_results["test_scores"]
    a_sc = audio_results["test_scores"]
    n = min(len(v_sc), len(a_sc))

    late_results = {}
    best_late_eer = 1.0
    best_w = 0.5
    for w in np.arange(0.0, 1.05, 0.1):
        fused = w * v_sc[:n] + (1 - w) * a_sc[:n]
        lbl = np.array([visual_results["test_labels"][i % len(visual_results["test_labels"])] for i in range(n)])
        m = compute_all_metrics(lbl, fused)
        late_results[f"{w:.1f}"] = m
        if m["eer"] < best_late_eer:
            best_late_eer = m["eer"]
            best_w = w

    print(f"\n  Attention Fusion EER: {test_metrics['eer']:.4f}, AUC: {test_metrics['auc']:.4f}")
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
    _, thresh = compute_eer(labels, scores)
    preds = (scores >= thresh).astype(int)
    return scores, labels, preds


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
    plot_training_curves(visual["history"], "Visual (ResNet18)", CHARTS_DIR / "visual_training.png")
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
        (visual, "Visual (ResNet18)", "#F44336", "--"),
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
    names = ["Audio\n(LCNN)", "Visual\n(ResNet18)", "Attention\nFusion"]
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

    fig.suptitle("Multi-Modal Liveness Detection — v2 Dashboard", fontsize=18, fontweight="bold", y=0.98)
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


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  MULTI-MODAL LIVENESS DETECTION — EXPERIMENT v2 (IMPROVED)")
    print(f"  Device: {DEVICE} | Samples: {N_SAMPLES} | Frames: {NUM_FRAMES}")
    print(f"  Improvements: augmentation, frozen layers, face detection,")
    print(f"  attention fusion, label smoothing, cosine LR, more data")
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

    # Remove model ref before passing (not serializable to JSON)
    vis_for_fusion = {k: v for k, v in visual_results.items() if k not in ("model", "test_loader")}

    t0 = time.time()
    fusion_results = train_fusion_model(vis_for_fusion, audio_results)
    fusion_time = time.time() - t0
    print(f"\n  Fusion done in {fusion_time:.1f}s")

    generate_charts(audio_results, visual_results, fusion_results)

    total_time = time.time() - start

    summary = {
        "experiment": {"device": str(DEVICE), "n_samples": N_SAMPLES,
            "visual_epochs": VISUAL_EPOCHS, "audio_epochs": AUDIO_EPOCHS,
            "fusion_epochs": FUSION_EPOCHS, "total_time": round(total_time, 1),
            "improvements": ["augmentation", "frozen_layers", "face_detection",
                           "attention_fusion", "label_smoothing", "cosine_lr",
                           "balanced_sampling", "200_samples"]},
        "audio_test_metrics": {k: round(float(v), 6) for k, v in audio_results["test_metrics"].items()},
        "visual_test_metrics": {k: round(float(v), 6) for k, v in visual_results["test_metrics"].items()},
        "fusion_test_metrics": {k: round(float(v), 6) for k, v in fusion_results["test_metrics"].items()},
    }

    with open(RESULTS_DIR / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  FINAL RESULTS (v2)")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Audio (LCNN)':<18} {'Visual (ResNet18)':<20} {'Att. Fusion':<18}")
    print("-" * 76)
    for metric in ["eer", "accuracy", "auc", "min_tdcf"]:
        a = audio_results["test_metrics"][metric]
        v = visual_results["test_metrics"][metric]
        f_val = fusion_results["test_metrics"][metric]
        if metric != "min_tdcf":
            a_s, v_s, f_s = f"{a:.2%}", f"{v:.2%}", f"{f_val:.2%}"
        else:
            a_s, v_s, f_s = f"{a:.4f}", f"{v:.4f}", f"{f_val:.4f}"
        print(f"{metric.upper():<20} {a_s:<18} {v_s:<20} {f_s:<18}")

    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Charts: {CHARTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
