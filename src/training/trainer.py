"""
Generic training loop for visual, audio, and fusion models.
Handles train/validation, checkpointing, and metric logging.
"""

import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics


class Trainer:
    """Unified trainer for single-modal or multi-modal models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        output_dir: str = "outputs",
        scheduler: Optional[object] = None,
        forward_fn: Optional[Callable] = None,
    ):
        """
        Args:
            forward_fn: custom forward function for multi-modal models.
                        Signature: forward_fn(model, batch) -> (logits, labels)
                        If None, assumes batch = (inputs, labels) and logits = model(inputs).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = scheduler
        self.forward_fn = forward_fn
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_eer": [],
            "val_accuracy": [],
            "val_auc": [],
        }

    def _default_forward(self, batch):
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(inputs)
        return logits, labels

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            if self.forward_fn:
                logits, labels = self.forward_fn(self.model, batch, self.device)
            else:
                logits, labels = self._default_forward(batch)

            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        all_scores = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            if self.forward_fn:
                logits, labels = self.forward_fn(self.model, batch, self.device)
            else:
                logits, labels = self._default_forward(batch)

            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_scores.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        metrics = compute_all_metrics(all_labels, all_scores)
        metrics["loss"] = total_loss / max(n_batches, 1)
        return metrics

    def train(self, epochs: int, patience: int = 10) -> Dict[str, list]:
        """Full training loop with early stopping.

        Args:
            epochs: maximum number of epochs
            patience: early stopping patience on validation EER
        Returns:
            training history dict
        """
        best_eer = 1.0
        no_improve = 0

        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler:
                try:
                    self.scheduler.step(val_metrics.get("eer", 1.0))
                except TypeError:
                    self.scheduler.step()

            elapsed = time.time() - start
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_eer"].append(val_metrics["eer"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_auc"].append(val_metrics["auc"])

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"EER: {val_metrics['eer']:.4f} | "
                f"AUC: {val_metrics['auc']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_metrics["eer"] < best_eer:
                best_eer = val_metrics["eer"]
                no_improve = 0
                self._save_checkpoint(epoch, val_metrics, "best_model.pt")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

        self._save_checkpoint(epoch, val_metrics, "last_model.pt")
        return self.history

    def _save_checkpoint(self, epoch: int, metrics: dict, filename: str):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, self.output_dir / filename)
