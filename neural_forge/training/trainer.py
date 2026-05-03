"""
NeuralForge - Financial Brain Trainer
Trains the LSTM+MLP model, emits live metrics + brain activations via callbacks.
"""

import os
import time
import json
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from neural_forge.models.financial_brain import FinancialBrain
from neural_forge.visualization.brain_probe import BrainProbe


CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["BUY", "HOLD", "SELL"]


class Trainer:
    def __init__(
        self,
        model:          FinancialBrain,
        train_loader,
        val_loader,
        class_weights:  Optional[torch.Tensor] = None,
        lr:             float = 3e-4,
        epochs:         int   = 30,
        device:         str   = "auto",
        probe_every:    int   = 10,
        on_epoch_end:   Optional[Callable] = None,
        on_batch_end:   Optional[Callable] = None,
        on_brain_state: Optional[Callable] = None,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model          = model.to(self.device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.epochs         = epochs
        self.probe_every    = probe_every
        self.on_epoch_end   = on_epoch_end
        self.on_batch_end   = on_batch_end
        self.on_brain_state = on_brain_state

        self.probe = BrainProbe(self.model)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        total_steps = epochs * len(train_loader)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr * 10,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        self.history      = []
        self.best_val_acc = 0.0
        self.best_epoch   = 0

        print(f"  Device : {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params : {n_params:,}")

    # ─── Training loop ──────────────────────────────────────

    def train(self):
        print(f"\n  Starting training for {self.epochs} epochs...\n")
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(epoch)
            val_loss,   val_acc   = self._eval_epoch(self.val_loader)

            elapsed = time.time() - t0
            lr_now  = self.scheduler.get_last_lr()[0]

            metrics = {
                "epoch":      epoch,
                "train_loss": round(train_loss, 4),
                "train_acc":  round(train_acc,  4),
                "val_loss":   round(val_loss,   4),
                "val_acc":    round(val_acc,    4),
                "lr":         round(lr_now,     8),
                "elapsed_s":  round(elapsed,    2),
            }

            self.history.append(metrics)
            self._save_history()

            print(
                f"  Epoch {epoch:03d}/{self.epochs} | "
                f"loss {train_loss:.4f} -> val {val_loss:.4f} | "
                f"acc {train_acc:.3f} -> val {val_acc:.3f} | "
                f"lr {lr_now:.2e} | {elapsed:.1f}s"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch   = epoch
                self._save_checkpoint("best_model.pt", metrics)
                print(f"    New best val_acc: {val_acc:.4f}")

            if self.on_epoch_end:
                self.on_epoch_end(metrics)

        print(f"\n  Training complete. Best val_acc: {self.best_val_acc:.4f} @ epoch {self.best_epoch}")
        self.probe.detach()
        return self.history

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits, attn = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            preds       = logits.argmax(dim=-1)
            correct    += (preds == y).sum().item()
            total      += y.size(0)
            total_loss += loss.item() * y.size(0)

            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1).mean(dim=0).cpu().tolist()
                # Pad to 3 if model only sees 2 classes in this batch
                while len(probs) < 3:
                    probs.append(0.0)

            batch_metrics = {
                "epoch":      epoch,
                "batch":      batch_idx,
                "batch_loss": round(loss.item(), 4),
                "batch_acc":  round((preds == y).float().mean().item(), 4),
                "signal_probs": {
                    "BUY":  round(probs[0], 4),
                    "HOLD": round(probs[1], 4),
                    "SELL": round(probs[2], 4),
                },
            }

            if self.on_batch_end:
                self.on_batch_end(batch_metrics)

            if self.on_brain_state and batch_idx % self.probe_every == 0:
                snap = self.probe.snapshot(attn_weights=attn)
                snap["epoch"] = epoch
                snap["batch"] = batch_idx
                self.on_brain_state(snap)

            self.probe.clear()

        return total_loss / total, correct / total

    @torch.no_grad()
    def _eval_epoch(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in loader:
            X, y      = X.to(self.device), y.to(self.device)
            logits, _ = self.model(X)
            loss      = self.criterion(logits, y)

            preds       = logits.argmax(dim=-1)
            correct    += (preds == y).sum().item()
            total      += y.size(0)
            total_loss += loss.item() * y.size(0)

        return total_loss / total, correct / total

    # ─── Checkpointing ──────────────────────────────────────

    def _save_checkpoint(self, filename: str, metrics: dict):
        path = CHECKPOINT_DIR / filename
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics":         metrics,
            "history":         self.history,
        }, path)

    def _save_history(self):
        with open(CHECKPOINT_DIR / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def load_best(self):
        path = CHECKPOINT_DIR / "best_model.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded best model from {path}")
        else:
            print("  No checkpoint found.")

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []

        for X, y in loader:
            X          = X.to(self.device)
            logits, _  = self.model(X)
            preds      = logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(y)

        preds  = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()

        from sklearn.metrics import classification_report, confusion_matrix

        # Determine which classes actually appear (guards against small datasets)
        present_labels = sorted(set(labels.tolist()) | set(preds.tolist()))
        present_names  = [CLASS_NAMES[i] for i in present_labels if i < len(CLASS_NAMES)]

        report = classification_report(
            labels, preds,
            labels=present_labels,
            target_names=present_names,
            output_dict=True,
            zero_division=0,
        )

        # Always return a full 3x3 confusion matrix (missing classes = 0 rows/cols)
        cm_raw = confusion_matrix(labels, preds, labels=present_labels)
        cm_full = np.zeros((3, 3), dtype=int)
        for i, li in enumerate(present_labels):
            for j, lj in enumerate(present_labels):
                cm_full[li][lj] = cm_raw[i][j]

        # Ensure all 3 class keys exist in report for the UI
        for i, name in enumerate(CLASS_NAMES):
            if name not in report:
                report[name] = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}

        return report, cm_full
