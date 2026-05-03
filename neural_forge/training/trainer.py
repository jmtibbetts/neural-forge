"""
NeuralForge Trainer
High-performance training loop with real-time metrics, callbacks, and mixed precision.
Optimized for RTX 5090 with bf16 + torch.compile.
"""

import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("neural_forge.trainer")


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    precision: str = "bf16"           # "fp32" | "fp16" | "bf16"
    compile_model: bool = True
    log_every: int = 10               # log every N steps
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5               # save checkpoint every N epochs
    early_stopping_patience: int = 0  # 0 = disabled
    scheduler: str = "cosine"         # "cosine" | "step" | "plateau" | "none"
    warmup_epochs: int = 1


@dataclass 
class TrainMetrics:
    epoch: int = 0
    step: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_acc: float = 0.0
    val_acc: float = 0.0
    lr: float = 0.0
    gpu_mem_gb: float = 0.0
    elapsed_sec: float = 0.0
    history: List[Dict] = field(default_factory=list)

    def snapshot(self) -> Dict:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6),
            "train_acc": round(self.train_acc, 4),
            "val_acc": round(self.val_acc, 4),
            "lr": self.lr,
            "gpu_mem_gb": round(self.gpu_mem_gb, 2),
            "elapsed_sec": round(self.elapsed_sec, 2),
        }


class Trainer:
    """
    Universal NeuralForge trainer.
    Works with any nn.Module + DataLoader pair.
    Emits real-time metrics via callback for live UI updates.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        device: torch.device,
        on_step: Optional[Callable[[TrainMetrics], None]] = None,
        on_epoch: Optional[Callable[[TrainMetrics], None]] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.on_step = on_step
        self.on_epoch = on_epoch
        self.metrics = TrainMetrics()

        # Mixed precision
        self._use_amp = config.precision in ("fp16", "bf16") and device.type == "cuda"
        self._amp_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
        self._scaler = GradScaler(enabled=(config.precision == "fp16"))

        # Compile (torch.compile for Inductor)
        if config.compile_model and torch.cuda.is_available():
            logger.info("Compiling model with torch.compile (Inductor)...")
            self.model = torch.compile(self.model, backend="inductor", mode="max-autotune")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
    ):
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        scheduler = self._build_scheduler(optimizer, len(train_loader))
        best_val_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            self.metrics.epoch = epoch

            # --- Training ---
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion, scheduler, epoch, start_time
            )
            self.metrics.train_loss = train_loss
            self.metrics.train_acc = train_acc

            # --- Validation ---
            if val_loader:
                val_loss, val_acc = self._val_epoch(val_loader, criterion)
                self.metrics.val_loss = val_loss
                self.metrics.val_acc = val_acc

            self.metrics.elapsed_sec = time.time() - start_time
            snap = self.metrics.snapshot()
            self.metrics.history.append(snap)

            if self.on_epoch:
                self.on_epoch(self.metrics)

            logger.info(
                f"Epoch {epoch}/{self.config.epochs} — "
                f"loss: {train_loss:.4f} | val_loss: {val_loss if val_loader else 'N/A':.4f}"
                if val_loader else
                f"Epoch {epoch}/{self.config.epochs} — loss: {train_loss:.4f}"
            )

            # Checkpointing
            if epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch, optimizer)

            # Early stopping
            if val_loader and self.config.early_stopping_patience > 0:
                if self.metrics.val_loss < best_val_loss:
                    best_val_loss = self.metrics.val_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, optimizer, name="best")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

        return self.metrics

    def _train_epoch(self, loader, optimizer, criterion, scheduler, epoch, start_time):
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for step, batch in enumerate(loader):
            self.metrics.step += 1

            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
            else:
                # PyG Data object
                batch = batch.to(self.device)
                x, y = batch, batch.y

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                if hasattr(x, 'x'):  # PyG batch
                    out = self.model(x)
                else:
                    out = self.model(x)
                loss = criterion(out, y)

            self._scaler.scale(loss).backward()

            if self.config.grad_clip > 0:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self._scaler.step(optimizer)
            self._scaler.update()

            if scheduler and hasattr(scheduler, 'step') and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            total_loss += loss.item() * (y.size(0) if hasattr(y, 'size') else 1)
            if out.shape[-1] > 1:
                preds = out.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
            total_samples += y.size(0) if hasattr(y, 'size') else 1

            # GPU memory
            if torch.cuda.is_available():
                self.metrics.gpu_mem_gb = torch.cuda.memory_allocated() / 1e9

            self.metrics.lr = optimizer.param_groups[0]["lr"]
            self.metrics.elapsed_sec = time.time() - start_time

            if step % self.config.log_every == 0 and self.on_step:
                self.on_step(self.metrics)

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc

    @torch.no_grad()
    def _val_epoch(self, loader, criterion):
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
            else:
                batch = batch.to(self.device)
                x, y = batch, batch.y

            with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                out = self.model(x) if not hasattr(x, 'x') else self.model(x)
                loss = criterion(out, y)

            total_loss += loss.item() * (y.size(0) if hasattr(y, 'size') else 1)
            if out.shape[-1] > 1:
                preds = out.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
            total_samples += y.size(0) if hasattr(y, 'size') else 1

        return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)

    def _build_scheduler(self, optimizer, steps_per_epoch):
        cfg = self.config
        total_steps = cfg.epochs * steps_per_epoch
        warmup_steps = cfg.warmup_epochs * steps_per_epoch

        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg.lr, total_steps=total_steps, pct_start=warmup_steps/total_steps
            )
        elif cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return None

    def _save_checkpoint(self, epoch, optimizer, name=None):
        import os
        fname = f"checkpoint_epoch{epoch}.pt" if name is None else f"checkpoint_{name}.pt"
        path = os.path.join(self.config.checkpoint_dir, fname)
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": self.metrics.snapshot(),
        }, path)
        logger.info(f"Saved checkpoint: {path}")
