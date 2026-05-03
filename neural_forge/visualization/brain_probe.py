"""
NeuralForge - Brain Probe
Hooks into the model during forward passes and captures:
  - LSTM hidden state activations (per layer, per timestep)
  - Attention weights (which candles the brain focuses on)
  - MLP layer activations (classifier head)
  - Gradient magnitudes (which neurons are learning)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional


class BrainProbe:
    """
    Attach to a FinancialBrain and intercept activations in real time.
    Usage:
        probe = BrainProbe(model)
        logits, attn = model(x)
        snapshot = probe.snapshot()   # dict ready for SocketIO emit
        probe.clear()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: List = []
        self._activations: Dict[str, np.ndarray] = {}
        self._gradients:   Dict[str, np.ndarray] = {}
        self._attach()

    # ─── Hook attachment ────────────────────────────────────

    def _attach(self):
        # LSTM output (all timesteps)
        self._hooks.append(
            self.model.lstm.register_forward_hook(self._make_hook("lstm_out", take_idx=0))
        )

        # Attention layer
        self._hooks.append(
            self.model.attention.register_forward_hook(self._make_hook("attention_raw"))
        )

        # Classifier layers
        for i, layer in enumerate(self.model.classifier):
            if isinstance(layer, (nn.Linear, nn.GELU, nn.LayerNorm)):
                name = f"classifier_{i}_{layer.__class__.__name__}"
                self._hooks.append(
                    layer.register_forward_hook(self._make_hook(name))
                )

        # Gradient hooks on classifier weights
        for i, layer in enumerate(self.model.classifier):
            if isinstance(layer, nn.Linear):
                name = f"grad_classifier_{i}"
                self._hooks.append(
                    layer.weight.register_hook(self._make_grad_hook(name))
                )

    def _make_hook(self, name: str, take_idx: Optional[int] = None):
        def hook(module, input, output):
            if take_idx is not None:
                data = output[take_idx]
            else:
                data = output
            if isinstance(data, torch.Tensor):
                self._activations[name] = data.detach().cpu().float().numpy()
        return hook

    def _make_grad_hook(self, name: str):
        def hook(grad):
            self._gradients[name] = grad.detach().cpu().float().abs().numpy()
        return hook

    # ─── Snapshot ───────────────────────────────────────────

    def snapshot(self, attn_weights: Optional[torch.Tensor] = None) -> dict:
        """
        Build a JSON-serialisable snapshot of the current brain state.
        attn_weights: (batch, seq_len) tensor from model.forward()
        """
        snap = {}

        # LSTM hidden activations — mean over batch, all timesteps, first 32 units
        if "lstm_out" in self._activations:
            lstm = self._activations["lstm_out"]  # (B, T, H)
            snap["lstm_heatmap"] = lstm[0].T[:32].tolist()  # (32, T) — first sample

        # Attention weights — which candles the brain focused on
        if attn_weights is not None:
            w = attn_weights.detach().cpu().float().numpy()
            snap["attention"] = w[0].tolist()  # (T,) — first sample

        # Classifier activations — each linear/activation layer
        classifier_acts = []
        for key in sorted(self._activations.keys()):
            if key.startswith("classifier_"):
                act = self._activations[key]
                if act.ndim == 2:
                    # (B, N) — take first sample, clip to 64 units
                    vals = act[0, :64].tolist()
                elif act.ndim == 1:
                    vals = act[:64].tolist()
                else:
                    continue
                classifier_acts.append({
                    "name": key.split("_", 2)[-1],
                    "activations": vals,
                })
        snap["classifier"] = classifier_acts

        # Gradient magnitudes — which neurons are learning hardest
        grad_mags = {}
        for key, grad in self._gradients.items():
            grad_mags[key] = float(grad.mean())
        snap["grad_magnitudes"] = grad_mags

        return snap

    def clear(self):
        self._activations.clear()
        self._gradients.clear()

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
