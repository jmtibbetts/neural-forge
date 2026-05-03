"""
NeuralForge - BrainProbe
Captures live internals from CorticalBrain for real-time visualization.
Works with both CorticalBrain (new) and FinancialBrain (legacy).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional


def _to_list(t, max_units=64):
    """Safely convert a tensor to a JSON-serializable list."""
    if t is None:
        return []
    arr = t.detach().cpu().float().numpy()
    # squeeze batch dim if present
    if arr.ndim == 3:
        arr = arr[0]          # (T, H) or (H, T)
    elif arr.ndim == 2 and arr.shape[0] > 1:
        arr = arr[0]          # take first sample
    elif arr.ndim == 2:
        arr = arr[0]
    return arr[:max_units].tolist() if arr.ndim == 1 else arr[:max_units].tolist()


class BrainProbe:
    """
    Attach to a model and capture activations for SocketIO streaming.
    Supports CorticalBrain internals dict (preferred) and FinancialBrain hooks (legacy).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: List = []
        self._activations: Dict[str, np.ndarray] = {}
        self._gradients:   Dict[str, np.ndarray] = {}
        self._last_internals: Optional[dict] = None

        self._is_cortical = hasattr(model, 'thalamus')
        if not self._is_cortical:
            self._attach_legacy()

    # ─── Legacy hooks (FinancialBrain) ──────────────────────────────────────

    def _attach_legacy(self):
        self._hooks.append(
            self.model.lstm.register_forward_hook(self._make_hook("lstm_out", take_idx=0))
        )
        self._hooks.append(
            self.model.attention.register_forward_hook(self._make_hook("attention_raw"))
        )
        for i, layer in enumerate(self.model.classifier):
            if isinstance(layer, (nn.Linear, nn.GELU, nn.LayerNorm)):
                name = f"classifier_{i}_{layer.__class__.__name__}"
                self._hooks.append(layer.register_forward_hook(self._make_hook(name)))
        for i, layer in enumerate(self.model.classifier):
            if isinstance(layer, nn.Linear):
                name = f"grad_classifier_{i}"
                self._hooks.append(layer.weight.register_hook(self._make_grad_hook(name)))

    def _make_hook(self, name, take_idx=None):
        def hook(module, input, output):
            data = output[take_idx] if take_idx is not None else output
            if isinstance(data, torch.Tensor):
                self._activations[name] = data.detach().cpu().float().numpy()
        return hook

    def _make_grad_hook(self, name):
        def hook(grad):
            self._gradients[name] = grad.detach().cpu().float().abs().numpy()
        return hook

    # ─── Receive CorticalBrain internals ────────────────────────────────────

    def record_internals(self, internals: dict):
        """Called each forward pass with the internals dict from CorticalBrain."""
        self._last_internals = internals

    # ─── Snapshot builders ───────────────────────────────────────────────────

    def snapshot(self, attn_weights=None) -> dict:
        if self._is_cortical:
            return self._snapshot_cortical()
        else:
            return self._snapshot_legacy(attn_weights)

    def _snapshot_cortical(self) -> dict:
        i = self._last_internals
        if i is None:
            return {}

        snap = {"type": "cortical"}

        # ── Thalamic gate: mean activation per timestep → attention-like strip
        if 'thalamus' in i:
            t = i['thalamus']           # (B, T, col_dim)
            snap['thalamus_activation'] = t[0].abs().mean(dim=-1).tolist()  # (T,)

        # ── Hippocampal temporal attention — which candles the brain consolidated
        if 'temporal_attn' in i:
            snap['attention'] = i['temporal_attn'][0].tolist()  # (T,)

        # ── Cortical columns — L2/3 activations (sparse representation)
        columns_viz = []
        for idx, col_l23 in enumerate(i.get('columns_L23', [])):
            # col_l23: (B, T, col_dim) — mean over time, first sample
            acts = col_l23[0].mean(dim=0)[:32].tolist()  # (32,)
            columns_viz.append({
                'name': f'Column {idx+1}',
                'activations': acts,
                'mean_fire_rate': float(col_l23[0].abs().mean()),
            })
        snap['cortical_columns'] = columns_viz

        # ── CA3 hippocampal heatmap (memory matrix)
        if 'hippocampus_ca3' in i:
            ca3 = i['hippocampus_ca3'][0]  # (T, memory_dim)
            snap['hippocampus_heatmap'] = ca3.T[:24].tolist()  # (24, T)

        # ── Prefrontal neuron activations
        if 'prefrontal' in i:
            pfc = i['prefrontal']          # (B, 64)
            snap['prefrontal'] = pfc[0][:64].tolist()

        # ── Dopamine level (neuromodulation)
        if 'dopamine' in i:
            d = i['dopamine']              # (B,)
            snap['dopamine'] = float(d[0])

        # ── Integrated cortical state (mini heatmap, T x memory_dim[:16])
        if 'integrated' in i:
            ig = i['integrated'][0]        # (T, memory_dim)
            snap['cortical_integration'] = ig.T[:16].tolist()  # (16, T)

        return snap

    def _snapshot_legacy(self, attn_weights=None) -> dict:
        snap = {"type": "legacy"}
        if "lstm_out" in self._activations:
            lstm = self._activations["lstm_out"]
            snap["lstm_heatmap"] = lstm[0].T[:32].tolist()
        if attn_weights is not None:
            w = attn_weights.detach().cpu().float().numpy()
            snap["attention"] = w[0].tolist()
        classifier_acts = []
        for key in sorted(self._activations.keys()):
            if key.startswith("classifier_"):
                act = self._activations[key]
                vals = act[0, :64].tolist() if act.ndim == 2 else act[:64].tolist()
                classifier_acts.append({"name": key.split("_", 2)[-1], "activations": vals})
        snap["classifier"] = classifier_acts
        grad_mags = {k: float(v.mean()) for k, v in self._gradients.items()}
        snap["grad_magnitudes"] = grad_mags
        return snap

    def clear(self):
        self._activations.clear()
        self._gradients.clear()
        self._last_internals = None

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
