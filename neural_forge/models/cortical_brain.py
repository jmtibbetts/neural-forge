"""
NeuralForge - CorticalBrain
A biologically-inspired neural architecture modeled on modern neuroscience:

  1. Thalamic Gate       — sensory relay, filters & routes incoming signals
  2. Cortical Columns    — 6 mini-columns (like L2/L3/L5 cortex), each specializing
                           in a different aspect of the input sequence
  3. Hippocampal Module  — sequence memory with recurrent dynamics (like CA1/CA3)
  4. Dopaminergic Head   — reward-modulated gating (neuromodulation)
  5. Prefrontal Decoder  — executive decision layer → BUY/HOLD/SELL

Key neuroscience concepts implemented:
  - Lateral inhibition within columns (winner-take-more via softmax competition)
  - Hebbian-like weight normalization (neurons that fire together stay together)
  - Dendritic nonlinearity (two-stage integration per neuron)
  - Neuromodulatory gain control (dopamine signal scales attention)
  - Sparse coding (top-k activation per column)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── Dendritic Neuron ─────────────────────────────────────────────────────────
class DendriticNeuron(nn.Module):
    """
    Two-compartment neuron: proximal dendrite + distal dendrite → soma.
    Proximal: direct feedforward input (like basal dendrites).
    Distal:   context/modulatory input (like apical dendrites in L5).
    Both are summed with a learnable gate before the somatic nonlinearity.
    """
    def __init__(self, in_dim: int, ctx_dim: int, out_dim: int):
        super().__init__()
        self.proximal = nn.Linear(in_dim,  out_dim)
        self.distal   = nn.Linear(ctx_dim, out_dim)
        self.gate     = nn.Parameter(torch.ones(out_dim) * 0.5)
        self.norm     = nn.LayerNorm(out_dim)

    def forward(self, x, ctx):
        prox   = self.proximal(x)
        dist   = self.distal(ctx)
        gate   = torch.sigmoid(self.gate)
        soma   = gate * prox + (1 - gate) * dist
        return F.gelu(self.norm(soma))


# ─── Cortical Column ──────────────────────────────────────────────────────────
class CorticalColumn(nn.Module):
    """
    A cortical mini-column processes a local receptive field of the input.
    Layers: L4 (input) → L2/3 (association) → L5 (output)
    Lateral inhibition via softmax competition between neurons in the same layer.
    Sparse coding: only top-k neurons per column stay active.
    """
    def __init__(self, in_dim: int, col_dim: int, sparsity_k: int = 8):
        super().__init__()
        self.sparsity_k = sparsity_k

        # L4: thalamic input → column
        self.L4 = nn.Sequential(nn.Linear(in_dim, col_dim), nn.LayerNorm(col_dim))

        # L2/3: associative layer with lateral inhibition
        self.L23 = nn.Linear(col_dim, col_dim)
        self.L23_norm = nn.LayerNorm(col_dim)

        # L5: output projection
        self.L5 = nn.Linear(col_dim, col_dim)
        self.L5_norm = nn.LayerNorm(col_dim)

        self.drop = nn.Dropout(0.1)

    def _lateral_inhibition(self, x):
        """Softmax competition: bright neurons suppress dim ones."""
        scale = F.softmax(x, dim=-1) * x.shape[-1]  # normalize but preserve magnitude
        return x * scale

    def _sparse_code(self, x):
        """Keep only top-k activations per sample (sparse representation)."""
        k = min(self.sparsity_k, x.shape[-1])
        topk_vals, _ = x.topk(k, dim=-1)
        threshold = topk_vals[..., -1:].detach()
        mask = (x >= threshold).float()
        return x * mask

    def forward(self, x):
        # L4: receive input
        h = F.gelu(self.L4(x))

        # L2/3: lateral inhibition + sparse coding
        h = self.L23_norm(self.L23(h))
        h = self._lateral_inhibition(h)
        h = self._sparse_code(h)
        h = self.drop(h)

        # L5: output with residual
        out = F.gelu(self.L5_norm(self.L5(h)))
        return out, h  # (output for next stage, L2/3 activations for visualization)


# ─── Thalamic Gate ────────────────────────────────────────────────────────────
class ThalamicGate(nn.Module):
    """
    Routes and filters sensory input before cortex.
    Implements predictive gating: learns which features to amplify or suppress.
    Analogous to thalamo-cortical relay nuclei (LGN, VPM, etc.)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.relay  = nn.Linear(in_dim, out_dim)
        self.gate   = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x):
        # (B, T, in_dim) -> (B, T, out_dim)
        relayed = F.gelu(self.relay(x))
        gate    = self.gate(x)
        return self.norm(relayed * gate)


# ─── Hippocampal Module ───────────────────────────────────────────────────────
class HippocampalModule(nn.Module):
    """
    Sequence memory — models CA3 (pattern completion) + CA1 (temporal coding).
    Uses a multi-head attention mechanism over the sequence (like place cells
    building a cognitive map of the time series).
    """
    def __init__(self, dim: int, num_heads: int = 4, seq_len: int = 30):
        super().__init__()
        self.dim = dim

        # CA3: self-attention for pattern completion (Hopfield-like)
        self.CA3 = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.CA3_norm = nn.LayerNorm(dim)

        # CA1: temporal attention — which moments to consolidate
        self.CA1_query  = nn.Linear(dim, dim)
        self.CA1_key    = nn.Linear(dim, dim)
        self.CA1_value  = nn.Linear(dim, dim)
        self.CA1_norm   = nn.LayerNorm(dim)

        # Entorhinal output projection
        self.entorhinal = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        """
        x: (B, T, dim)
        returns: context (B, dim), temporal_attn (B, T)
        """
        # CA3: pattern completion via self-attention
        ca3_out, _ = self.CA3(x, x, x)
        ca3_out = self.CA3_norm(x + ca3_out)

        # CA1: temporal consolidation — attend to the full sequence
        Q = self.CA1_query(ca3_out[:, -1:, :])   # use last timestep as query
        K = self.CA1_key(ca3_out)
        V = self.CA1_value(ca3_out)

        scale  = math.sqrt(self.dim)
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale   # (B, 1, T)
        temporal_attn = F.softmax(scores, dim=-1)            # (B, 1, T)
        context = torch.bmm(temporal_attn, V).squeeze(1)    # (B, dim)

        # Entorhinal cortex: compress to episodic summary
        memory = self.entorhinal(context)

        return memory, temporal_attn.squeeze(1), ca3_out  # ca3_out for viz


# ─── Dopaminergic Neuromodulator ──────────────────────────────────────────────
class DopaminergicModule(nn.Module):
    """
    Models dopaminergic neuromodulation:
    - Computes a 'reward prediction' signal from the current state
    - This signal scales attention (like VTA/SNc dopamine gating prefrontal cortex)
    - High dopamine = confident, exploratory state
    - Low dopamine = cautious, conservative
    """
    def __init__(self, dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.gain_modulator = nn.Linear(1, dim)

    def forward(self, x):
        dopamine = self.predictor(x)              # (B, 1) — scalar neuromod signal
        gain     = torch.sigmoid(self.gain_modulator(dopamine))  # (B, dim)
        return x * gain, dopamine.squeeze(-1)     # modulated features, dopamine level


# ─── CorticalBrain ────────────────────────────────────────────────────────────
class CorticalBrain(nn.Module):
    """
    Full biologically-inspired brain for financial signal processing.

    Processing pipeline:
      Input candles → Thalamic Gate → 6 Cortical Columns (parallel)
                    → Column integration → Hippocampal Module (memory)
                    → Dopaminergic Modulation → Prefrontal Decoder → BUY/HOLD/SELL
    """

    NUM_COLUMNS = 6

    def __init__(
        self,
        input_size:  int = 12,
        col_dim:     int = 64,
        memory_dim:  int = 128,
        num_classes: int = 3,
        seq_len:     int = 30,
        sparsity_k:  int = 8,
    ):
        super().__init__()
        self.input_size = input_size
        self.col_dim    = col_dim
        self.memory_dim = memory_dim
        self.seq_len    = seq_len

        # ── Thalamic relay ────────────────────────────────
        self.thalamus = ThalamicGate(input_size, col_dim)

        # ── Cortical columns (each specializes differently) ─
        self.columns = nn.ModuleList([
            CorticalColumn(col_dim, col_dim, sparsity_k=sparsity_k)
            for _ in range(self.NUM_COLUMNS)
        ])

        # ── Column integration (cortico-cortical connections) ─
        integrated_dim = col_dim * self.NUM_COLUMNS
        self.cortical_integrator = nn.Sequential(
            nn.Linear(integrated_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
        )

        # ── Hippocampal memory module ─────────────────────
        self.hippocampus = HippocampalModule(memory_dim, num_heads=4, seq_len=seq_len)

        # ── Dendritic prefrontal neurons ──────────────────
        # Combines hippocampal memory + cortical context
        self.prefrontal = DendriticNeuron(memory_dim, memory_dim, 64)

        # ── Dopaminergic neuromodulator ───────────────────
        self.dopamine = DopaminergicModule(64)

        # ── Decision layer (motor output) ─────────────────
        self.decision = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, T, input_size)
        Returns:
            logits:        (B, num_classes)
            temporal_attn: (B, T)           — hippocampal attention for viz
            internals:     dict             — all internal states for visualization
        """
        B, T, _ = x.shape
        internals = {}

        # ── 1. Thalamic relay ─────────────────────────────
        thal_out = self.thalamus(x)                    # (B, T, col_dim)
        internals['thalamus'] = thal_out.detach().cpu().float()

        # ── 2. Cortical columns (parallel processing) ─────
        col_outputs = []
        col_l23s    = []
        for col in self.columns:
            out, l23 = col(thal_out)                   # both (B, T, col_dim)
            col_outputs.append(out)
            col_l23s.append(l23)
        internals['columns_L23'] = [l.detach().cpu().float() for l in col_l23s]
        internals['columns_L5']  = [o.detach().cpu().float() for o in col_outputs]

        # ── 3. Cortical integration ───────────────────────
        stacked   = torch.cat(col_outputs, dim=-1)    # (B, T, col_dim*N_cols)
        integrated = self.cortical_integrator(stacked) # (B, T, memory_dim)
        internals['integrated'] = integrated.detach().cpu().float()

        # ── 4. Hippocampal memory ─────────────────────────
        memory, temporal_attn, ca3_out = self.hippocampus(integrated)
        internals['hippocampus_ca3']   = ca3_out.detach().cpu().float()
        internals['temporal_attn']     = temporal_attn.detach().cpu().float()

        # ── 5. Prefrontal dendritic integration ───────────
        # Proximal: hippocampal memory, Distal: last cortical state
        cortical_ctx = integrated[:, -1, :]           # last timestep
        pfc_out = self.prefrontal(memory, cortical_ctx)
        internals['prefrontal'] = pfc_out.detach().cpu().float()

        # ── 6. Dopaminergic modulation ────────────────────
        modulated, dopamine_level = self.dopamine(pfc_out)
        internals['dopamine'] = dopamine_level.detach().cpu().float()

        # ── 7. Decision / motor output ────────────────────
        logits = self.decision(modulated)             # (B, num_classes)

        return logits, temporal_attn, internals

    @property
    def label_map(self):
        return {0: "BUY", 1: "HOLD", 2: "SELL"}
