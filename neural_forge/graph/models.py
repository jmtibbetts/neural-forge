"""
NeuralForge Graph Neural Network Models
Built on PyTorch Geometric — GCN, GAT, GraphSAGE, GIN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import (
        GCNConv, GATv2Conv, SAGEConv, GINConv,
        global_mean_pool, global_max_pool, global_add_pool,
        BatchNorm
    )
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[NeuralForge] PyTorch Geometric not installed — graph models unavailable.")


def _check_pyg():
    if not HAS_PYG:
        raise ImportError(
            "PyTorch Geometric is required for graph models.\n"
            "Install: pip install torch-geometric"
        )


class GCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, 2017)
    Supports node classification and graph classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        task: str = "node",        # "node" | "graph"
        pooling: str = "mean",     # "mean" | "max" | "add"
    ):
        super().__init__()
        _check_pyg()
        self.task = task
        self.dropout = dropout
        self.pooling = pooling

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GCNConv(in_ch, out_ch))
            if i < num_layers - 1:
                self.bns.append(BatchNorm(out_ch))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.bns):
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph" and batch is not None:
            x = self._pool(x, batch)

        return x

    def _pool(self, x, batch):
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        return global_add_pool(x, batch)


class GAT(nn.Module):
    """
    Graph Attention Network v2 (Brody et al., 2022)
    Multi-head attention for expressive node/graph representations.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.6,
        task: str = "node",
        pooling: str = "mean",
    ):
        super().__init__()
        _check_pyg()
        self.task = task
        self.dropout = dropout
        self.pooling = pooling

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels * heads
            is_last = i == num_layers - 1
            out_ch = out_channels if is_last else hidden_channels
            h = 1 if is_last else heads
            self.convs.append(GATv2Conv(in_ch, out_ch, heads=h, dropout=dropout, concat=not is_last))
            if not is_last:
                self.bns.append(BatchNorm(out_ch * h))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.bns):
                x = self.bns[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph" and batch is not None:
            x = self._pool(x, batch)

        return x

    def _pool(self, x, batch):
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        return global_add_pool(x, batch)


class GraphSAGE(nn.Module):
    """
    GraphSAGE — Inductive Representation Learning (Hamilton et al., 2017)
    Great for large-scale graphs and inductive learning.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        aggr: str = "mean",        # "mean" | "max" | "lstm"
        task: str = "node",
        pooling: str = "mean",
    ):
        super().__init__()
        _check_pyg()
        self.task = task
        self.dropout = dropout
        self.pooling = pooling

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(in_ch, out_ch, aggr=aggr))
            if i < num_layers - 1:
                self.bns.append(BatchNorm(out_ch))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.bns):
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph" and batch is not None:
            x = self._pool(x, batch)

        return x

    def _pool(self, x, batch):
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        return global_add_pool(x, batch)


class GIN(nn.Module):
    """
    Graph Isomorphism Network (Xu et al., 2019)
    Maximally powerful GNN for graph classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 5,
        dropout: float = 0.5,
        task: str = "graph",
        pooling: str = "add",
    ):
        super().__init__()
        _check_pyg()
        self.task = task
        self.dropout = dropout
        self.pooling = pooling

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            mlp = nn.Sequential(
                nn.Linear(in_ch, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm(hidden_channels))

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        if self.task == "graph" and batch is not None:
            if self.pooling == "add":
                x = global_add_pool(x, batch)
            elif self.pooling == "mean":
                x = global_mean_pool(x, batch)
            else:
                x = global_max_pool(x, batch)

        return self.head(x)
