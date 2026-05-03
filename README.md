# ⚡ NeuralForge

> **Futuristic Neural Network Training, Visualization & Programmable Framework**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)

Built for extreme hardware — **RTX 5090 32GB | i9-284K Ultra | 128GB DDR5 6400MHz**

---

## 🚀 Features

- **Neural Network Studio** — Visual drag-and-drop network builder with real-time graph rendering
- **PyTorch + PyTorch Geometric** — Full graph neural network support (GNN, GCN, GAT, GraphSAGE)
- **ONNX Model Zoo** — Browse, download, and run 100+ pretrained ONNX models
- **Netron Integration** — In-browser model architecture visualization
- **Live Training Dashboard** — Real-time loss curves, GPU utilization, memory graphs
- **Programmable API** — Claude (Sonnet) AI assistant for network design and hyperparameter tuning
- **Bootstrap 5 UI** — Futuristic dark-mode interface with neon accents
- **Multi-GPU Ready** — Automatic DataParallel / DDP for RTX 5090

---

## 🏗️ Architecture

```
neural-forge/
├── neural_forge/
│   ├── core/           # Engine, device management, config
│   ├── models/         # Custom model definitions
│   ├── training/       # Trainer, schedulers, callbacks
│   ├── visualization/  # Plotly/D3 real-time charts
│   ├── graph/          # PyG integration layer
│   ├── onnx_utils/     # ONNX export, import, model zoo
│   └── ui/             # Flask backend for web UI
├── web/
│   ├── static/         # CSS, JS, assets
│   └── templates/      # Jinja2 HTML templates
├── configs/            # YAML training configs
├── scripts/            # CLI tools
├── notebooks/          # Jupyter examples
└── tests/              # Test suite
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/jmtibbetts/neural-forge.git
cd neural-forge

# 2. Install (CUDA 12.x + RTX 5090)
pip install -r requirements.txt

# 3. Launch the Studio
python -m neural_forge.ui.app

# 4. Open browser
# http://localhost:8080
```

---

## 🔧 Requirements

- Windows 11 Pro
- Python 3.11+
- CUDA 12.x drivers (for RTX 5090)
- 16GB+ RAM recommended (128GB supported)

---

## 📡 Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch 2.x |
| Graph Networks | PyTorch Geometric |
| Model Format | ONNX + ONNX Runtime |
| Model Viz | Netron |
| AI Assistant | Anthropic Claude (Sonnet) |
| Web UI | Flask + Bootstrap 5 |
| Charts | Plotly.js + Chart.js |
| Network Viz | D3.js / Cytoscape.js |
| GPU Monitoring | pynvml + psutil |

---

## 📄 License

MIT — build freely.
