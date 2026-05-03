"""
NeuralForge Studio - Flask + SocketIO UI
Biological CorticalBrain + visualization dashboard
"""

import json
import threading
from pathlib import Path

import torch
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from neural_forge.core.engine import NeuralForgeEngine, ForgeConfig
from neural_forge.models.cortical_brain import CorticalBrain
from neural_forge.models.financial_brain import FinancialBrain
from neural_forge.training.trainer import Trainer
from neural_forge.training.data_pipeline import load_financial_data

app      = Flask(__name__, template_folder="../../web/templates")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

_engine          = NeuralForgeEngine(ForgeConfig())
_training_active = False
_current_config  = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    history_path = Path("checkpoints/history.json")
    history = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    return jsonify({"training": _training_active, "history": history, "config": _current_config})


@app.route("/api/system")
def system_info():
    info = _engine.info()
    best = Path("checkpoints/best_model.pt")
    onnx = Path("checkpoints/best_model.onnx")
    info["checkpoint_exists"] = best.exists()
    info["onnx_exists"]       = onnx.exists()
    if best.exists():
        ckpt = torch.load(best, map_location="cpu", weights_only=True)
        m = ckpt.get("metrics", {})
        info["best_val_acc"] = m.get("val_acc", None)
        info["best_epoch"]   = m.get("epoch",   None)
    return jsonify(info)


@app.route("/api/export_onnx", methods=["POST"])
def export_onnx():
    try:
        best = Path("checkpoints/best_model.pt")
        if not best.exists():
            return jsonify({"ok": False, "error": "No checkpoint found. Train first."})

        cfg    = request.json or {}
        window = int(cfg.get("window", 30))
        feat   = int(cfg.get("feat_dim", 11))
        col_dim    = int(cfg.get("col_dim", 64))
        memory_dim = int(cfg.get("memory_dim", 128))
        model_type = cfg.get("model_type", "cortical")

        if model_type == "cortical":
            model = CorticalBrain(input_size=feat, col_dim=col_dim,
                                  memory_dim=memory_dim, seq_len=window)
        else:
            hidden = int(cfg.get("hidden", 128))
            layers = int(cfg.get("layers", 2))
            model  = FinancialBrain(input_size=feat, hidden_size=hidden, num_layers=layers)

        ckpt = torch.load(best, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        dummy    = torch.randn(1, window, feat)
        out_path = Path("checkpoints/best_model.onnx")

        # CorticalBrain returns 3 outputs — wrap for ONNX
        class OnnxWrapper(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                logits, attn, _ = self.m(x)
                return logits, attn

        export_model = OnnxWrapper(model) if model_type == "cortical" else model

        torch.onnx.export(
            export_model, (dummy,), str(out_path),
            input_names=["candles"], output_names=["logits", "attention"],
            dynamic_axes={"candles": {0: "batch"}}, opset_version=17,
        )
        return jsonify({"ok": True, "path": str(out_path)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@socketio.on("start_training")
def handle_start_training(config):
    global _training_active, _current_config
    if _training_active:
        emit("error", {"msg": "Training already in progress."})
        return

    _training_active = True
    _current_config  = config

    def run():
        global _training_active
        try:
            ticker      = config.get("ticker",  "SPY")
            period      = config.get("period",   "5y")
            window      = int(config.get("window",    30))
            forward     = int(config.get("forward",    5))
            thresh      = float(config.get("thresh", 0.015))
            epochs      = int(config.get("epochs",    30))
            lr          = float(config.get("lr",     3e-4))
            batch_size  = int(config.get("batch",     64))
            col_dim     = int(config.get("col_dim",   64))
            memory_dim  = int(config.get("memory_dim",128))
            sparsity_k  = int(config.get("sparsity_k", 8))
            probe_every = int(config.get("probe_every", 5))
            model_type  = config.get("model_type", "cortical")

            socketio.emit("log", {"msg": f"Downloading {ticker} ({period})..."})

            train_loader, val_loader, test_loader, feat_dim, class_weights = load_financial_data(
                ticker=ticker, period=period, window=window,
                forward=forward, thresh=thresh, batch_size=batch_size,
            )

            socketio.emit("log", {"msg": f"Data ready. Features={feat_dim}. Building brain..."})

            if model_type == "cortical":
                model = CorticalBrain(
                    input_size=feat_dim, col_dim=col_dim,
                    memory_dim=memory_dim, seq_len=window,
                    sparsity_k=sparsity_k,
                )
                socketio.emit("log", {"msg": f"🧠 CorticalBrain: Thalamus → 6 Columns → Hippocampus → Dopamine → PFC"})
            else:
                hidden = int(config.get("hidden", 128))
                layers = int(config.get("layers",   2))
                model  = FinancialBrain(input_size=feat_dim, hidden_size=hidden, num_layers=layers)
                socketio.emit("log", {"msg": f"Using FinancialBrain (LSTM)"})

            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                class_weights=class_weights, lr=lr, epochs=epochs,
                probe_every=probe_every,
                on_epoch_end   = lambda m: socketio.emit("epoch_metrics",  m),
                on_batch_end   = lambda m: socketio.emit("batch_metrics",   m),
                on_brain_state = lambda s: socketio.emit("brain_state",     s),
            )

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            socketio.emit("log", {"msg": f"Training on {trainer.device} | {n_params:,} params"})

            history = trainer.train()
            trainer.load_best()
            report, cm = trainer.evaluate(test_loader)

            socketio.emit("training_complete", {
                "history": history, "report": report,
                "confusion_matrix": cm.tolist(),
                "feat_dim": feat_dim, "window": window,
                "col_dim": col_dim, "memory_dim": memory_dim,
                "model_type": model_type,
            })
            socketio.emit("log", {"msg": "✅ Done! Best model saved to checkpoints/best_model.pt"})

        except Exception as e:
            import traceback
            socketio.emit("error", {"msg": str(e)})
            socketio.emit("log",   {"msg": traceback.format_exc()})
        finally:
            _training_active = False

    threading.Thread(target=run, daemon=True).start()
    emit("log", {"msg": "Training thread started."})


@socketio.on("stop_training")
def handle_stop():
    global _training_active
    _training_active = False
    emit("log", {"msg": "Stop requested."})


if __name__ == "__main__":
    print("\n  NeuralForge Studio — CorticalBrain Edition")
    print("  Open: http://localhost:8080\n")
    socketio.run(app, host="0.0.0.0", port=8080, debug=False)
