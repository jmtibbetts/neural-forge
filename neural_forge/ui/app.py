"""
NeuralForge Studio - Flask + SocketIO UI
Real-time training dashboard for the Financial Brain.
"""

import json
import threading
from pathlib import Path

import torch
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from neural_forge.models.financial_brain import FinancialBrain
from neural_forge.training.trainer import Trainer
from neural_forge.training.data_pipeline import load_financial_data

app     = Flask(__name__, template_folder="../../web/templates")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─── Global training state ──────────────────────────────────
_training_thread  = None
_training_active  = False
_current_config   = {}


# ─── HTTP Routes ────────────────────────────────────────────

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
    return jsonify({
        "training": _training_active,
        "history":  history,
        "config":   _current_config,
    })


@app.route("/api/config", methods=["POST"])
def set_config():
    global _current_config
    _current_config = request.json
    return jsonify({"ok": True})


# ─── SocketIO events ────────────────────────────────────────

@socketio.on("start_training")
def handle_start_training(config):
    global _training_thread, _training_active, _current_config
    if _training_active:
        emit("error", {"msg": "Training already in progress."})
        return

    _current_config  = config
    _training_active = True

    def run():
        global _training_active
        try:
            ticker     = config.get("ticker",     "SPY")
            period     = config.get("period",      "5y")
            window     = int(config.get("window",  30))
            forward    = int(config.get("forward", 5))
            thresh     = float(config.get("thresh",0.015))
            epochs     = int(config.get("epochs",  30))
            lr         = float(config.get("lr",    3e-4))
            batch_size = int(config.get("batch",   64))
            hidden     = int(config.get("hidden",  128))
            layers     = int(config.get("layers",  2))

            socketio.emit("log", {"msg": f"Loading data for {ticker}..."})

            train_loader, val_loader, test_loader, feat_dim, class_weights = load_financial_data(
                ticker=ticker, period=period, window=window,
                forward=forward, thresh=thresh, batch_size=batch_size,
            )

            socketio.emit("log", {"msg": f"Data loaded. Features: {feat_dim}, training..."})

            model = FinancialBrain(
                input_size=feat_dim,
                hidden_size=hidden,
                num_layers=layers,
            )

            def on_epoch_end(metrics):
                socketio.emit("epoch_metrics", metrics)

            def on_batch_end(bm):
                socketio.emit("batch_metrics", bm)

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                lr=lr,
                epochs=epochs,
                on_epoch_end=on_epoch_end,
                on_batch_end=on_batch_end,
            )

            history = trainer.train()

            # Final eval
            trainer.load_best()
            report, cm = trainer.evaluate(test_loader)

            socketio.emit("training_complete", {
                "history": history,
                "report":  report,
                "confusion_matrix": cm.tolist(),
            })
            socketio.emit("log", {"msg": "Training complete! Best model saved."})

        except Exception as e:
            socketio.emit("error", {"msg": str(e)})
            import traceback
            socketio.emit("log", {"msg": traceback.format_exc()})
        finally:
            _training_active = False

    _training_thread = threading.Thread(target=run, daemon=True)
    _training_thread.start()
    emit("log", {"msg": "Training started in background thread."})


@socketio.on("stop_training")
def handle_stop():
    global _training_active
    _training_active = False
    emit("log", {"msg": "Stop requested (current epoch will finish)."})


# ─── Entry point ────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  NeuralForge Studio")
    print("  ==================")
    print("  Open: http://localhost:8080\n")
    socketio.run(app, host="0.0.0.0", port=8080, debug=False)
