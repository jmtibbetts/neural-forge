"""
NeuralForge Web UI
Flask + SocketIO app — futuristic dark-mode neural network studio
"""

import os
import json
import threading
import torch
import psutil
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

logger = logging.getLogger("neural_forge.ui")

# Paths
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR.parent.parent / "web" / "templates"
STATIC_DIR = BASE_DIR.parent.parent / "web" / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
app.config["SECRET_KEY"] = os.urandom(24).hex()
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ─── Global state ─────────────────────────────────────────────────────────────
_training_active = False
_training_metrics = []
_loaded_model = None


# ─── REST Endpoints ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/system")
def system_info():
    info = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 1),
        "ram_percent": psutil.virtual_memory().percent,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_vram_total_gb": round(mem.total / 1e9, 1),
                "gpu_vram_used_gb": round(mem.used / 1e9, 1),
                "gpu_util_percent": util.gpu,
                "gpu_mem_percent": util.memory,
                "gpu_temp_c": temp,
            })
        except Exception as e:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
            info["gpu_vram_used_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    return jsonify(info)


@app.route("/api/onnx/models")
def onnx_models():
    from neural_forge.onnx_utils import ONNXModelZoo
    zoo = ONNXModelZoo()
    category = request.args.get("category")
    return jsonify(zoo.list_models(category=category))


@app.route("/api/onnx/download/<name>", methods=["POST"])
def onnx_download(name):
    from neural_forge.onnx_utils import ONNXModelZoo
    zoo = ONNXModelZoo()
    try:
        path = zoo.download(name)
        return jsonify({"status": "ok", "path": str(path)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/onnx/info/<name>")
def onnx_info(name):
    from neural_forge.onnx_utils import ONNXModelZoo
    zoo = ONNXModelZoo()
    return jsonify(zoo.model_info(name))


@app.route("/api/netron/serve/<path:model_path>")
def netron_serve(model_path):
    """Serve a model file for Netron visualization."""
    return send_from_directory("/", model_path)


@app.route("/api/assistant/chat", methods=["POST"])
def assistant_chat():
    """Route to Claude Sonnet for AI-assisted network design."""
    data = request.json
    message = data.get("message", "")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 400
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=(
                "You are NeuralForge AI, an expert neural network architect and trainer. "
                "Help the user design, debug, and optimize neural networks. "
                "Be concise and practical. Use PyTorch and PyTorch Geometric syntax."
            ),
            messages=[{"role": "user", "content": message}],
        )
        return jsonify({"response": response.content[0].text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/metrics")
def training_metrics():
    return jsonify({"metrics": _training_metrics, "active": _training_active})


# ─── SocketIO Events ──────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    logger.info(f"Client connected: {request.sid}")
    emit("status", {"msg": "Connected to NeuralForge"})


@socketio.on("start_training")
def on_start_training(data):
    global _training_active
    if _training_active:
        emit("error", {"msg": "Training already in progress"})
        return
    _training_active = True
    emit("training_started", {"msg": "Training started"})
    # Training runs in background thread
    thread = threading.Thread(target=_mock_training_loop, args=(data,), daemon=True)
    thread.start()


@socketio.on("stop_training")
def on_stop_training():
    global _training_active
    _training_active = False
    emit("training_stopped", {"msg": "Training stopped"})


def _mock_training_loop(config):
    """Demo training loop — replace with real Trainer integration."""
    global _training_active, _training_metrics
    import time, math, random
    epochs = config.get("epochs", 20)
    for epoch in range(1, epochs + 1):
        if not _training_active:
            break
        time.sleep(0.5)
        step_noise = random.gauss(0, 0.01)
        train_loss = 2.5 * math.exp(-0.15 * epoch) + 0.1 + step_noise
        val_loss = 2.6 * math.exp(-0.13 * epoch) + 0.12 + step_noise * 1.2
        metrics = {
            "epoch": epoch,
            "total_epochs": epochs,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "train_acc": round(min(0.99, 0.3 + epoch * 0.035 + step_noise), 4),
            "val_acc": round(min(0.97, 0.28 + epoch * 0.033 + step_noise), 4),
            "lr": round(1e-3 * (0.95 ** epoch), 7),
            "gpu_mem_gb": round(random.uniform(4, 18), 2),
            "gpu_util": random.randint(70, 99),
            "gpu_temp": random.randint(65, 82),
        }
        _training_metrics.append(metrics)
        socketio.emit("training_update", metrics)
    _training_active = False
    socketio.emit("training_complete", {"msg": "Training complete!", "epochs": epochs})


# ─── Entry Point ──────────────────────────────────────────────────────────────

def run(host="127.0.0.1", port=8080, debug=False):
    print(f"\n⚡ NeuralForge Studio → http://{host}:{port}\n")
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()
