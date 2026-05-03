"""
NeuralForge ONNX Model Zoo
Browse, download, and run models from the official ONNX Model Zoo.
https://github.com/onnx/models
"""

import os
import requests
import onnx
import onnxruntime as ort
import numpy as np
from typing import Optional, Dict, List, Any
from pathlib import Path
import logging

logger = logging.getLogger("neural_forge.onnx_zoo")

# Official ONNX Model Zoo catalog (curated subset)
ONNX_MODEL_ZOO = {
    # Vision - Classification
    "resnet50": {
        "category": "vision/classification",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx",
        "description": "ResNet-50 v2 — ImageNet classification",
        "inputs": {"data": (1, 3, 224, 224)},
        "opset": 7,
    },
    "mobilenet_v2": {
        "category": "vision/classification",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx",
        "description": "MobileNet V2 — Lightweight ImageNet classification",
        "inputs": {"input": (1, 3, 224, 224)},
        "opset": 10,
    },
    "efficientnet_lite4": {
        "category": "vision/classification",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "description": "EfficientNet Lite 4 — Efficient ImageNet classification",
        "inputs": {"images:0": (1, 224, 224, 3)},
        "opset": 11,
    },
    "squeezenet": {
        "category": "vision/classification",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
        "description": "SqueezeNet 1.1 — Compact ImageNet classification",
        "inputs": {"data": (1, 3, 224, 224)},
        "opset": 7,
    },
    # Vision - Detection
    "yolov3": {
        "category": "vision/detection",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx",
        "description": "YOLOv3 — Real-time object detection",
        "inputs": {"input_1": (1, 3, 416, 416), "image_shape": (1, 2)},
        "opset": 10,
    },
    "ssd": {
        "category": "vision/detection",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd/model/ssd-10.onnx",
        "description": "SSD — Single Shot MultiBox Detector",
        "inputs": {"image": (1, 3, 1200, 1200)},
        "opset": 10,
    },
    # NLP
    "bert_base": {
        "category": "nlp/language_model",
        "url": "https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx",
        "description": "BERT Base — SQuAD question answering",
        "inputs": {},
        "opset": 10,
    },
    # Body Analysis
    "emotion_ferplus": {
        "category": "vision/body_analysis",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        "description": "FER+ Emotion Recognition — 8 emotion classes",
        "inputs": {"Input3": (1, 1, 64, 64)},
        "opset": 8,
    },
    "age_gender": {
        "category": "vision/body_analysis",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/age_gender/models/age_googlenet.onnx",
        "description": "Age estimation from face",
        "inputs": {"data": (1, 3, 224, 224)},
        "opset": 8,
    },
}


class ONNXModelZoo:
    """
    Browse, download, and run ONNX Model Zoo models.
    """

    def __init__(self, cache_dir: str = "./models/onnx_zoo"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, ort.InferenceSession] = {}

    def list_models(self, category: Optional[str] = None) -> List[Dict]:
        """List available models, optionally filtered by category."""
        models = []
        for name, info in ONNX_MODEL_ZOO.items():
            if category and not info["category"].startswith(category):
                continue
            cached = self._model_path(name).exists()
            models.append({
                "name": name,
                "category": info["category"],
                "description": info["description"],
                "opset": info["opset"],
                "cached": cached,
            })
        return models

    def download(self, name: str, force: bool = False) -> Path:
        """Download a model from the ONNX Model Zoo."""
        if name not in ONNX_MODEL_ZOO:
            raise ValueError(f"Unknown model: {name}. Call list_models() to see available models.")

        path = self._model_path(name)
        if path.exists() and not force:
            logger.info(f"Model '{name}' already cached at {path}")
            return path

        url = ONNX_MODEL_ZOO[name]["url"]
        logger.info(f"Downloading {name} from ONNX Model Zoo...")

        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Downloading {name}: {pct:.1f}% ({downloaded//1024//1024} MB)", end="")
        print()
        logger.info(f"Downloaded '{name}' to {path}")
        return path

    def load(self, name: str, providers: Optional[List[str]] = None) -> ort.InferenceSession:
        """Load a model into an ONNX Runtime session."""
        if name in self._sessions:
            return self._sessions[name]

        path = self._model_path(name)
        if not path.exists():
            path = self.download(name)

        if providers is None:
            available = ort.get_available_providers()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

        session = ort.InferenceSession(str(path), providers=providers)
        self._sessions[name] = session
        logger.info(f"Loaded '{name}' | providers: {providers}")
        return session

    def run(self, name: str, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Run inference with a zoo model."""
        session = self.load(name)
        return session.run(None, inputs)

    def model_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a model."""
        path = self._model_path(name)
        info = dict(ONNX_MODEL_ZOO.get(name, {}))
        info["cached"] = path.exists()
        if path.exists():
            model = onnx.load(str(path))
            info["ir_version"] = model.ir_version
            info["graph_nodes"] = len(model.graph.node)
            info["inputs"] = {i.name: list(d.dim_value for d in i.type.tensor_type.shape.dim)
                              for i in model.graph.input}
            info["outputs"] = [o.name for o in model.graph.output]
        return info

    def export_pytorch(
        self,
        model,
        dummy_input,
        name: str,
        opset: int = 17,
        output_dir: str = "./models/exports",
    ) -> Path:
        """Export a PyTorch model to ONNX format."""
        import torch
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{name}.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            str(path),
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        logger.info(f"Exported to ONNX: {path}")
        return path

    def _model_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.onnx"
