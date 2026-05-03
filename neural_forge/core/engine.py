"""
NeuralForge Core Engine
Manages device selection, global config, and framework lifecycle.
"""

import os
import warnings
# Suppress deprecated pynvml warning — use nvidia-ml-py3 instead
warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()
logger = logging.getLogger("neural_forge")


@dataclass
class ForgeConfig:
    """Global framework configuration."""
    device: str = "auto"                    # "auto" | "cuda" | "cpu" | "cuda:0"
    precision: str = "bf16"                 # "fp32" | "fp16" | "bf16"
    compile_model: bool = True              # torch.compile (Inductor backend)
    num_workers: int = 16                   # DataLoader workers
    pin_memory: bool = True
    seed: int = 42
    log_level: str = "INFO"
    anthropic_api_key: Optional[str] = None


class NeuralForgeEngine:
    """
    The main NeuralForge engine.
    Initializes device, logging, and global state.
    """

    def __init__(self, config: Optional[ForgeConfig] = None):
        self.config = config or ForgeConfig()
        self._setup_logging()
        self.device = self._resolve_device()
        self._print_banner()

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def _resolve_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
            else:
                device = torch.device("cpu")
                logger.warning("No CUDA GPU detected — running on CPU")
        else:
            device = torch.device(self.config.device)

        # Set global seed
        torch.manual_seed(self.config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

        return device

    def _print_banner(self):
        gpu_info = "N/A"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_info = f"{gpu_name} ({vram:.1f} GB)"

        text = Text()
        text.append("⚡ NeuralForge v0.1.0\n", style="bold cyan")
        text.append(f"  Device   : {self.device}\n", style="green")
        text.append(f"  GPU      : {gpu_info}\n", style="green")
        text.append(f"  Precision: {self.config.precision}\n", style="green")
        text.append(f"  Compile  : {self.config.compile_model}\n", style="green")
        text.append(f"  PyTorch  : {torch.__version__}", style="dim")

        console.print(Panel(text, border_style="cyan", padding=(0, 2)))

    @property
    def dtype(self) -> torch.dtype:
        mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        return mapping.get(self.config.precision, torch.float32)

    def info(self) -> dict:
        """Return system info dict."""
        info = {
            "device": str(self.device),
            "precision": self.config.precision,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update({
                "gpu_name": props.name,
                "gpu_vram_gb": round(props.total_memory / 1e9, 2),
                "gpu_sm_count": props.multi_processor_count,
                "cuda_version": torch.version.cuda,
            })
        return info
