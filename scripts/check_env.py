from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata

PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "biopython",
    "transformers",
    "lightning",
    "torch",
]


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def nvidia_smi() -> str:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return "nvidia-smi not found"
    try:
        return subprocess.check_output(
            [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:
        return f"nvidia-smi failed: {exc}"


def torch_info() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {name: package_version(name) for name in PACKAGE_NAMES},
    "nvidia_smi": nvidia_smi(),
    "torch": torch_info(),
}
print(json.dumps(payload, indent=2, ensure_ascii=False))
