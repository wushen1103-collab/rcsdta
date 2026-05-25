from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class GPUStatus:
    index: int
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_gpu: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_nvidia_smi_query(payload: str) -> list[GPUStatus]:
    gpus: list[GPUStatus] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Unexpected nvidia-smi line: {line}")
        gpus.append(
            GPUStatus(
                index=int(parts[0]),
                name=parts[1],
                memory_used_mib=int(parts[2].replace("MiB", "").strip()),
                memory_total_mib=int(parts[3].replace("MiB", "").strip()),
                utilization_gpu=int(parts[4].replace("%", "").strip()),
            )
        )
    return gpus


def query_nvidia_gpus() -> list[GPUStatus]:
    try:
        payload = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return parse_nvidia_smi_query(payload)


def select_idle_gpus(
    gpus: list[GPUStatus],
    max_memory_used_mib: int = 1024,
    max_utilization_gpu: int = 10,
) -> list[int]:
    return [
        gpu.index
        for gpu in gpus
        if gpu.memory_used_mib <= max_memory_used_mib and gpu.utilization_gpu <= max_utilization_gpu
    ]


def recommend_num_workers_per_job(
    logical_cpus: int,
    concurrent_jobs: int,
    reserve_cpus: int = 8,
    max_workers_per_job: int = 16,
) -> int:
    if concurrent_jobs <= 0:
        raise ValueError("concurrent_jobs must be positive")
    usable_cpus = max(1, logical_cpus - reserve_cpus)
    workers = max(1, usable_cpus // concurrent_jobs)
    return min(max_workers_per_job, workers)


def _read_memory_info_gb() -> tuple[float | None, float | None]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None, None
    values: dict[str, int] = {}
    for line in meminfo.read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    total_gb = values.get("MemTotal", 0) / 1024 / 1024
    available_gb = values.get("MemAvailable", 0) / 1024 / 1024
    return total_gb, available_gb


def build_resource_snapshot(workspace: str | Path) -> dict[str, object]:
    workspace_path = Path(workspace)
    logical_cpus = os.cpu_count() or 1
    total_memory_gb, available_memory_gb = _read_memory_info_gb()
    disk = shutil.disk_usage(workspace_path)
    gpus = query_nvidia_gpus()
    idle_gpu_indices = select_idle_gpus(gpus)
    concurrent_gpu_jobs = max(1, len(idle_gpu_indices)) if idle_gpu_indices else 1

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace_path.resolve()),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "cpu": {
            "logical_cores": logical_cpus,
        },
        "memory": {
            "total_gb": total_memory_gb,
            "available_gb": available_memory_gb,
        },
        "disk": {
            "total_gb": disk.total / 1024 / 1024 / 1024,
            "available_gb": disk.free / 1024 / 1024 / 1024,
        },
        "gpu": {
            "devices": [gpu.to_dict() for gpu in gpus],
            "idle_gpu_indices": idle_gpu_indices,
            "total_gpus": len(gpus),
        },
        "recommendations": {
            "suggested_concurrent_gpu_jobs": len(idle_gpu_indices),
            "suggested_num_workers_per_job": recommend_num_workers_per_job(
                logical_cpus=logical_cpus,
                concurrent_jobs=concurrent_gpu_jobs,
                reserve_cpus=24 if logical_cpus >= 128 else 8,
                max_workers_per_job=16,
            ),
        },
    }
