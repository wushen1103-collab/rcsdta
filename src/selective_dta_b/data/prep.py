from __future__ import annotations

import json
from pathlib import Path

from selective_dta_b.data.registry import DatasetSpec


def prepare_dataset_workspace(workspace: str, dataset: DatasetSpec) -> dict[str, str]:
    workspace_path = Path(workspace)
    dataset_dir = workspace_path / "data" / "raw" / dataset.raw_subdir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = dataset_dir / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(dataset.to_dict(), indent=2))

    return {
        "dataset_dir": str(dataset_dir),
        "metadata_path": str(metadata_path),
    }
