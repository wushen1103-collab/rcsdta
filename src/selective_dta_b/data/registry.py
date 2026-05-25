from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    file_format: str
    affinity_measure: str
    affinity_unit: str
    model_target: str
    primary_split: str
    raw_subdir: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def get_default_registry() -> dict[str, DatasetSpec]:
    return {
        "bindingdb": DatasetSpec(
            name="bindingdb",
            task="dta",
            file_format="csv",
            affinity_measure="Kd",
            affinity_unit="nM",
            model_target="pKd",
            primary_split="unseen_target",
            raw_subdir="bindingdb",
            description="Primary large-scale DTA dataset for B experiment.",
        ),
        "davis": DatasetSpec(
            name="davis",
            task="dta",
            file_format="csv",
            affinity_measure="Kd",
            affinity_unit="nM",
            model_target="pKd",
            primary_split="unseen_target",
            raw_subdir="davis",
            description="Compact kinase-focused benchmark for reproducible evaluation.",
        ),
        "kiba": DatasetSpec(
            name="kiba",
            task="dta",
            file_format="csv",
            affinity_measure="KIBA",
            affinity_unit="score",
            model_target="KIBA_score",
            primary_split="unseen_target",
            raw_subdir="kiba",
            description="Benchmark with KIBA score labels for auxiliary evaluation.",
        ),
    }
