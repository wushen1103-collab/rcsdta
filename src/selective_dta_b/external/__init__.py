__all__ = [
    "KANPM_DATASET_NAME_MAP",
    "KANPM_RUNNING_SET_NAME_MAP",
    "build_contact_map_payload",
    "build_embedding_payload",
    "ensure_deepdtagen_sequence_capacity",
    "install_fairseq_shim",
    "load_kanpm_drug_records",
    "load_kanpm_protein_records",
    "load_deepdtagen_modules",
    "materialize_deepdtagen_dataset",
    "load_deepdtagen_modules",
    "materialize_kanpm_dataset",
    "prepare_deepdtagen_data",
    "resolve_kanpm_pretrained_paths",
    "save_pickle",
    "stage_deepdtagen_split",
    "stage_deepdtagen_split",
]


def __getattr__(name: str):
    if name == "materialize_deepdtagen_dataset":
        from selective_dta_b.external.deepdtagen import materialize_deepdtagen_dataset

        return materialize_deepdtagen_dataset

    if name in {
        "ensure_deepdtagen_sequence_capacity",
        "install_fairseq_shim",
        "load_deepdtagen_modules",
        "prepare_deepdtagen_data",
        "stage_deepdtagen_split",
    }:
        from selective_dta_b.external.deepdtagen_runtime import (
            ensure_deepdtagen_sequence_capacity,
            install_fairseq_shim,
            load_deepdtagen_modules,
            prepare_deepdtagen_data,
            stage_deepdtagen_split,
        )

        return {
            "ensure_deepdtagen_sequence_capacity": ensure_deepdtagen_sequence_capacity,
            "install_fairseq_shim": install_fairseq_shim,
            "load_deepdtagen_modules": load_deepdtagen_modules,
            "prepare_deepdtagen_data": prepare_deepdtagen_data,
            "stage_deepdtagen_split": stage_deepdtagen_split,
        }[name]

    if name in {
        "KANPM_DATASET_NAME_MAP",
        "KANPM_RUNNING_SET_NAME_MAP",
        "materialize_kanpm_dataset",
    }:
        from selective_dta_b.external.kanpm import (
            KANPM_DATASET_NAME_MAP,
            KANPM_RUNNING_SET_NAME_MAP,
            materialize_kanpm_dataset,
        )

        return {
            "KANPM_DATASET_NAME_MAP": KANPM_DATASET_NAME_MAP,
            "KANPM_RUNNING_SET_NAME_MAP": KANPM_RUNNING_SET_NAME_MAP,
            "materialize_kanpm_dataset": materialize_kanpm_dataset,
        }[name]

    if name in {
        "build_contact_map_payload",
        "build_embedding_payload",
        "load_kanpm_drug_records",
        "load_kanpm_protein_records",
        "resolve_kanpm_pretrained_paths",
        "save_pickle",
    }:
        from selective_dta_b.external.kanpm_pretrain import (
            build_contact_map_payload,
            build_embedding_payload,
            load_kanpm_drug_records,
            load_kanpm_protein_records,
            resolve_kanpm_pretrained_paths,
            save_pickle,
        )

        return {
            "build_contact_map_payload": build_contact_map_payload,
            "build_embedding_payload": build_embedding_payload,
            "load_kanpm_drug_records": load_kanpm_drug_records,
            "load_kanpm_protein_records": load_kanpm_protein_records,
            "resolve_kanpm_pretrained_paths": resolve_kanpm_pretrained_paths,
            "save_pickle": save_pickle,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

