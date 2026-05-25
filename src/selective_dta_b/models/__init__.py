from selective_dta_b.models.char_baseline import CharBaselineDTAModule, build_character_vocab, tokenize_character_sequences
from selective_dta_b.models.char_heteroscedastic import CharHeteroscedasticDTAModule
from selective_dta_b.models.deepdta import DeepDTDAModule
from selective_dta_b.models.graphdta import GraphDTDAModule
from selective_dta_b.models.moltrans import MolTransDTAModule


MODEL_REGISTRY = {
    "baseline": CharBaselineDTAModule,
    "deepdta": DeepDTDAModule,
    "graphdta": GraphDTDAModule,
    "heteroscedastic": CharHeteroscedasticDTAModule,
    "moltrans": MolTransDTAModule,
}


def build_model(model_type: str, **kwargs):
    if model_type not in MODEL_REGISTRY:
        raise KeyError(f"Unsupported model_type: {model_type}")
    return MODEL_REGISTRY[model_type](**kwargs)


__all__ = [
    "CharBaselineDTAModule",
    "CharHeteroscedasticDTAModule",
    "DeepDTDAModule",
    "GraphDTDAModule",
    "MolTransDTAModule",
    "MODEL_REGISTRY",
    "build_character_vocab",
    "build_model",
    "tokenize_character_sequences",
]

