"""Model hyperparameters, read from the HuggingFace config.json.

Nothing is hardcoded here. The model's own config file is the source of truth.
"""

import json
from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    rms_eps: float
    rope_theta: float
    tie_embeddings: bool
    max_pos: int
    eos_id: int

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            d = json.load(f)

        return cls(
            n_layers=d["num_hidden_layers"],
            hidden=d["hidden_size"],
            n_heads=d["num_attention_heads"],
            n_kv_heads=d["num_key_value_heads"],
            head_dim=d["head_dim"],
            intermediate=d["intermediate_size"],
            vocab=d["vocab_size"],
            rms_eps=d["rms_norm_eps"],
            rope_theta=d["rope_theta"],
            tie_embeddings=d["tie_word_embeddings"],
            max_pos=d["max_position_embeddings"],
            eos_id=d["eos_token_id"],
        )
