"""Correctness tests: each engine function is checked against a reference.

Run from the repo root with `pytest tests/`.
Set MODEL_DIR to point at a local Qwen3-0.6B snapshot; otherwise the files are
fetched from the HuggingFace hub (cached after the first run).
"""

import os
from huggingface_hub import snapshot_download
from engine.config import ModelConfig

def model_dir() -> str:
    if "MODEL_DIR" in os.environ:
        return os.environ["MODEL_DIR"]
    return snapshot_download("Qwen/Qwen3-0.6B")

def test_config() -> None:
    config = model_dir() + "/config.json"
    assert os.path.exists(config)

    cfg = ModelConfig.from_json(config)
    assert cfg.hidden == 1024
    assert cfg.n_kv_heads == 8
    assert cfg.rope_theta == 1000000
