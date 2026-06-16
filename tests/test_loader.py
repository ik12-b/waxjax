# tests/test_loader.py
"""Test loader: baca safetensors, handle bfloat16, sharding."""

import pytest
import numpy as np
from pathlib import Path
from waxjax.core.loader import load_safetensors


class TestLoader:

    def test_load_basic(self, qwen2_dir):
        weights = load_safetensors(qwen2_dir)
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_all_float32(self, qwen2_dir):
        """Semua tensor harus float32 setelah load."""
        weights = load_safetensors(qwen2_dir)
        for key, arr in weights.items():
            assert arr.dtype == np.float32, f"{key} bukan float32: {arr.dtype}"

    def test_expected_keys_present_qwen2(self, qwen2_dir):
        weights = load_safetensors(qwen2_dir)
        assert "model.embed_tokens.weight" in weights
        assert "model.layers.0.self_attn.q_proj.weight" in weights
        assert "model.layers.0.self_attn.q_proj.bias" in weights
        assert "model.norm.weight" in weights

    def test_expected_keys_present_nllb(self, nllb_dir):
        weights = load_safetensors(nllb_dir)
        assert "model.shared.weight" in weights
        assert "model.encoder.layers.0.self_attn.q_proj.weight" in weights
        assert "model.decoder.layers.0.encoder_attn.q_proj.weight" in weights
        assert "model.encoder.embed_positions.weight" in weights

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_safetensors(str(tmp_path / "nonexistent"))

    def test_bfloat16_handling(self, tmp_path):
        """Tensor bfloat16 harus di-cast ke float32 tanpa error."""
        import torch
        from safetensors.torch import save_file

        bf16_tensor = torch.randn(4, 8).to(torch.bfloat16)
        save_file({"layer.weight": bf16_tensor}, str(tmp_path / "model.safetensors"))

        weights = load_safetensors(str(tmp_path))
        assert weights["layer.weight"].dtype == np.float32

    def test_sharded_loading(self, tmp_path):
        """Dua file safetensors di-merge dengan benar."""
        import torch
        from safetensors.torch import save_file

        save_file({"shard1.weight": torch.randn(4, 4)},
                  str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file({"shard2.weight": torch.randn(4, 4)},
                  str(tmp_path / "model-00002-of-00002.safetensors"))

        weights = load_safetensors(str(tmp_path))
        assert "shard1.weight" in weights
        assert "shard2.weight" in weights
        assert len(weights) == 2
