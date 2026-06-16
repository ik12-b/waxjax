# tests/test_qwen2.py
"""Integration test end-to-end untuk arsitektur Qwen2."""

import pytest
import numpy as np
import jax.numpy as jnp
from flax.traverse_util import flatten_dict
import waxjax


class TestQwen2Load:

    def test_load_returns_nested_dict(self, qwen2_dir):
        params = waxjax.load(qwen2_dir)
        assert isinstance(params, dict)
        assert "model" in params

    def test_load_frozen_dict(self, qwen2_dir):
        from flax.core import FrozenDict
        params = waxjax.load(qwen2_dir, format="frozen")
        assert isinstance(params, FrozenDict)

    def test_embedding_shape(self, qwen2_dir):
        params = waxjax.load(qwen2_dir)
        embed  = np.array(params["model"]["embed_tokens"]["embedding"])
        # (V=512, H=64) — tidak di-transpose
        assert embed.shape == (512, 64)

    def test_linear_kernel_transposed(self, qwen2_dir):
        params = waxjax.load(qwen2_dir)
        flat   = flatten_dict(params)
        q_kernel = np.array(flat[("model","layers","0","self_attn","q_proj","kernel")])
        # PT shape (QH*HD=64, H=64) → Flax shape (H=64, QH*HD=64)
        assert q_kernel.shape == (64, 64)

    def test_rmsnorm_has_scale_not_kernel(self, qwen2_dir):
        params = waxjax.load(qwen2_dir)
        flat   = flatten_dict(params)
        assert ("model","layers","0","input_layernorm","scale") in flat
        assert ("model","layers","0","input_layernorm","kernel") not in flat

    def test_rmsnorm_no_bias(self, qwen2_dir):
        """Qwen2 pakai RMSNorm, tidak ada bias."""
        params = waxjax.load(qwen2_dir)
        flat   = flatten_dict(params)
        assert ("model","layers","0","input_layernorm","bias") not in flat

    def test_tied_lm_head_created(self, qwen2_dir):
        """lm_head.kernel harus ada meski tidak disimpan di file."""
        params = waxjax.load(qwen2_dir)
        flat   = flatten_dict(params)
        assert ("lm_head","kernel") in flat
        # Shape: (H=64, V=512)
        assert flat[("lm_head","kernel")].shape == (64, 512)

    def test_tied_lm_head_values_correct(self, qwen2_dir):
        """lm_head.kernel == embed.T secara numerik."""
        params = waxjax.load(qwen2_dir)
        flat   = flatten_dict(params)
        embed  = np.array(flat[("model","embed_tokens","embedding")])
        lm     = np.array(flat[("lm_head","kernel")])
        assert np.allclose(embed.T, lm, atol=1e-6)

    def test_all_layers_present(self, qwen2_dir):
        params = waxjax.load(qwen2_dir)
        flat   = flatten_dict(params)
        for i in range(2):  # NL=2 di fixture
            assert ("model","layers",str(i),"self_attn","q_proj","kernel") in flat
            assert ("model","layers",str(i),"mlp","gate_proj","kernel") in flat

    def test_all_values_jnp_array(self, qwen2_dir):
        params = waxjax.load(qwen2_dir)
        flat   = flatten_dict(params)
        for key, val in flat.items():
            assert isinstance(val, jnp.ndarray), f"{key} bukan jnp.ndarray"

    def test_no_nan_or_inf(self, qwen2_dir):
        params = waxjax.load(qwen2_dir)
        flat   = flatten_dict(params)
        for key, val in flat.items():
            arr = np.array(val)
            assert not np.any(np.isnan(arr)), f"NaN di {key}"
            assert not np.any(np.isinf(arr)), f"Inf di {key}"
