# tests/test_nllb.py
"""Integration test end-to-end untuk arsitektur NLLB/M2M-100."""

import pytest
import numpy as np
from flax.traverse_util import flatten_dict
import waxjax


class TestNLLBLoad:

    def test_load_success(self, nllb_dir):
        params = waxjax.load(nllb_dir)
        assert isinstance(params, dict)

    def test_shared_embedding_shape(self, nllb_dir):
        """model.shared.embedding shape (V, D) tidak di-transpose."""
        params = waxjax.load(nllb_dir)
        embed  = np.array(params["model"]["shared"]["embedding"])
        assert embed.shape == (512, 64)

    def test_tied_embed_tokens_skipped(self, nllb_dir):
        """encoder/decoder embed_tokens tidak masuk params (tied ke shared)."""
        params = waxjax.load(nllb_dir)
        flat   = flatten_dict(params)
        assert ("model","encoder","embed_tokens","embedding") not in flat
        assert ("model","decoder","embed_tokens","embedding") not in flat

    def test_positional_embedding_present(self, nllb_dir):
        params = waxjax.load(nllb_dir)
        flat   = flatten_dict(params)
        assert ("model","encoder","embed_positions","embedding") in flat
        assert ("model","decoder","embed_positions","embedding") in flat
        # Shape (MAXPOS+2, D) = (132, 64)
        pos = np.array(flat[("model","encoder","embed_positions","embedding")])
        assert pos.shape == (132, 64)

    def test_layernorm_has_scale_and_bias(self, nllb_dir):
        """NLLB pakai LayerNorm → punya scale DAN bias (beda dari Qwen2 RMSNorm)."""
        params = waxjax.load(nllb_dir)
        flat   = flatten_dict(params)
        assert ("model","encoder","layers","0","self_attn_layer_norm","scale") in flat
        assert ("model","encoder","layers","0","self_attn_layer_norm","bias")  in flat

    def test_fc1_fc2_kernel_transposed(self, nllb_dir):
        """fc1/fc2 harus di-transpose: PT (FFN,D) → Flax (D,FFN)."""
        params = waxjax.load(nllb_dir)
        flat   = flatten_dict(params)
        fc1 = np.array(flat[("model","encoder","layers","0","fc1","kernel")])
        fc2 = np.array(flat[("model","encoder","layers","0","fc2","kernel")])
        # D=64, FFN=128
        assert fc1.shape == (64, 128)   # transposed dari (128,64)
        assert fc2.shape == (128, 64)   # transposed dari (64,128)

    def test_out_proj_not_o_proj(self, nllb_dir):
        """NLLB pakai out_proj bukan o_proj seperti Qwen2."""
        params = waxjax.load(nllb_dir)
        flat   = flatten_dict(params)
        assert ("model","encoder","layers","0","self_attn","out_proj","kernel") in flat
        assert ("model","encoder","layers","0","self_attn","o_proj","kernel") not in flat

    def test_cross_attention_present(self, nllb_dir):
        """Decoder harus punya encoder_attn (cross-attention)."""
        params = waxjax.load(nllb_dir)
        flat   = flatten_dict(params)
        assert ("model","decoder","layers","0","encoder_attn","q_proj","kernel") in flat
        assert ("model","decoder","layers","0","encoder_attn_layer_norm","scale") in flat

    def test_encoder_has_no_cross_attention(self, nllb_dir):
        """Encoder tidak punya encoder_attn."""
        params = waxjax.load(nllb_dir)
        flat   = flatten_dict(params)
        assert ("model","encoder","layers","0","encoder_attn","q_proj","kernel") not in flat

    def test_lm_head_tied_to_shared(self, nllb_dir):
        """lm_head.kernel == model.shared.embedding.T"""
        params = waxjax.load(nllb_dir)
        flat   = flatten_dict(params)
        shared = np.array(flat[("model","shared","embedding")])  # (V,D)
        lm     = np.array(flat[("lm_head","kernel")])            # (D,V)
        assert lm.shape == (64, 512)
        assert np.allclose(shared.T, lm, atol=1e-6)

    def test_both_encoder_decoder_layers(self, nllb_dir):
        params = waxjax.load(nllb_dir)
        flat   = flatten_dict(params)
        for i in range(2):
            # Encoder
            assert ("model","encoder","layers",str(i),"self_attn","q_proj","kernel") in flat
            assert ("model","encoder","layers",str(i),"fc1","kernel") in flat
            # Decoder
            assert ("model","decoder","layers",str(i),"self_attn","q_proj","kernel") in flat
            assert ("model","decoder","layers",str(i),"encoder_attn","q_proj","kernel") in flat