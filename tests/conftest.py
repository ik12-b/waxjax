# ╔══════════════════════════════════════════════════════════════╗
# ║   TEST SUITE — waxjax                                        ║
# ║   Jalankan: pytest tests/ -v                                 ║
# ╚══════════════════════════════════════════════════════════════╝

# tests/conftest.py
"""
Shared fixtures untuk semua test.
Download model kecil sekali, cache untuk semua test.
"""

import pytest
import json
import numpy as np
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qwen2_dir(tmp_path_factory):
    """
    Buat fake Qwen2 model directory dengan tensor sintetis.
    Tidak perlu download model asli — shape dan key names sudah cukup
    untuk test mapping logic.
    """
    import numpy as np
    from safetensors.torch import save_file
    import torch

    model_dir = tmp_path_factory.mktemp("qwen2")

    # Config mirip Qwen2.5-0.5B tapi kecil
    config = {
        "model_type"          : "qwen2",
        "architectures"       : ["Qwen2ForCausalLM"],
        "hidden_size"         : 64,
        "intermediate_size"   : 128,
        "num_hidden_layers"   : 2,
        "num_attention_heads" : 4,
        "num_key_value_heads" : 2,
        "vocab_size"          : 512,
        "rms_norm_eps"        : 1e-6,
        "tie_word_embeddings" : True,
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    H, MLP, V, QH, KVH, HD, NL = 64, 128, 512, 4, 2, 16, 2

    tensors = {}
    tensors["model.embed_tokens.weight"] = torch.randn(V, H)

    for i in range(NL):
        p = f"model.layers.{i}"
        tensors[f"{p}.self_attn.q_proj.weight"]               = torch.randn(QH*HD, H)
        tensors[f"{p}.self_attn.q_proj.bias"]                 = torch.randn(QH*HD)
        tensors[f"{p}.self_attn.k_proj.weight"]               = torch.randn(KVH*HD, H)
        tensors[f"{p}.self_attn.k_proj.bias"]                 = torch.randn(KVH*HD)
        tensors[f"{p}.self_attn.v_proj.weight"]               = torch.randn(KVH*HD, H)
        tensors[f"{p}.self_attn.v_proj.bias"]                 = torch.randn(KVH*HD)
        tensors[f"{p}.self_attn.o_proj.weight"]               = torch.randn(H, QH*HD)
        tensors[f"{p}.mlp.gate_proj.weight"]                  = torch.randn(MLP, H)
        tensors[f"{p}.mlp.up_proj.weight"]                    = torch.randn(MLP, H)
        tensors[f"{p}.mlp.down_proj.weight"]                  = torch.randn(H, MLP)
        tensors[f"{p}.input_layernorm.weight"]                = torch.randn(H)
        tensors[f"{p}.post_attention_layernorm.weight"]       = torch.randn(H)

    tensors["model.norm.weight"] = torch.randn(H)
    # lm_head TIDAK disimpan karena tie_word_embeddings=True

    save_file(tensors, str(model_dir / "model.safetensors"))
    return str(model_dir)


@pytest.fixture(scope="session")
def nllb_dir(tmp_path_factory):
    """Fake NLLB/M2M-100 model directory."""
    from safetensors.torch import save_file
    import torch

    model_dir = tmp_path_factory.mktemp("nllb")

    config = {
        "model_type"              : "m2m_100",
        "architectures"           : ["M2M100ForConditionalGeneration"],
        "tokenizer_class"         : "NllbTokenizer",
        "d_model"                 : 64,
        "encoder_ffn_dim"         : 128,
        "decoder_ffn_dim"         : 128,
        "encoder_layers"          : 2,
        "decoder_layers"          : 2,
        "encoder_attention_heads" : 4,
        "decoder_attention_heads" : 4,
        "vocab_size"              : 512,
        "max_position_embeddings" : 130,
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    D, FFN, V, H, MAXPOS = 64, 128, 512, 4, 130

    tensors = {}
    # Shared embedding (master)
    tensors["model.shared.weight"]                    = torch.randn(V, D)
    tensors["model.encoder.embed_tokens.weight"]      = torch.randn(V, D)  # tied → di-skip
    tensors["model.decoder.embed_tokens.weight"]      = torch.randn(V, D)  # tied → di-skip
    tensors["model.encoder.embed_positions.weight"]   = torch.randn(MAXPOS+2, D)
    tensors["model.decoder.embed_positions.weight"]   = torch.randn(MAXPOS+2, D)

    for side in ("encoder", "decoder"):
        for i in range(2):
            p = f"model.{side}.layers.{i}"
            # Self-attention
            tensors[f"{p}.self_attn.q_proj.weight"]         = torch.randn(D, D)
            tensors[f"{p}.self_attn.q_proj.bias"]           = torch.randn(D)
            tensors[f"{p}.self_attn.k_proj.weight"]         = torch.randn(D, D)
            tensors[f"{p}.self_attn.k_proj.bias"]           = torch.randn(D)
            tensors[f"{p}.self_attn.v_proj.weight"]         = torch.randn(D, D)
            tensors[f"{p}.self_attn.v_proj.bias"]           = torch.randn(D)
            tensors[f"{p}.self_attn.out_proj.weight"]       = torch.randn(D, D)
            tensors[f"{p}.self_attn.out_proj.bias"]         = torch.randn(D)
            tensors[f"{p}.self_attn_layer_norm.weight"]     = torch.randn(D)
            tensors[f"{p}.self_attn_layer_norm.bias"]       = torch.randn(D)
            # FFN
            tensors[f"{p}.fc1.weight"]                      = torch.randn(FFN, D)
            tensors[f"{p}.fc1.bias"]                        = torch.randn(FFN)
            tensors[f"{p}.fc2.weight"]                      = torch.randn(D, FFN)
            tensors[f"{p}.fc2.bias"]                        = torch.randn(D)
            tensors[f"{p}.final_layer_norm.weight"]         = torch.randn(D)
            tensors[f"{p}.final_layer_norm.bias"]           = torch.randn(D)
            # Cross-attention (decoder only)
            if side == "decoder":
                tensors[f"{p}.encoder_attn.q_proj.weight"]       = torch.randn(D, D)
                tensors[f"{p}.encoder_attn.q_proj.bias"]         = torch.randn(D)
                tensors[f"{p}.encoder_attn.k_proj.weight"]       = torch.randn(D, D)
                tensors[f"{p}.encoder_attn.k_proj.bias"]         = torch.randn(D)
                tensors[f"{p}.encoder_attn.v_proj.weight"]       = torch.randn(D, D)
                tensors[f"{p}.encoder_attn.v_proj.bias"]         = torch.randn(D)
                tensors[f"{p}.encoder_attn.out_proj.weight"]     = torch.randn(D, D)
                tensors[f"{p}.encoder_attn.out_proj.bias"]       = torch.randn(D)
                tensors[f"{p}.encoder_attn_layer_norm.weight"]   = torch.randn(D)
                tensors[f"{p}.encoder_attn_layer_norm.bias"]     = torch.randn(D)

    tensors["model.encoder.layer_norm.weight"]  = torch.randn(D)
    tensors["model.encoder.layer_norm.bias"]    = torch.randn(D)
    tensors["model.decoder.layer_norm.weight"]  = torch.randn(D)
    tensors["model.decoder.layer_norm.bias"]    = torch.randn(D)
    # lm_head tidak disimpan → tied ke model.shared

    save_file(tensors, str(model_dir / "model.safetensors"))
    return str(model_dir)
