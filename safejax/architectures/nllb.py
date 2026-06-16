"""
NLLB-200 (facebook/nllb-200-distilled-600M, 1.3B, 3.3B)
model_type: "m2m_100"  ← dari config.json

Struktur key PyTorch yang lengkap:
  SHARED / EMBEDDING
    model.shared.weight                              → model.shared.embedding
    model.encoder.embed_tokens.weight                → (tied, skip)
    model.decoder.embed_tokens.weight                → (tied, skip)
    model.encoder.embed_positions.weight             → model.encoder.embed_positions.embedding
    model.decoder.embed_positions.weight             → model.decoder.embed_positions.embedding

  ENCODER per layer (12 layer di 600M):
    model.encoder.layers.N.self_attn.q_proj.weight  → kernel (transpose)
    model.encoder.layers.N.self_attn.k_proj.weight  → kernel (transpose)
    model.encoder.layers.N.self_attn.v_proj.weight  → kernel (transpose)
    model.encoder.layers.N.self_attn.out_proj.weight → kernel (transpose)
    model.encoder.layers.N.self_attn.q_proj.bias    → bias
    model.encoder.layers.N.self_attn.k_proj.bias    → bias
    model.encoder.layers.N.self_attn.v_proj.bias    → bias
    model.encoder.layers.N.self_attn.out_proj.bias  → bias
    model.encoder.layers.N.self_attn_layer_norm.weight → scale
    model.encoder.layers.N.self_attn_layer_norm.bias   → bias
    model.encoder.layers.N.fc1.weight               → kernel (transpose)
    model.encoder.layers.N.fc1.bias                 → bias
    model.encoder.layers.N.fc2.weight               → kernel (transpose)
    model.encoder.layers.N.fc2.bias                 → bias
    model.encoder.layers.N.final_layer_norm.weight  → scale
    model.encoder.layers.N.final_layer_norm.bias    → bias
    model.encoder.layer_norm.weight                 → scale  (post-encoder norm)
    model.encoder.layer_norm.bias                   → bias

  DECODER per layer (sama + cross-attention):
    model.decoder.layers.N.self_attn.*              → sama seperti encoder
    model.decoder.layers.N.self_attn_layer_norm.*   → scale/bias
    model.decoder.layers.N.encoder_attn.q_proj.*    → kernel/bias
    model.decoder.layers.N.encoder_attn.k_proj.*    → kernel/bias
    model.decoder.layers.N.encoder_attn.v_proj.*    → kernel/bias
    model.decoder.layers.N.encoder_attn.out_proj.*  → kernel/bias
    model.decoder.layers.N.encoder_attn_layer_norm.* → scale/bias
    model.decoder.layers.N.fc1.*                    → kernel/bias
    model.decoder.layers.N.fc2.*                    → kernel/bias
    model.decoder.layers.N.final_layer_norm.*       → scale/bias
    model.decoder.layer_norm.*                      → scale/bias (post-decoder norm)

  LM HEAD
    lm_head.weight                                  → lm_head.kernel (transpose)
                                                      (tied dengan model.shared.weight)

  QUIRKS khusus NLLB vs model lain:
    1. LayerNorm (bukan RMSNorm) → punya WEIGHT dan BIAS
       weight → scale, bias → bias
    2. MLP pakai fc1/fc2 (bukan gate_proj/up_proj/down_proj)
       fc1 dan fc2 KEDUANYA perlu transpose
    3. self_attn_layer_norm (underscore, bukan dot) = nama yang berbeda
    4. model.shared di-tied ke encoder_embed, decoder_embed, DAN lm_head
    5. embed_positions adalah learned positional embedding
       shape (max_pos, d_model) = (1026, 1024) → TIDAK perlu transpose
    6. out_proj (bukan o_proj seperti Qwen2/Llama)
"""

from ..core.rules import MappingRule
from ._registry import ArchitectureConfig, register_architecture


# ── Rename helpers ────────────────────────────────────────────

def _to_kernel(k: str) -> str:
    """*.weight → *.kernel"""
    return k.replace(".weight", ".kernel")

def _to_scale(k: str) -> str:
    """LayerNorm weight → scale"""
    return k.replace(".weight", ".scale")

def _to_embedding(k: str) -> str:
    """Embedding weight → embedding"""
    return k.replace(".weight", ".embedding")

def _identity(k: str) -> str:
    return k


# ── Rules ─────────────────────────────────────────────────────

NLLB_RULES = [

    # ── SHARED EMBEDDING ──────────────────────────────────────
    # model.shared adalah master embedding untuk seluruh model
    # encoder/decoder embed_tokens di-tied ke sini, jadi SKIP keduanya
    MappingRule(
        match     = "model.shared.weight",
        rename    = _to_embedding,
        transform = "no_op",              # (V, d_model) tidak perlu transpose
    ),
    MappingRule(
        match     = "model.encoder.embed_tokens.weight",
        rename    = lambda k: "__skip__",  # tied ke model.shared
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.decoder.embed_tokens.weight",
        rename    = lambda k: "__skip__",  # tied ke model.shared
        transform = "no_op",
    ),

    # ── POSITIONAL EMBEDDINGS ─────────────────────────────────
    # Learned positional embedding, shape (max_pos+2, d_model)
    # TIDAK perlu transpose (bukan linear projection)
    MappingRule(
        match     = "model.encoder.embed_positions.weight",
        rename    = _to_embedding,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.decoder.embed_positions.weight",
        rename    = _to_embedding,
        transform = "no_op",
    ),

    # ── ENCODER: SELF-ATTENTION PROJECTIONS ───────────────────
    # q/k/v/out_proj weight → kernel + TRANSPOSE
    MappingRule(
        match     = "model.encoder.layers.*.self_attn.q_proj.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.encoder.layers.*.self_attn.k_proj.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.encoder.layers.*.self_attn.v_proj.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.encoder.layers.*.self_attn.out_proj.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    # bias: tidak perlu transpose, tidak perlu rename
    MappingRule(
        match     = "model.encoder.layers.*.self_attn.*.bias",
        rename    = _identity,
        transform = "no_op",
    ),

    # ── ENCODER: LAYER NORM ───────────────────────────────────
    # self_attn_layer_norm dan final_layer_norm
    # LayerNorm punya weight (→ scale) DAN bias (→ bias)
    MappingRule(
        match     = "model.encoder.layers.*.self_attn_layer_norm.weight",
        rename    = _to_scale,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.encoder.layers.*.self_attn_layer_norm.bias",
        rename    = _identity,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.encoder.layers.*.final_layer_norm.weight",
        rename    = _to_scale,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.encoder.layers.*.final_layer_norm.bias",
        rename    = _identity,
        transform = "no_op",
    ),

    # ── ENCODER: FFN (fc1, fc2) ───────────────────────────────
    # NLLB pakai fc1/fc2 dengan activation relu, bukan SwiGLU
    # Keduanya perlu transpose
    MappingRule(
        match     = "model.encoder.layers.*.fc1.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.encoder.layers.*.fc1.bias",
        rename    = _identity,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.encoder.layers.*.fc2.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.encoder.layers.*.fc2.bias",
        rename    = _identity,
        transform = "no_op",
    ),

    # ── ENCODER: POST-ENCODER LAYER NORM ─────────────────────
    MappingRule(
        match     = "model.encoder.layer_norm.weight",
        rename    = _to_scale,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.encoder.layer_norm.bias",
        rename    = _identity,
        transform = "no_op",
    ),

    # ── DECODER: SELF-ATTENTION ───────────────────────────────
    MappingRule(
        match     = "model.decoder.layers.*.self_attn.q_proj.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.self_attn.k_proj.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.self_attn.v_proj.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.self_attn.out_proj.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.self_attn.*.bias",
        rename    = _identity,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.self_attn_layer_norm.weight",
        rename    = _to_scale,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.self_attn_layer_norm.bias",
        rename    = _identity,
        transform = "no_op",
    ),

    # ── DECODER: CROSS-ATTENTION (encoder_attn) ───────────────
    # Nama berbeda dari self_attn: encoder_attn + encoder_attn_layer_norm
    MappingRule(
        match     = "model.decoder.layers.*.encoder_attn.q_proj.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.encoder_attn.k_proj.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.encoder_attn.v_proj.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.encoder_attn.out_proj.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.encoder_attn.*.bias",
        rename    = _identity,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.encoder_attn_layer_norm.weight",
        rename    = _to_scale,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.encoder_attn_layer_norm.bias",
        rename    = _identity,
        transform = "no_op",
    ),

    # ── DECODER: FFN ─────────────────────────────────────────
    MappingRule(
        match     = "model.decoder.layers.*.fc1.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.fc1.bias",
        rename    = _identity,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.fc2.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.fc2.bias",
        rename    = _identity,
        transform = "no_op",
    ),

    # ── DECODER: FINAL LAYER NORM ─────────────────────────────
    MappingRule(
        match     = "model.decoder.layers.*.final_layer_norm.weight",
        rename    = _to_scale,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.decoder.layers.*.final_layer_norm.bias",
        rename    = _identity,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.decoder.layer_norm.weight",
        rename    = _to_scale,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.decoder.layer_norm.bias",
        rename    = _identity,
        transform = "no_op",
    ),

    # ── LM HEAD ───────────────────────────────────────────────
    # Tied dengan model.shared — kalau ada di file, konversi normal
    # Kalau tidak ada (tied), di-handle oleh tied_weights dict
    MappingRule(
        match     = "lm_head.weight",
        rename    = _to_kernel,
        transform = "transpose_2d",
    ),
]


# ── Register ──────────────────────────────────────────────────

register_architecture(ArchitectureConfig(
    name        = "nllb",
    model_types = ["m2m_100"],      # NLLB dan M2M-100 pakai model_type yang sama
    rules       = NLLB_RULES,

    # Tied weights NLLB:
    # model.shared adalah master embedding
    # lm_head.kernel = model.shared.embedding.T
    tied_weights = {
        "model.shared.weight": "lm_head.kernel",
    },

    # Deteksi lebih spesifik: m2m_100 bisa M2M-100 ATAU NLLB
    # Bedakan via tokenizer_class di config.json
    detect_fn = lambda cfg: cfg.get("tokenizer_class") in (
        "NllbTokenizer", "NllbTokenizerFast", None
    ),
))