from ..core.rules import MappingRule
from ._registry import ArchitectureConfig, register_architecture

# ── Rename helpers ────────────────────────────────────────────

def _linear(k: str) -> str:
    return k.replace(".weight", ".kernel")

def _norm(k: str) -> str:
    return k.replace(".weight", ".scale")

def _embed(k: str) -> str:
    return k.replace(".weight", ".embedding")

# ── Rules Qwen2/2.5 ──────────────────────────────────────────
# Urutan tidak penting karena priority dihitung dari len(pattern)

QWEN2_RULES = [
    # Embedding: TIDAK transpose, rename weight → embedding
    MappingRule(
        match     = "model.embed_tokens.weight",
        rename    = _embed,
        transform = "no_op",
    ),
    # RMSNorm layers: TIDAK transpose, rename weight → scale
    MappingRule(
        match     = "model.layers.*.input_layernorm.weight",
        rename    = _norm,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.layers.*.post_attention_layernorm.weight",
        rename    = _norm,
        transform = "no_op",
    ),
    MappingRule(
        match     = "model.norm.weight",
        rename    = _norm,
        transform = "no_op",
    ),
    # Attention projections: TRANSPOSE, rename weight → kernel
    MappingRule(
        match     = "model.layers.*.self_attn.q_proj.weight",
        rename    = _linear,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.layers.*.self_attn.k_proj.weight",
        rename    = _linear,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.layers.*.self_attn.v_proj.weight",
        rename    = _linear,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.layers.*.self_attn.o_proj.weight",
        rename    = _linear,
        transform = "transpose_2d",
    ),
    # QKV bias: TIDAK transpose
    MappingRule(
        match     = "model.layers.*.self_attn.*.bias",
        rename    = lambda k: k,   # bias tetap bias
        transform = "no_op",
    ),
    # MLP: TRANSPOSE
    MappingRule(
        match     = "model.layers.*.mlp.gate_proj.weight",
        rename    = _linear,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.layers.*.mlp.up_proj.weight",
        rename    = _linear,
        transform = "transpose_2d",
    ),
    MappingRule(
        match     = "model.layers.*.mlp.down_proj.weight",
        rename    = _linear,
        transform = "transpose_2d",
    ),
    # LM head: TRANSPOSE
    MappingRule(
        match     = "lm_head.weight",
        rename    = _linear,
        transform = "transpose_2d",
    ),
]

register_architecture(ArchitectureConfig(
    name         = "qwen2",
    model_types  = ["qwen2"],          # handle Qwen2 dan Qwen2.5 (sama model_type)
    rules        = QWEN2_RULES,
    tied_weights = {
        # Jika lm_head.weight tidak ada di file (tied),
        # buat dari model.embed_tokens.embedding (sudah di-rename)
        "model.embed_tokens.weight": "lm_head.kernel",
    },
))