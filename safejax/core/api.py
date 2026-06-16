import json
from pathlib import Path
from typing import Literal, List, Optional

from .loader   import load_safetensors
from .mapper   import apply_rules
from .nester   import to_nested_dict, to_frozen_dict
from .rules    import MappingRule
from ..architectures._registry import get_architecture, list_architectures


def load(
    model_path : str,
    format     : Literal["nested", "frozen"] = "nested",
    rules      : Optional[List[MappingRule]] = None,
) -> dict:
    """
    Load HuggingFace SafeTensors model ke JAX params.

    Args:
        model_path : direktori berisi config.json + *.safetensors
        format     : "nested" → plain dict (default)
                     "frozen" → flax.core.FrozenDict
        rules      : custom MappingRule list. Jika None,
                     otomatis detect dari config.json.

    Returns:
        JAX nested params dict

    Example:
        params = safejax.load("./Qwen2.5-0.5B")
        params = safejax.load("./model", format="frozen")
    """
    cfg          = _read_config(model_path)
    model_type   = cfg.get("model_type", "unknown")
    pt_weights   = load_safetensors(model_path)
    tied_weights = {}

    if rules is None:
        arch = get_architecture(model_type)
        if arch is None:
            raise ValueError(
                f"model_type '{model_type}' belum didukung.\n"
                f"Arsitektur yang tersedia: {list_architectures()}\n"
                f"Gunakan parameter 'rules' untuk custom mapping."
            )
        rules        = arch.rules
        tied_weights = arch.tied_weights

    flat   = apply_rules(pt_weights, rules, tied_weights)
    nested = to_nested_dict(flat)

    if format == "frozen":
        return to_frozen_dict(flat)
    return nested


def inspect(model_path: str):
    """Tampilkan info model dan mapping yang akan diterapkan."""
    cfg        = _read_config(model_path)
    model_type = cfg.get("model_type", "unknown")
    arch       = get_architecture(model_type)

    files = sorted(Path(model_path).glob("*.safetensors"))
    pt_weights = load_safetensors(model_path)

    print(f"\n  model_type   : {model_type}")
    print(f"  architecture : {cfg.get('architectures', ['?'])[0]}")
    print(f"  num_layers   : {cfg.get('num_hidden_layers', '?')}")
    print(f"  tied_embed   : {cfg.get('tie_word_embeddings', '?')}")
    print(f"  shards       : {len(files)} file(s)")
    print(f"  tensors      : {len(pt_weights)}")

    if arch:
        print(f"  mapping      : ✓ built-in ({arch.name})")
        print(f"\n  Sample mapping (5 pertama):")
        shown = 0
        for pt_key in pt_weights:
            for rule in sorted(arch.rules, key=lambda r: r.priority, reverse=True):
                if rule.matches(pt_key):
                    new_key = rule.apply_rename(pt_key)
                    arr     = pt_weights[pt_key]
                    print(f"    {pt_key}")
                    print(f"    → {new_key}  {arr.shape}")
                    shown += 1
                    break
            if shown >= 5:
                break
    else:
        print(f"  mapping      : ✗ belum didukung")
        print(f"  Gunakan load(rules=[...]) untuk custom mapping.")


def verify(jax_params, source_dir: str, sample_n: int = 10, atol: float = 1e-5):
    """Verifikasi JAX params vs PyTorch source."""
    from .verifier import verify as _verify
    return _verify(jax_params, source_dir, sample_n, atol)


def save(params: dict, output_path: str):
    """Simpan JAX params ke safetensors format."""
    try:
        from safetensors.flax import save_file
        from flax.traverse_util import flatten_dict
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("pip install safetensors flax")

    flat = flatten_dict(params)
    flat_str = {".".join(k): v for k, v in flat.items()}
    save_file(flat_str, output_path)
    print(f"  ✓ Disimpan ke: {output_path}")


def _read_config(model_path: str) -> dict:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json tidak ditemukan di {model_path}")
    with open(cfg_path) as f:
        return json.load(f)