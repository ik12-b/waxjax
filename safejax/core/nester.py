from typing import Dict, Tuple
import numpy as np



def to_nested_dict(flat: Dict[Tuple, np.ndarray]) -> dict:
    """
    {('a','b','c'): arr} → {'a': {'b': {'c': arr}}}
    Semua array dikonversi ke jnp.array.
    """
    try:
        import jax.numpy as jnp
    except Exception:  # pragma: no cover - only triggers when jax absent
        raise ImportError(
            "jax is required to convert arrays to jnp.ndarray. "
            "Install jax to use to_nested_dict()"
        )

    root = {}
    for keys, arr in flat.items():
        d = root
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = jnp.array(arr)
    return root


def to_frozen_dict(flat: Dict[Tuple, np.ndarray]):
    """Sama seperti to_nested_dict tapi return FrozenDict (butuh Flax)."""
    try:
        from flax.core import freeze
    except ImportError:
        raise ImportError(
            "format='frozen' membutuhkan Flax. "
            "Install dengan: pip install flax"
        )
    nested = to_nested_dict(flat)
    return freeze(nested)
