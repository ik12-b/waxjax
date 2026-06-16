import numpy as np
from typing import Dict, Tuple
from pathlib import Path


def verify(
    jax_params : dict,
    source_dir : str,
    sample_n   : int = 10,
    atol       : float = 1e-5,
) -> dict:
    """
    Bandingkan JAX params vs PyTorch source.
    Return report dict dengan hasil verifikasi.
    """
    from .loader import load_safetensors
    from flax.traverse_util import flatten_dict

    pt_weights = load_safetensors(source_dir)
    flat_jax   = flatten_dict(jax_params)   # {tuple: jnp.array}

    keys_to_check = list(pt_weights.keys())[:sample_n]
    results = []

    for pt_key in keys_to_check:
        pt_arr = pt_weights[pt_key].astype(np.float32)
        results.append(_check_one(pt_key, pt_arr, flat_jax, atol))

    passed   = sum(1 for r in results if r["ok"])
    max_diff = max((r["max_diff"] for r in results if r["max_diff"] is not None), default=0)

    report = {
        "passed"   : passed,
        "total"    : len(results),
        "max_diff" : max_diff,
        "details"  : results,
    }

    _print_report(report)
    return report


def _check_one(pt_key, pt_arr, flat_jax, atol) -> dict:
    import numpy as np
    from .mapper import _key_to_tuple

    # Coba cari di flat_jax — key mungkin sudah di-rename
    # Heuristic: cari key yang isinya sama shape (transposed atau tidak)
    found_key  = None
    found_arr  = None

    # Prefer exact-name matches after common renames before falling back to
    # shape-only heuristic. This avoids picking the wrong tensor when many
    # tensors share the same shape.
    candidates = {}
    for jax_key, jax_arr in flat_jax.items():
        candidates[".".join(jax_key)] = np.array(jax_arr)

    # Common rename heuristics
    name_variants = [
        pt_key,
        pt_key.replace('.weight', '.kernel'),
        pt_key.replace('.weight', '.embedding'),
        pt_key.replace('.weight', '.scale'),
        pt_key.replace('.o_proj', '.out_proj'),
    ]

    for name in name_variants:
        if name in candidates:
            found_key = _key_to_tuple(name)
            found_arr = candidates[name]
            # If shapes differ, try transposing the pt array to compare.
            if found_arr.shape == pt_arr.shape:
                pass
            elif found_arr.shape == pt_arr.T.shape:
                pt_arr = pt_arr.T
            break

    # Fallback: shape-based search (original heuristic)
    if found_arr is None:
        for jax_key, jax_arr in flat_jax.items():
            jax_np = np.array(jax_arr)
            if jax_np.shape == pt_arr.shape:
                found_key = jax_key
                found_arr = jax_np
                break
            if jax_np.shape == pt_arr.T.shape:
                found_key  = jax_key
                found_arr  = jax_np
                pt_arr     = pt_arr.T   # bandingkan setelah transpose
                break

    if found_arr is None:
        return {"pt_key": pt_key, "ok": False,
                "max_diff": None, "reason": "tidak ditemukan di JAX params"}

    max_diff = float(np.max(np.abs(found_arr - pt_arr)))
    ok       = max_diff <= atol

    return {
        "pt_key"  : pt_key,
        "jax_key" : found_key,
        "ok"      : ok,
        "max_diff": max_diff,
        "reason"  : None if ok else f"max_diff={max_diff:.2e} > atol={atol:.2e}",
    }


def _print_report(report: dict):
    p, t = report["passed"], report["total"]
    print(f"\n  Verifikasi: {p}/{t} tensor match")
    print(f"  Max diff  : {report['max_diff']:.2e}")
    for r in report["details"]:
        mark = "✓" if r["ok"] else "✗"
        info = r.get("reason") or f"max_diff={r['max_diff']:.2e}"
        print(f"  {mark}  {r['pt_key']:<50}  {info}")
