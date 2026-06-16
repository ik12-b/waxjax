import numpy as np
from typing import Dict, List, Tuple
from .rules import MappingRule


def apply_rules(
    flat_weights : Dict[str, np.ndarray],
    rules        : List[MappingRule],
    tied_weights : Dict[str, str] = None,
) -> Dict[Tuple, np.ndarray]:
    """
    Terapkan rules ke flat dict PT weights.

    Args:
        flat_weights : {pt_key: np.ndarray}
        rules        : list MappingRule, diurutkan by priority
        tied_weights : {sumber_pt_key: target_flax_tuple_str}
                       untuk handle tied embedding dll

    Returns:
        {flax_tuple_key: np.ndarray}
    """
    # Sort rules by priority descending (lebih spesifik duluan)
    sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)

    result    = {}
    result    = {}
    src_map   = {}  # map original pt_key -> flax_tuple produced
    no_match  = []

    for pt_key, arr in flat_weights.items():
        matched = False
        for rule in sorted_rules:
            if rule.matches(pt_key):
                new_key = rule.apply_rename(pt_key)
                new_arr = rule.apply_transform(arr)
                # Sentinel: if rename returns '__skip__', do not include
                # this tensor in the output.
                if new_key == "__skip__":
                    matched = True
                    break

                flax_tuple = _key_to_tuple(new_key)
                result[flax_tuple] = new_arr
                src_map[pt_key] = flax_tuple
                matched = True
                break

        if not matched:
            no_match.append(pt_key)

    if no_match:
        import warnings
        warnings.warn(
            f"{len(no_match)} tensor tidak match rule apapun dan dilewati:\n"
            + "\n".join(f"  - {k}" for k in no_match[:5])
            + (f"\n  ... dan {len(no_match)-5} lainnya" if len(no_match) > 5 else "")
        )

    # Handle tied weights
    if tied_weights:
        for src_pt_key, target_key_str in tied_weights.items():
            target_tuple = _key_to_tuple(target_key_str)
            # src_pt_key refers to the original pytorch key; find the
            # corresponding flax tuple produced earlier (if any).
            src_tuple = src_map.get(src_pt_key)

            if target_tuple not in result and src_tuple in result:
                # Tied biasanya butuh transpose
                result[target_tuple] = result[src_tuple].T
            elif target_tuple not in result:
                import warnings
                warnings.warn(f"Tied weight source tidak ditemukan: {src_pt_key}")

    return result


def _key_to_tuple(dot_key: str) -> tuple:
    """'model.layers.0.self_attn.q_proj.kernel' → ('model','layers','0',...)"""
    return tuple(dot_key.split("."))
