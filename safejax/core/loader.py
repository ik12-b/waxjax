import numpy as np
from pathlib import Path
from typing import Dict


def load_safetensors(model_path: str) -> Dict[str, np.ndarray]:
    """
    Load semua tensor dari direktori model.
    Handles:
      - Single file: model.safetensors
      - Sharded: model-00001-of-00003.safetensors
      - bfloat16 via PyTorch jika tersedia, fallback ke ml_dtypes
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("pip install safetensors")

    model_dir = Path(model_path)
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"Tidak ada .safetensors di: {model_path}")

    weights = {}
    for fp in files:
        weights.update(_load_single_file(fp))

    return weights


def _load_single_file(fp: Path) -> Dict[str, np.ndarray]:
    from safetensors import safe_open

    # Coba PyTorch dulu (handle bfloat16 paling reliable)
    _torch_available = _check_torch()

    result = {}
    if _torch_available:
        with safe_open(str(fp), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                result[key] = tensor.float().numpy()
    else:
        # Fallback: numpy + ml_dtypes untuk bfloat16
        result = _load_via_numpy(fp)

    return result


def _load_via_numpy(fp: Path) -> Dict[str, np.ndarray]:
    """
    Load tanpa PyTorch.
    bfloat16 di-handle via ml_dtypes jika tersedia,
    atau di-cast via uint16 reinterpret sebagai last resort.
    """
    from safetensors import safe_open

    try:
        import ml_dtypes  # noqa: F401
        _has_ml_dtypes = True
    except ImportError:
        _has_ml_dtypes = False

    result = {}
    with safe_open(str(fp), framework="numpy") as f:
        for key in f.keys():
            try:
                arr = f.get_tensor(key)
                result[key] = _to_float32(arr, _has_ml_dtypes)
            except TypeError:
                # bfloat16 fallback: load sebagai raw bytes lalu reinterpret
                arr = _load_bf16_fallback(f, key)
                result[key] = arr

    return result


def _to_float32(arr: np.ndarray, has_ml_dtypes: bool) -> np.ndarray:
    if str(arr.dtype) == "bfloat16":
        if has_ml_dtypes:
            import ml_dtypes
            return arr.astype(ml_dtypes.bfloat16).astype(np.float32)
        else:
            # Reinterpret uint16 bits sebagai bfloat16 lalu cast
            return arr.view(np.uint16).astype(np.float32)
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    return arr.astype(np.float32)


def _load_bf16_fallback(f, key: str) -> np.ndarray:
    """Last resort: baca raw bytes, reinterpret sebagai float32."""
    import struct
    raw = f.get_tensor(key)
    # bf16 → float32 via bit manipulation
    u16 = np.frombuffer(raw.tobytes(), dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(raw.shape)


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False
