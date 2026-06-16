from dataclasses import dataclass, field
from typing import Callable, Optional, Union
import re

# ── Built-in transforms ───────────────────────────────────────

def transpose_2d(arr):
    """Transpose 2D array: PyTorch (out,in) → JAX (in,out)"""
    import numpy as np
    arr = np.array(arr)
    if arr.ndim == 2:
        return arr.T
    return arr

def no_op(arr):
    return arr

TRANSFORMS = {
    "transpose_2d" : transpose_2d,
    "no_op"        : no_op,
    None           : no_op,
}

# ── MappingRule ───────────────────────────────────────────────

@dataclass
class MappingRule:
    """
    Satu aturan konversi untuk sekelompok tensor.

    Args:
        match     : glob pattern, contoh "*.weight", "*.norm.weight"
                    Lebih spesifik = prioritas lebih tinggi.
        rename    : fungsi (str → str) untuk ubah nama key.
                    Contoh: lambda k: k.replace(".weight", ".kernel")
        transform : "transpose_2d" | "no_op" | None | Callable
                    Operasi yang diterapkan ke array setelah rename.
        priority  : rule dengan priority lebih tinggi dipakai duluan
                    ketika beberapa rule match. Default auto dari
                    spesifisitas pattern (lebih panjang = lebih tinggi).
    """
    match     : str
    rename    : Callable[[str], str]
    transform : Union[str, Callable, None] = None
    priority  : Optional[int]              = None

    def __post_init__(self):
        # Resolve transform ke callable
        if isinstance(self.transform, str) or self.transform is None:
            if self.transform not in TRANSFORMS:
                raise ValueError(
                    f"Transform '{self.transform}' tidak dikenal. "
                    f"Pilih dari: {list(TRANSFORMS.keys())} atau berikan callable."
                )
            self._transform_fn = TRANSFORMS[self.transform]
        else:
            self._transform_fn = self.transform

        # Auto-priority dari panjang pattern (lebih spesifik = lebih tinggi)
        if self.priority is None:
            self.priority = len(self.match)

    def matches(self, key: str) -> bool:
        """Cek apakah key cocok dengan pattern glob ini."""
        # Konversi glob ke regex: * → [^.]*, ** → .*
        pattern = re.escape(self.match)
        pattern = pattern.replace(r"\*\*", ".*")
        pattern = pattern.replace(r"\*", r"[^.]*")
        return bool(re.fullmatch(pattern, key))

    def apply_rename(self, key: str) -> str:
        return self.rename(key)

    def apply_transform(self, arr):
        return self._transform_fn(arr)
