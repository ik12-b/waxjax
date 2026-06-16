from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from ..core.rules import MappingRule


@dataclass
class ArchitectureConfig:
    """
    Konfigurasi konversi untuk satu family arsitektur.

    Args:
        name          : nama unik, contoh "qwen2"
        model_types   : list nilai "model_type" dari config.json
                        yang di-handle config ini.
                        Contoh: ["qwen2"] untuk Qwen2 dan Qwen2.5
        rules         : list MappingRule
        tied_weights  : dict {src_pt_key: target_renamed_key}
                        untuk handle shared weights
        detect_fn     : fungsi opsional (config_dict → bool)
                        untuk deteksi lebih spesifik selain model_type
    """
    name         : str
    model_types  : List[str]
    rules        : List[MappingRule]
    tied_weights : Dict[str, str]          = field(default_factory=dict)
    detect_fn    : Optional[Callable]      = None


# Global registry
_REGISTRY: Dict[str, ArchitectureConfig] = {}


def register_architecture(config: ArchitectureConfig):
    """Daftarkan ArchitectureConfig ke global registry."""
    for model_type in config.model_types:
        if model_type in _REGISTRY:
            raise ValueError(
                f"model_type '{model_type}' sudah terdaftar "
                f"oleh '{_REGISTRY[model_type].name}'. "
                f"Gunakan force=True untuk override."
            )
        _REGISTRY[model_type] = config


def get_architecture(model_type: str) -> Optional[ArchitectureConfig]:
    return _REGISTRY.get(model_type)


def list_architectures() -> List[str]:
    return sorted(set(cfg.name for cfg in _REGISTRY.values()))


# Auto-register semua built-in saat module di-import
def _load_builtins():
    # Import only existing built-in architecture modules. Other names
    # previously referenced here (llama, mistral, bert) do not exist
    # in this package and caused import errors during test/import time.
    from . import qwen2, nllb  # noqa: F401

_load_builtins()