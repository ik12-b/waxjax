"""Compatibility shim: expose the public API under the expected
package name `waxjax` while the real implementation lives in
`safejax` (to preserve the original repository layout).
"""
# Re-export the public API without importing `safejax` package object
# to avoid circular-import issues during package initialization.
from .core.api import load, save, inspect, verify
from .core.rules import MappingRule
from .architectures._registry import register_architecture, ArchitectureConfig

__version__ = "0.1.0"

__all__ = [
    "load", "save", "inspect", "verify",
    "MappingRule", "register_architecture", "ArchitectureConfig",
]
