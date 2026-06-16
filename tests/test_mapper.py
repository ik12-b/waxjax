# tests/test_mapper.py
"""Test mapper: apply rules, skip, tied weights."""

import pytest
import numpy as np
from waxjax.core.mapper import apply_rules
from waxjax.core.rules import MappingRule


def _linear_rules():
    return [
        MappingRule("**.weight", rename=lambda k: k.replace(".weight", ".kernel"),
                    transform="transpose_2d"),
        MappingRule("**.bias",   rename=lambda k: k, transform="no_op"),
    ]


class TestMapper:

    def test_basic_rename_and_transpose(self):
        weights = {"fc.weight": np.ones((4, 8))}
        result  = apply_rules(weights, _linear_rules())
        assert ("fc", "kernel") in result
        assert result[("fc", "kernel")].shape == (8, 4)

    def test_bias_not_transposed(self):
        weights = {"fc.bias": np.ones(8)}
        result  = apply_rules(weights, _linear_rules())
        assert result[("fc", "bias")].shape == (8,)

    def test_skip_sentinel(self):
        """Key dengan rename → '__skip__' tidak masuk ke output."""
        rules = [
            MappingRule(
                "tied.weight",
                rename=lambda k: "__skip__",
                transform="no_op",
            )
        ]
        weights = {"tied.weight": np.ones((4, 4))}
        result  = apply_rules(weights, rules)
        assert ("tied", "weight") not in result
        assert len(result) == 0

    def test_priority_order(self):
        """Rule lebih spesifik menang atas rule lebih umum."""
        specific = MappingRule(
            "model.norm.weight",
            rename=lambda k: k.replace(".weight", ".scale"),
            transform="no_op",
        )
        generic = MappingRule(
            "**.weight",
            rename=lambda k: k.replace(".weight", ".kernel"),
            transform="transpose_2d",
        )
        weights = {"model.norm.weight": np.ones(8)}
        result  = apply_rules(weights, [generic, specific])
        # Harus pakai rule spesifik → .scale, bukan .kernel
        assert ("model", "norm", "scale") in result
        assert ("model", "norm", "kernel") not in result

    def test_tied_weights_auto_created(self):
        """Tied weight dibuat otomatis jika source ada tapi target tidak."""
        rules = [
            MappingRule(
                "model.shared.weight",
                rename=lambda k: k.replace(".weight", ".embedding"),
                transform="no_op",
            )
        ]
        weights = {"model.shared.weight": np.ones((512, 64))}
        result  = apply_rules(
            weights, rules,
            tied_weights={"model.shared.weight": "lm_head.kernel"}
        )
        # embedding harus ada
        assert ("model", "shared", "embedding") in result
        # lm_head.kernel harus dibuat otomatis dengan transpose
        assert ("lm_head", "kernel") in result
        assert result[("lm_head", "kernel")].shape == (64, 512)

    def test_unmatched_keys_warn(self):
        """Key tanpa rule matching menghasilkan warning, tidak error."""
        weights = {"unknown.mystery": np.ones((4, 4))}
        with pytest.warns(UserWarning, match="tidak match"):
            result = apply_rules(weights, _linear_rules())
        assert len(result) == 0
