# tests/test_rules.py
"""Unit test untuk MappingRule — matching dan transform."""

import pytest
import numpy as np
from waxjax.core.rules import MappingRule, transpose_2d, no_op


class TestMappingRuleMatch:

    def test_exact_match(self):
        rule = MappingRule("lm_head.weight", rename=lambda k: k, transform="no_op")
        assert rule.matches("lm_head.weight")
        assert not rule.matches("lm_head.bias")

    def test_single_wildcard(self):
        rule = MappingRule(
            "model.layers.*.self_attn.q_proj.weight",
            rename=lambda k: k, transform="no_op"
        )
        assert rule.matches("model.layers.0.self_attn.q_proj.weight")
        assert rule.matches("model.layers.11.self_attn.q_proj.weight")
        # * tidak boleh match dot
        assert not rule.matches("model.layers.0.1.self_attn.q_proj.weight")

    def test_double_wildcard(self):
        rule = MappingRule(
            "model.**.weight",
            rename=lambda k: k, transform="no_op"
        )
        assert rule.matches("model.layers.0.self_attn.q_proj.weight")
        assert rule.matches("model.norm.weight")

    def test_priority_auto(self):
        specific = MappingRule(
            "model.layers.*.self_attn_layer_norm.weight",
            rename=lambda k: k, transform="no_op"
        )
        generic = MappingRule(
            "model.layers.*.*.weight",
            rename=lambda k: k, transform="no_op"
        )
        assert specific.priority > generic.priority

    def test_invalid_transform_raises(self):
        with pytest.raises(ValueError, match="tidak dikenal"):
            MappingRule("*.weight", rename=lambda k: k, transform="unknown_transform")


class TestMappingRuleTransform:

    def test_transpose_2d(self):
        arr = np.ones((4, 8))
        rule = MappingRule("*.weight", rename=lambda k: k, transform="transpose_2d")
        result = rule.apply_transform(arr)
        assert result.shape == (8, 4)

    def test_no_transpose_1d(self):
        """1D array (bias) tidak boleh di-transpose meski pakai transpose_2d."""
        arr = np.ones(8)
        result = transpose_2d(arr)
        assert result.shape == (8,)

    def test_custom_callable_transform(self):
        def double(arr):
            return arr * 2
        rule = MappingRule("*.weight", rename=lambda k: k, transform=double)
        arr = np.array([1.0, 2.0])
        assert np.allclose(rule.apply_transform(arr), [2.0, 4.0])

    def test_rename_applied(self):
        rule = MappingRule(
            "*.weight",
            rename=lambda k: k.replace(".weight", ".kernel"),
            transform="no_op"
        )
        assert rule.apply_rename("model.fc.weight") == "model.fc.kernel"
