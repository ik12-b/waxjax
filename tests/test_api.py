# tests/test_api.py
"""Test public API: inspect, verify, unsupported model."""

import pytest
import json
import waxjax


class TestInspect:

    def test_inspect_runs_without_error(self, qwen2_dir, capsys):
        waxjax.inspect(qwen2_dir)
        out = capsys.readouterr().out
        assert "qwen2" in out
        assert "✓" in out

    def test_inspect_shows_unsupported(self, tmp_path, capsys):
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "unknown_model_xyz"})
        )
        # Buat dummy safetensors
        import torch
        from safetensors.torch import save_file
        save_file({"w": torch.randn(2,2)}, str(tmp_path / "model.safetensors"))

        waxjax.inspect(str(tmp_path))
        out = capsys.readouterr().out
        assert "✗" in out or "belum didukung" in out


class TestUnsupportedModel:

    def test_load_unknown_raises_valueerror(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "unknown_xyz"})
        )
        import torch
        from safetensors.torch import save_file
        save_file({"w": torch.randn(2,2)}, str(tmp_path / "model.safetensors"))

        with pytest.raises(ValueError, match="belum didukung"):
            waxjax.load(str(tmp_path))

    def test_load_with_custom_rules_bypasses_registry(self, tmp_path):
        """Custom rules bisa dipakai untuk model yang belum terdaftar."""
        from waxjax.core.rules import MappingRule
        import torch
        from safetensors.torch import save_file

        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "custom_unknown"})
        )
        save_file({"fc.weight": torch.randn(4, 8)},
                  str(tmp_path / "model.safetensors"))

        rules = [
            MappingRule("fc.weight",
                        rename=lambda k: k.replace(".weight", ".kernel"),
                        transform="transpose_2d")
        ]
        params = waxjax.load(str(tmp_path), rules=rules)
        assert "fc" in params
        assert "kernel" in params["fc"]


class TestVerify:

    def test_verify_passes_for_correct_conversion(self, qwen2_dir):
        params = waxjax.load(qwen2_dir)
        report = waxjax.verify(params, qwen2_dir, sample_n=5)
        assert report["passed"] == report["total"]
        assert report["max_diff"] < 1e-5
