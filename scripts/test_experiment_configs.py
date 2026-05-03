import pytest
import sys
import os

sys.path.insert(0, os.path.abspath("scripts"))
from experiment_configs import _scale, get_e2e_configs, ALL_EXPERIMENTS

def test_scale():
    cfg = {"embed_size": 256, "num_layers": 4}
    scaled = _scale(cfg, True)
    assert scaled["embed_size"] == 32
    assert scaled["num_layers"] == 2
    assert scaled["epochs"] == 1

    scaled_false = _scale(cfg, False)
    assert scaled_false["embed_size"] == 256

def test_get_e2e_configs():
    configs = get_e2e_configs()
    assert len(configs) == len(ALL_EXPERIMENTS)


def test_main():
    import sys
    from unittest.mock import patch
    with patch.object(sys, "argv", ["experiment_configs.py"]):
        with patch("builtins.print") as mock_print:
            import experiment_configs
            if hasattr(experiment_configs, "__name__") and experiment_configs.__name__ == "experiment_configs":
                # To simulate __main__ execution, we'd have to run it directly
                pass

def test_main_exec():
    import subprocess
    result = subprocess.run(["python", "scripts/experiment_configs.py"], capture_output=True, text=True)
    assert "=== Full configs ===" in result.stdout
    assert "=== E2E (CPU-scaled) configs ===" in result.stdout
