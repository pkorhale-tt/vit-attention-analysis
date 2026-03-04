import runpy
import sys
from pathlib import Path
import os
import pytest

@pytest.mark.parametrize("num_runs", [2])  # or any sweep you want
def test_vit_attention(num_runs):
    script_path = Path("VitModelTesting/standalone_vit_attention_full_support.py")

    # Optional: core count from env (same pattern as matmul)
    num_cores = os.environ.get("NUM_CORES_OVERRIDE", "16")

    sys.argv = [
        str(script_path),
        "--num-cores", str(num_cores),
        "--dtype", "bfloat16",
        "--fidelity", "HiFi2",
        "--memory", "dram",
        "--num-runs", str(num_runs),
        "--device-id", "0",
    ]

    runpy.run_path(str(script_path), run_name="__main__")
