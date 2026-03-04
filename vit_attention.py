import subprocess
from pathlib import Path
import time
import os

# =============================
# CONFIG
# =============================

CORES = list(range(1, 4, 3))

TT_METAL_ROOT = Path(".").resolve()
OUTPUT_ROOT = TT_METAL_ROOT / "vit_attention_fpu"
OUTPUT_ROOT.mkdir(exist_ok=True)

# Enable device profiler globally
os.environ["TT_METAL_DEVICE_PROFILER"] = "1"

# OPTIONAL: if you also want FPU counters like before, uncomment:
# os.environ["TT_METAL_PROFILER_PROGRAM_CAPTURE_PERF_COUNTERS"] = "fpu"

# =============================
# DISCOVER COMPLETED CORES
# =============================

completed_cores = set()

for p in OUTPUT_ROOT.glob("core_*"):
    try:
        c = int(p.name.replace("core_", ""))
        completed_cores.add(c)
    except ValueError:
        continue

if completed_cores:
    print(f"🔄 Found existing runs for cores: {sorted(completed_cores)}\n")
else:
    print("🆕 No previous results found. Starting fresh...\n")

# =============================
# SWEEP OVER CORES
# =============================

for core in CORES:
    if core in completed_cores:
        print(f"⏩ Skipping core={core} (already has output folder)")
        continue

    print(f"▶ Running core={core}")

    # Each core gets its own output folder under vit_attention_core_sweep
    output_folder = OUTPUT_ROOT / f"core_{core}"
    os.environ["NUM_CORES_OVERRIDE"] = str(core)

    # profile_this.py invocation:
    # -o: where tracy/ttnn reports go
    # -c: pytest test that calls your ViT attention script
    cmd = f"""
    ./tools/tracy/profile_this.py \
      -o {output_folder} \
      -n vit_attention \
      --profiler-capture-perf-counters=fpu \
      -c 'pytest VitModelTesting/test_vit_attention_dynamic.py::test_vit_attention -q --disable-warnings'
    """

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Failed at core={core}. You can rerun to resume.")
        continue

    # Small delay to avoid any timestamp collisions
    time.sleep(1)

    # Light sanity check: did we get a report dir?
    reports_dir = output_folder / "reports"
    if not reports_dir.exists():
        print(f"⚠ No reports/ folder found for core={core}")
    else:
        # Just log that something was generated
        print(f"✅ Tracy/TTNN reports generated under: {reports_dir}")

    print()

print("\n✅ Core sweep complete (or resumed).")
print(f"📁 All per-core outputs under: {OUTPUT_ROOT}")
