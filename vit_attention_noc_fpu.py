import subprocess
from pathlib import Path
import time
import os
import shutil

# =============================
# CONFIG
# =============================


# Enable device profiler + sync (recommended for NPE)
os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
os.environ["TT_METAL_PROFILER_SYNC"] = "1"

# Cores: adjust to list(range(1, 65, 3)) for full sweep
CORES = list(range(1, 5))  # 1, 4, ...

TT_METAL_ROOT = Path(".").resolve()

# Root folder for everything (both FPU + NOC)
OUTPUT_ROOT = TT_METAL_ROOT / "vit_attention_core_sweep"
OUTPUT_ROOT.mkdir(exist_ok=True)

# Enable device profiler globally
os.environ["TT_METAL_DEVICE_PROFILER"] = "1"

print(f"Outputs will go under: {OUTPUT_ROOT}\n")

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
# SWEEP OVER CORES (FPU + NOC)
# =============================

for core in CORES:
    # Per-core root: vit_attention_core_sweep/core_<core>/
    core_root = OUTPUT_ROOT / f"core_{core}"
    core_root.mkdir(exist_ok=True)

    os.environ["NUM_CORES_OVERRIDE"] = str(core)

    print(f"▶ Running core={core}")

    # --------------------------------------------------
    # 1) FPU RUN (profile_this.py)
    #    -> vit_attention_core_sweep/core_<core>/fpu/reports/...
    # --------------------------------------------------
    fpu_folder = core_root / "fpu"
    fpu_folder.mkdir(exist_ok=True)

    # Skip if FPU already has reports
    fpu_reports_dir = fpu_folder / "reports"
    if fpu_reports_dir.exists():
        print(f"  ⏩ [FPU] Skipping core={core} (reports already exist at {fpu_reports_dir})")
    else:
        print(f"  ▶ [FPU] core={core}")
        cmd_fpu = f"""
        ./tools/tracy/profile_this.py \
          -o {fpu_folder} \
          -n vit_attention_fpu \
          --profiler-capture-perf-counters=fpu \
          -c 'pytest VitModelTesting/test_vit_attention_dynamic.py::test_vit_attention -q --disable-warnings'
        """

        try:
            subprocess.run(cmd_fpu, shell=True, check=True, cwd=TT_METAL_ROOT)
        except subprocess.CalledProcessError:
            print(f"  ❌ [FPU] Failed at core={core}. Continuing to next core.")
            continue

        time.sleep(1)

        if fpu_reports_dir.exists():
            print(f"  ✅ [FPU] Reports at: {fpu_reports_dir}")
        else:
            print(f"  ⚠ [FPU] No reports/ folder found for core={core}")

    # --------------------------------------------------
    # 2) NOC/DRAM RUN (NPE via python -m tracy)
    #    -> vit_attention_core_sweep/core_<core>/noc/<timestamp>/...
    # --------------------------------------------------
    noc_root = core_root / "noc"
    noc_root.mkdir(exist_ok=True)

    # We use python -m tracy directly to avoid host/device mismatch assert.
    print(f"  ▶ [NOC/NPE] core={core}")
    cmd_noc = f"""
    python -m tracy -p -r -v \
        --collect-noc-traces \
        -m pytest VitModelTesting/test_vit_attention_dynamic.py::test_vit_attention \
        --disable-warnings -q
    """

    try:
        subprocess.run(cmd_noc, shell=True, check=True, cwd=TT_METAL_ROOT)
    except subprocess.CalledProcessError:
        print(f"  ❌ [NOC/NPE] Failed at core={core}. You can rerun to resume.")
        continue

    time.sleep(1)

    # tracy -r writes to generated/profiler/reports/<timestamp>/
    reports_root = TT_METAL_ROOT / "generated" / "profiler" / "reports"
    timestamp_dirs = sorted(
        [d for d in reports_root.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )

    if not timestamp_dirs:
        print(f"  ⚠ [NOC/NPE] No reports/ folder found in generated/profiler/reports for core={core}")
    else:
        latest = timestamp_dirs[-1]
        target = noc_root / latest.name
        if not target.exists():
            shutil.copytree(latest, target)
        print(f"  ✅ [NOC/NPE] Reports at: {target}")

    print()

print("\n✅ Core sweep complete (FPU + NOC).")
print(f"📁 All outputs under: {OUTPUT_ROOT}")
