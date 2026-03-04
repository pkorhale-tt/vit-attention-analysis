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

# Cores: adjust as needed (e.g., range(1, 65, 3))
CORES = list(range(1, 5))

TT_METAL_ROOT = Path(".").resolve()

# Root folder for everything (NORMAL + FPU + NOC)
OUTPUT_ROOT = TT_METAL_ROOT / "vit_attention_core_sweep_temp"
OUTPUT_ROOT.mkdir(exist_ok=True)

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

# SWEEP OVER CORES (NORMAL + FPU + NOC)

# =============================

for core in CORES:
    core_root = OUTPUT_ROOT / f"core_{core}"
    core_root.mkdir(exist_ok=True)

    os.environ["NUM_CORES_OVERRIDE"] = str(core)

    print(f"▶ Running core={core}")

    # --------------------------------------------------
    # 0) NORMAL RUN (python -m tracy, no extra flags)
    #    -> vit_attention_core_sweep/core_<core>/normal_trace_report/<timestamp>/...
    # --------------------------------------------------
    normal_root = core_root / "normal_trace_report"
    normal_root.mkdir(exist_ok=True)

    print(f"  ▶ [NORMAL] core={core}")
    cmd_normal = f"""
    python -m tracy -p -r -m pytest \
        VitModelTesting/test_vit_attention_dynamic.py::test_vit_attention \
        -q --disable-warnings
    """

    try:
        subprocess.run(cmd_normal, shell=True, check=True, cwd=TT_METAL_ROOT)
    except subprocess.CalledProcessError:
        print(f"  ❌ [NORMAL] Failed at core={core}. Continuing to next core.")
        continue

    time.sleep(1)

    # tracy -r writes to generated/profiler/reports/<timestamp>/
    reports_root = TT_METAL_ROOT / "generated" / "profiler" / "reports"
    timestamp_dirs = sorted(
        [d for d in reports_root.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )

    if not timestamp_dirs:
        print(f"  ⚠ [NORMAL] No reports/ folder found in generated/profiler/reports for core={core}")
        continue
    else:
        latest = timestamp_dirs[-1]
        target = normal_root / latest.name
        if not target.exists():
            shutil.copytree(latest, target)
        print(f"  ✅ [NORMAL] Reports at: {target}")

    # --------------------------------------------------
    # 1) FPU RUN (profile_this.py)
    #    -> vit_attention_core_sweep/core_<core>/fpu/reports/...
    # --------------------------------------------------
    fpu_folder = core_root / "fpu"
    fpu_folder.mkdir(exist_ok=True)

    fpu_reports_dir = fpu_folder / "reports"
    if fpu_reports_dir.exists():
        print(f"  ⏩ [FPU] Skipping core={core} (reports already exist at {fpu_reports_dir})")
    else:
        print(f"  ▶ [FPU] core={core}")
        cmd_fpu = f"""
        ./tools/tracy/profile_this.py \
          -o {fpu_folder} \
          -n vit_attention_fpu \
          --profiler-capture-perf-counters=fpu,sfpu,noc,dram \
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

    print(f"  ▶ [NOC/NPE] core={core}")
    cmd_noc = f"""
    python -m tracy -p -r -v \
        --collect-noc-traces \
        -m pytest VitModelTesting/test_vit_attention_dynamic.py::test_vit_attention \
        -q --disable-warnings
    """

    try:
        subprocess.run(cmd_noc, shell=True, check=True, cwd=TT_METAL_ROOT)
    except subprocess.CalledProcessError:
        print(f"  ❌ [NOC/NPE] Failed at core={core}. You can rerun to resume.")
        continue

    time.sleep(1)

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

print("\n✅ Core sweep complete (NORMAL + FPU + NOC).")
print(f"📁 All outputs under: {OUTPUT_ROOT}")
