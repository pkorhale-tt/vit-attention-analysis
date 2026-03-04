import torch
import ttnn
import math
import argparse
import sys


def get_core_grid(device, num_cores: int = None):
    """
    Calculate core grid based on user-provided single core value.
    """
    grid_size = device.compute_with_storage_grid_size()
    total_cores = grid_size.x * grid_size.y
    
    if num_cores is None:
        target_num_cores = total_cores
    else:
        if num_cores <= 0:
            raise ValueError("num_cores must be positive")
        target_num_cores = min(num_cores, total_cores)
    
    print(f"Device grid    : {grid_size.x} x {grid_size.y} = {total_cores} cores")
    print(f"Requested cores: {num_cores if num_cores else 'all'}")
    print(f"Using cores    : {target_num_cores}")
    
    if target_num_cores <= grid_size.x:
        core_grid = ttnn.CoreGrid(x=target_num_cores, y=1)
    else:
        x = min(grid_size.x, int(math.sqrt(target_num_cores)))
        y = min(grid_size.y, math.ceil(target_num_cores / x))
        core_grid = ttnn.CoreGrid(x=x, y=y)
    
    print(f"Core grid      : {core_grid.x} x {core_grid.y} = {core_grid.x * core_grid.y} cores\n")
    
    return core_grid


def parse_dtype(dtype_str: str):
    """Map string to ttnn dtype."""
    s = dtype_str.lower()
    if s in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    if s in ("fp32", "float32"):
        return ttnn.float32
    if s in ("bfp8", "bfloat8_b", "bfp8_b"):
        return ttnn.bfloat8_b
    raise ValueError(f"Unsupported dtype string: {dtype_str}")


def parse_math_fidelity(fidelity_str: str):
    """Map string to ttnn.MathFidelity."""
    s = fidelity_str.lower()
    if s == "lofi":
        return ttnn.MathFidelity.LoFi
    if s == "hifi2":
        return ttnn.MathFidelity.HiFi2
    if s == "hifi3":
        return ttnn.MathFidelity.HiFi3
    if s == "hifi4":
        return ttnn.MathFidelity.HiFi4
    raise ValueError(f"Unsupported math fidelity: {fidelity_str}")


def parse_memory_config(memory_str: str):
    """Map string to ttnn memory config."""
    s = memory_str.lower()
    if s == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if s == "l1":
        return ttnn.L1_MEMORY_CONFIG
    raise ValueError(f"Unsupported memory config: {memory_str}")


def run_matmul_operations(
    num_cores: int = None,
    dtype_str: str = "bfloat16",
    fidelity_str: str = "HiFi2",
    memory_str: str = "dram",
    num_runs: int = 1,
    device_id: int = 0
):
    """
    Run matrix operations with user-defined parameters.
    """
    
    # Parse parameters
    dtype = parse_dtype(dtype_str)
    math_fidelity = parse_math_fidelity(fidelity_str)
    memory_config = parse_memory_config(memory_str)
    
    # Initialize TT device
    tt_device = ttnn.open_device(device_id=device_id)
    #ttnn.enable_program_cache(tt_device)
    
    print("=" * 50)
    print("TTNN Matrix Operations")
    print("=" * 50)
    print(f"Device ID      : {device_id}")
    print(f"Data type      : {dtype_str}")
    print(f"Math fidelity  : {fidelity_str}")
    print(f"Memory config  : {memory_str}")
    print(f"Num runs       : {num_runs}\n")
    
    try:
        # Get core grid
        core_grid = get_core_grid(tt_device, num_cores)
        
        # Define compute kernel config
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False
        )
        
        for run in range(num_runs):
            print(f"=== Run {run + 1}/{num_runs} ===\n")
            
            # =====================================================
            # 1️⃣ LINEAR / PROJECTION
            # (1,8,224,768) @ (1,1,768,768) → (1,8,224,768)
            # =====================================================
            print("--- Linear Projection ---")
            
            A_linear = ttnn.from_torch(
                torch.randn(1, 8, 224, 768),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=tt_device,
                memory_config=memory_config
            )
            
            B_linear = ttnn.from_torch(
                torch.randn(1, 1, 768, 768),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=tt_device,
                memory_config=memory_config
            )
            
            out_linear = ttnn.matmul(
                A_linear,
                B_linear,
                memory_config=memory_config,
                core_grid=core_grid,
                compute_kernel_config=compute_config
            )
            
            print("Input_0:", A_linear.shape)
            print("Input_1:", B_linear.shape)
            print("Output :", out_linear.shape)
            
            ttnn.deallocate(A_linear)
            ttnn.deallocate(B_linear)
            ttnn.deallocate(out_linear)
            
            # =====================================================
            # 2️⃣ ATTENTION (Q × Kᵀ)
            # (8,12,224,64) @ (8,12,64,224) → (8,12,224,224)
            # =====================================================
            print("\n--- Attention (Q x K^T) ---")
            
            Q = ttnn.from_torch(
                torch.randn(8, 12, 224, 64),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=tt_device,
                memory_config=memory_config
            )
            
            K = ttnn.from_torch(
                torch.randn(8, 12, 224, 64),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=tt_device,
                memory_config=memory_config
            )
            
            K_T = ttnn.transpose(K, -2, -1, memory_config=memory_config)
            
            attn_scores = ttnn.matmul(
                Q,
                K_T,
                memory_config=memory_config,
                core_grid=core_grid,
                compute_kernel_config=compute_config
            )
            
            print("Q shape     :", Q.shape)
            print("K^T shape   :", K_T.shape)
            print("Output shape:", attn_scores.shape)
            
            ttnn.deallocate(Q)
            ttnn.deallocate(K)
            ttnn.deallocate(K_T)
            
            # =====================================================
            # 3️⃣ ATTENTION × V
            # (8,12,224,224) @ (8,12,224,64) → (8,12,224,64)
            # =====================================================
            print("\n--- Attention x V ---")
            
            V = ttnn.from_torch(
                torch.randn(8, 12, 224, 64),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=tt_device,
                memory_config=memory_config
            )
            
            attn_output = ttnn.matmul(
                attn_scores,
                V,
                memory_config=memory_config,
                core_grid=core_grid,
                compute_kernel_config=compute_config
            )
            
            print("Attention shape:", attn_scores.shape)
            print("V shape        :", V.shape)
            print("Output shape   :", attn_output.shape)
            
            ttnn.deallocate(attn_scores)
            ttnn.deallocate(V)
            ttnn.deallocate(attn_output)
            
            print()
        
        print("--- All operations completed successfully! ---")
        
    finally:
        ttnn.close_device(tt_device)
        print("Device closed.")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="TTNN Matrix Operations with configurable parameters"
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="Number of cores to use (default: all available)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type (bfloat16, bfloat8_b, float32)"
    )
    parser.add_argument(
        "--fidelity",
        type=str,
        default="HiFi2",
        help="Math fidelity (LoFi, HiFi2, HiFi3, HiFi4)"
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="dram",
        help="Memory config (dram, l1)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of iterations to run"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID to use"
    )
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    
    try:
        run_matmul_operations(
            num_cores=args.num_cores,
            dtype_str=args.dtype,
            fidelity_str=args.fidelity,
            memory_str=args.memory,
            num_runs=args.num_runs,
            device_id=args.device_id
        )
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

'''
# Basic with 16 cores
python -m tracy -p -r -v standalone_vit_attention_full_support.py --num-cores 16

# Full configuration - 16 cores, bfloat16, HiFi2, DRAM, 10 runs
python -m tracy -p -r -v standalone_vit_attention_full_support.py --num-cores 16 --dtype bfloat16 --fidelity HiFi2 --memory dram --num-runs 10 --device-id 0

# Full configuration - 32 cores, bfloat8_b, HiFi4, L1, 5 runs
python -m tracy -p -r -v standalone_vit_attention_full_support.py --num-cores 32 --dtype bfloat8_b --fidelity HiFi4 --memory l1 --num-runs 5 --device-id 0

# Full configuration - 64 cores (all), bfloat16, HiFi2, DRAM, 10 runs
python -m tracy -p -r -v standalone_vit_attention_full_support.py --num-cores 64 --dtype bfloat16 --fidelity HiFi2 --memory dram --num-runs 10 --device-id 0

# Using all cores (default)
python -m tracy -p -r -v standalone_vit_attention_full_support.py --dtype bfloat16 --fidelity HiFi2 --memory dram --num-runs 10

# Minimal command
python -m tracy -p -r -v standalone_vit_attention_full_support.py --num-cores 16
'''
