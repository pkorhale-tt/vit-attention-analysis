import torch
import ttnn
import math
import argparse
import sys


def get_core_grid(device, num_cores: int = None):
    """
    Calculate core grid based on user-provided single core value.
    Similar logic to the reference code.
    
    Args:
        device: TT device
        num_cores: Number of cores to use (None = use all available)
    
    Returns:
        ttnn.CoreGrid object
    """
    # Get device grid size
    grid_size = device.compute_with_storage_grid_size()
    total_cores = grid_size.x * grid_size.y
    
    # Determine target cores
    if num_cores is None:
        target_num_cores = total_cores  # default: all cores
    else:
        if num_cores <= 0:
            raise ValueError("num_cores must be positive")
        target_num_cores = min(num_cores, total_cores)
    
    print(f"Device grid    : {grid_size.x} x {grid_size.y} = {total_cores} cores")
    print(f"Requested cores: {num_cores if num_cores else 'all'}")
    print(f"Using cores    : {target_num_cores}")
    
    # Simple rectangular grid selection
    if target_num_cores <= grid_size.x:
        core_grid = ttnn.CoreGrid(x=target_num_cores, y=1)
    else:
        x = min(grid_size.x, int(math.sqrt(target_num_cores)))
        y = min(grid_size.y, math.ceil(target_num_cores / x))
        core_grid = ttnn.CoreGrid(x=x, y=y)
    
    print(f"Core grid      : {core_grid.x} x {core_grid.y} = {core_grid.x * core_grid.y} cores\n")
    
    return core_grid


def run_matmul_operations(num_cores: int = None):
    """
    Run matrix operations with user-defined number of cores.
    
    Args:
        num_cores: Number of cores to use (None = use all available)
    """
    
    # Initialize TT device
    device_id = 0
    tt_device = ttnn.open_device(device_id=device_id)
    
    # Enable program cache for better performance
    ttnn.enable_program_cache(tt_device)
    
    print("Running on: TT Device")
    print(f"Device ID: {device_id}\n")
    
    try:
        # Get core grid based on user input
        core_grid = get_core_grid(tt_device, num_cores)
        
        # Define compute kernel config with HiFi2
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False
        )
        
        # =====================================================
        # 1️⃣ LINEAR / PROJECTION
        # (1,8,224,768) @ (1,1,768,768)
        # → (1,8,224,768)
        # =====================================================
        print("--- Linear Projection ---")
        
        A_linear_torch = torch.randn(1, 8, 224, 768)
        B_linear_torch = torch.randn(1, 1, 768, 768)
        
        A_linear = ttnn.from_torch(
            A_linear_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        
        B_linear = ttnn.from_torch(
            B_linear_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        
        out_linear = ttnn.matmul(
            A_linear,
            B_linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        # (8,12,224,64) @ (8,12,64,224)
        # → (8,12,224,224)
        # =====================================================
        print("\n--- Attention (Q x K^T) ---")
        
        Q_torch = torch.randn(8, 12, 224, 64)
        K_torch = torch.randn(8, 12, 224, 64)
        
        Q = ttnn.from_torch(
            Q_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        
        K = ttnn.from_torch(
            K_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        
        K_T = ttnn.transpose(K, -2, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        attn_scores = ttnn.matmul(
            Q,
            K_T,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        # (8,12,224,224) @ (8,12,224,64)
        # → (8,12,224,64)
        # =====================================================
        print("\n--- Attention x V ---")
        
        V_torch = torch.randn(8, 12, 224, 64)
        
        V = ttnn.from_torch(
            V_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        
        attn_output = ttnn.matmul(
            attn_scores,
            V,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=core_grid,
            compute_kernel_config=compute_config
        )
        
        print("Attention shape:", attn_scores.shape)
        print("V shape        :", V.shape)
        print("Output shape   :", attn_output.shape)
        
        ttnn.deallocate(attn_scores)
        ttnn.deallocate(V)
        ttnn.deallocate(attn_output)
        
        print("\n--- All operations completed successfully! ---")
        
    finally:
        # Close TT device
        ttnn.close_device(tt_device)
        print("Device closed.")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="TTNN Matrix Operations with configurable cores"
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="Number of cores to use (default: all available)"
    )
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    
    try:
        run_matmul_operations(num_cores=args.num_cores)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
