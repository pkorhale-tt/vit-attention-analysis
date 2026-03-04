This repository contains scripts to analyze the sweep patterns of Vision Transformer (ViT) attention.
=======================================================================================================

Wormhole ViT Attention Profiling (NORMAL / FPU / NOC-NPE)

This repo contains scripts to profile a ViT attention workload on a Tenstorrent Wormhole system using tt-metal, tracy, and tt-npe. It automates running the same test with different profiling modes and core counts, and organizes the reports per core.

Supported profiling modes:

NORMAL – standard Tracy device profiling.
FPU – Tracy + device perf counters for FPU utilization.
NOC / NPE – Tracy NoC tracing integrated with tt-npe to get NoC and DRAM BW utilization.
(Optional) FULL PERF COUNTERS – fpu,sfpu,noc,dram.
