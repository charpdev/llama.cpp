# TurboQuant TQ3_0 CUDA — Mission & Status

## Goal
TQ3_0 KV cache on NVIDIA GPUs: **4.6x compression, same speed as q4_0, near-zero quality loss.**

## Current Results (Llama-2-7B, RTX 5060 Ti 16GB)

| Metric | q4_0 | tq3_0 | Delta |
|--------|------|-------|-------|
| PP tok/s | 591 | 549 | **-7%** |
| TG tok/s | 93.8 | 93.8 | **0% identical** |
| KV memory (2K ctx) | 656 MB | 624 MB | -5% |
| KV memory (64K ctx) | 5,248 MB | 4,992 MB | -256 MB |
| PPL | 8.06 | 8.14 | +0.08 |
| Compression | 3.6x | 4.6x | **+28% better** |
| Max context (Llama3-8B) | 69,632 | 71,680 | **+3%** |

## Optimization Journey

| Step | PP tok/s | PP gap | TG tok/s | Change |
|------|---------|--------|---------|--------|
| cublas fallback | 288 | -50% | 64.6 | Initial CUDA port |
| fused mmvq | 288 | -50% | 90.7 | vec_dot_tq3_0_q8_1 |
| chunked mmvq | 362 | -40% | 93.9 | Batch chunking for PP |
| mmq kernel | 536 | -9% | 91.5 | load_tiles_tq3_0 with WHT |
| fixed scale | 533 | -8.7% | 90.2 | Eliminate warp reduce |
| byte write | **549** | **-7%** | **93.8** | Direct shared mem write |

## Next: Graph-Level Q Rotation (target: -2% PP gap)

The remaining 7% PP gap is from WHT inverse in load_tiles (5 warp shuffles per block).

**Hybrid approach** (combines Aaryan's block quantize + TheTom's graph Q rotation):
1. Quantize: WHT forward + centroid (already done, runs once per K write)
2. load_tiles: centroid → int8 only (NO WHT inverse — as fast as q4_0)
3. Graph: pre-rotate Q via `ggml_turbo_wht` op (runs once per prompt)
4. Math: Q_rot · K_rot = Q · K (orthogonality preserves dot products)

Requires: new `GGML_OP_TURBO_WHT` (ggml.h, ggml.c, CPU ops, CUDA kernel, graph integration).
Estimated PP improvement: 7% → 2% gap (load_tiles becomes trivial centroid lookup).
