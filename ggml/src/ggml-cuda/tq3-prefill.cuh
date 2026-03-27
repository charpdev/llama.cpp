// tq3-prefill.cuh — Native TQ3_0 prefill (PP) kernel
// Design: WHT once per weight block, reuse across all N tokens
// Avoids the q8_0 bridge and the full fp16 dequant of all weights

#pragma once

#include "tq3-native.cuh"

// Tile sizes
#define TQ3_PREFILL_TILE_N 8   // tokens per warp
#define TQ3_PREFILL_WARP   32  // threads per warp = QK_TQ3_0

// One warp computes: one weight row × TQ3_PREFILL_TILE_N tokens
// Grid: (ne01, ceil(ne11/TILE_N))
// Block: 32 threads (one warp)
//
// For each TQ3_0 block in the weight row:
//   1. Subgroup leader loads packed bytes, broadcasts via shuffle
//   2. Each lane extracts its 3-bit index, looks up centroid
//   3. WHT butterfly (5 stages, warp shuffle) — done ONCE per block
//   4. Apply sign + rms scale → exact float x_j
//   5. For each of TILE_N tokens: accumulate x_j * act[j][token]
//
// This amortizes the WHT cost (5 shuffles) across TILE_N tokens.

__global__ void tq3_prefill_kernel(
    const block_tq3_0 * __restrict__ weights,  // [ne01, ne00/32] blocks
    const float       * __restrict__ act,       // [ne11, ne00] row-major
    float             * __restrict__ dst,       // [ne01, ne11] row-major
    int ne00,   // weight inner dim (must be multiple of 32)
    int ne01,   // weight rows (output features)
    int ne11)   // number of tokens
{
    const int out_row   = blockIdx.x;  // which weight row
    const int tok_tile  = blockIdx.y;  // which tile of tokens
    const int lane      = threadIdx.x; // 0..31

    if (out_row >= ne01) return;

    const int n_blocks = ne00 / QK_TQ3_0;
    const int tok0 = tok_tile * TQ3_PREFILL_TILE_N;
    const int tok1 = min(tok0 + TQ3_PREFILL_TILE_N, ne11);

    // Accumulators for TILE_N tokens
    float acc[TQ3_PREFILL_TILE_N] = {};

    const block_tq3_0 * w_row = weights + out_row * n_blocks;

    for (int blk = 0; blk < n_blocks; blk++) {
        const block_tq3_0 * bq = w_row + blk;
        const float rms = __half2float(bq->d);

        // Step 1: subgroup leader loads packed bytes, broadcasts
        const int g      = lane / 8;
        const int r      = lane % 8;
        const int leader = g * 8;

        uint32_t packed = 0;
        if (r == 0) {
            const uint8_t * qp = bq->qs + g * 3;
            packed = (uint32_t)qp[0] | ((uint32_t)qp[1] << 8) | ((uint32_t)qp[2] << 16);
        }
        packed = __shfl_sync(0xFFFFFFFF, packed, leader);

        // Step 2: centroid lookup
        float val = ggml_cuda_tq3_centroid(ggml_cuda_tq3_unpack_idx(packed, r));

        // Step 3: WHT butterfly — done ONCE, reused for all tokens
        #pragma unroll
        for (int step = 1; step < 32; step <<= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
            val = (lane & step) ? (other - val) : (other + val);
        }

        // Step 4: apply sign + scale
        const float w_j = val * ggml_cuda_tq3_sign(lane) * (rms / sqrtf((float)QK_TQ3_0));

        // Step 5: dot with each token's activation slice
        const int act_base = blk * QK_TQ3_0 + lane;  // element index in activation
        #pragma unroll
        for (int t = 0; t < TQ3_PREFILL_TILE_N; t++) {
            const int tok = tok0 + t;
            if (tok < ne11) {
                acc[t] += w_j * act[tok * ne00 + act_base];
            }
        }
    }

    // Warp reduce each accumulator and write output
    #pragma unroll
    for (int t = 0; t < TQ3_PREFILL_TILE_N; t++) {
        float sum = acc[t];
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1)
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, m);
        if (lane == 0) {
            const int tok = tok0 + t;
            if (tok < ne11)
                dst[out_row * ne11 + tok] = sum;
        }
    }
}

// Launch wrapper
static void tq3_prefill_launch(
    const block_tq3_0 * weights,
    const float       * act,
    float             * dst,
    int ne00, int ne01, int ne11,
    cudaStream_t stream)
{
    const dim3 grid(ne01, (ne11 + TQ3_PREFILL_TILE_N - 1) / TQ3_PREFILL_TILE_N);
    const dim3 block(TQ3_PREFILL_WARP);
    tq3_prefill_kernel<<<grid, block, 0, stream>>>(weights, act, dst, ne00, ne01, ne11);
}
