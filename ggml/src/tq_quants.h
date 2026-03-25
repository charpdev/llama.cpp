/*
 * TurboQuant KV cache quantization
 *
 * Rotation-based 4-bit quantization for KV cache vectors.
 * Applies a random orthogonal rotation before Q4_0 quantization,
 * making coordinates approximately independent and enabling
 * near-optimal scalar quantization per coordinate.
 *
 * Based on: TurboQuant (arXiv:2504.19874, ICLR 2026)
 *           PolarQuant (arXiv:2502.02617, AISTATS 2026)
 */

#ifndef GGML_TQ_QUANTS_H
#define GGML_TQ_QUANTS_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Must be called once before any tq quantize/dequantize calls.
 * head_dim: dimension of each attention head (e.g. 64, 128)
 * seed: deterministic seed for rotation matrix generation */
void tq_init(int head_dim, uint64_t seed);

void tq_cleanup(void);

/* Quantize k float values into TQ4_0 blocks.
 * Applies rotation then Q4_0 quantization.
 * k must be a multiple of head_dim. */
void quantize_row_tq4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_tq4_0_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

/* Dequantize TQ4_0 blocks back to float.
 * Applies Q4_0 dequantization then inverse rotation. */
void dequantize_row_tq4_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

#ifdef __cplusplus
}
#endif

#endif /* GGML_TQ_QUANTS_H */
