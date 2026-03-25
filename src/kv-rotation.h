/*
 * KV cache rotation matrix generation
 *
 * Generates a deterministic random orthogonal matrix for KV cache
 * rotation (TurboQuant/PolarQuant style). The rotation Gaussianizes
 * coordinate distributions before quantization, reducing error.
 *
 * Based on: TurboQuant (arXiv:2504.19874), PolarQuant (arXiv:2502.02617)
 */

#ifndef KV_ROTATION_H
#define KV_ROTATION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Generate a head_dim x head_dim orthogonal rotation matrix (row-major, transposed).
 * Caller must free() the returned pointer.
 * Returns R^T so that ggml_mul_mat(result, x) computes R * x. */
float * kv_rotation_generate(int head_dim, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif
