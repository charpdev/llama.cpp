/*
 * TurboQuant CUDA kernels
 *
 * GPU-side rotation for TQ4_0 KV cache quantization.
 * - tq_cuda_init: uploads rotation matrix to GPU
 * - tq_cuda_rotate: applies rotation to float tensor in-place on GPU
 *
 * The rotation is applied before SET_ROWS quantizes into the KV cache,
 * and the inverse rotation is applied after dequantization on read.
 */

#include "common.cuh"

static float * d_rotation   = nullptr;
static float * d_rotation_t = nullptr;
static int     d_head_dim   = 0;

/* rotate kernel: one CUDA block per head
 * blockDim.x = head_dim, each thread computes one output element */
static __global__ void k_tq_rotate(
        float       * __restrict__ dst,
        const float * __restrict__ src,
        const float * __restrict__ R,
        const int head_dim,
        const int64_t n_elements) {

    const int head_idx = blockIdx.x;
    const int i = threadIdx.x;  /* element within head */

    if (i >= head_dim) return;

    const int64_t base = (int64_t)head_idx * head_dim;
    if (base + head_dim > n_elements) return;

    float sum = 0.0f;
    for (int j = 0; j < head_dim; j++) {
        sum += R[i * head_dim + j] * src[base + j];
    }
    dst[base + i] = sum;
}

/* shared-memory version for head_dim <= 1024 */
static __global__ void k_tq_rotate_smem(
        float       * __restrict__ dst,
        const float * __restrict__ src,
        const float * __restrict__ R,
        const int head_dim,
        const int64_t n_elements) {

    extern __shared__ float smem[];

    const int head_idx = blockIdx.x;
    const int i = threadIdx.x;

    const int64_t base = (int64_t)head_idx * head_dim;
    if (base + head_dim > n_elements) return;

    /* load source head into shared memory */
    if (i < head_dim) {
        smem[i] = src[base + i];
    }
    __syncthreads();

    if (i < head_dim) {
        float sum = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            sum += R[i * head_dim + j] * smem[j];
        }
        dst[base + i] = sum;
    }
}

extern "C" void tq_cuda_init(const float * rotation, const float * rotation_t, int head_dim) {
    d_head_dim = head_dim;
    const int n = head_dim * head_dim;

    if (d_rotation)   { CUDA_CHECK(cudaFree(d_rotation));   d_rotation   = nullptr; }
    if (d_rotation_t) { CUDA_CHECK(cudaFree(d_rotation_t)); d_rotation_t = nullptr; }

    CUDA_CHECK(cudaMalloc(&d_rotation,   n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rotation_t, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rotation,   rotation,   n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rotation_t, rotation_t, n * sizeof(float), cudaMemcpyHostToDevice));
}

extern "C" void tq_cuda_cleanup(void) {
    if (d_rotation)   { CUDA_CHECK(cudaFree(d_rotation));   d_rotation   = nullptr; }
    if (d_rotation_t) { CUDA_CHECK(cudaFree(d_rotation_t)); d_rotation_t = nullptr; }
    d_head_dim = 0;
}

/* Apply rotation R to a float buffer on GPU.
 * n_elements must be a multiple of head_dim.
 * Can operate in-place (dst == src). */
extern "C" void tq_cuda_rotate(float * dst, const float * src, bool inverse, int64_t n_elements, cudaStream_t stream) {
    if (d_head_dim == 0) return;

    const float * R = inverse ? d_rotation_t : d_rotation;
    const int n_heads = (int)(n_elements / d_head_dim);

    if (d_head_dim <= 1024) {
        k_tq_rotate_smem<<<n_heads, d_head_dim, d_head_dim * sizeof(float), stream>>>(
            dst, src, R, d_head_dim, n_elements);
    } else {
        k_tq_rotate<<<n_heads, d_head_dim, 0, stream>>>(
            dst, src, R, d_head_dim, n_elements);
    }
}

extern "C" int tq_cuda_get_head_dim(void) { return d_head_dim; }
