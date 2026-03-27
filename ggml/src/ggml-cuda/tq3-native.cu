#include "tq3-native.cuh"

__global__ void ggml_cuda_native_tq3_dot_kernel(
        const block_tq3_0 * __restrict__ in,
        const block_q8_0  * __restrict__ act,
        float * __restrict__ out,
        int nblocks) {

    const int blk = blockIdx.x;
    if (blk >= nblocks) {
        return;
    }

    out[blk] = vec_dot_tq3_0_q8_0_native_block(in + blk, act + blk);
}
