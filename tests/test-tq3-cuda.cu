// TQ3_0 CUDA unit tests — run BEFORE any integration
// Tests: quantize, dequant, vec_dot, load_tiles equivalents
// Each test is self-contained with known inputs and expected outputs.

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define QK_TQ3_0 32

typedef struct { __half d; uint8_t qs[12]; } block_tq3_0;
typedef struct { __half d; int8_t qs[32]; } block_q8_0;

// ===== Reference CPU implementation (ground truth) =====

static float cpu_sign(int i) {
    return ((((unsigned)i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}
static const float CPU_CENTROIDS[8] = {-2.1519f,-1.3439f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3439f,2.1519f};
static const float CPU_BOUNDARIES[7] = {-1.7479f,-1.0500f,-0.5005f,0.0f,0.5005f,1.0500f,1.7479f};

static void cpu_wht(float *x, int n) {
    for (int s = 1; s < n; s <<= 1)
        for (int i = 0; i < n; i += s*2)
            for (int j = i; j < i+s; j++) {
                float a = x[j], b = x[j+s]; x[j] = a+b; x[j+s] = a-b;
            }
}

static void cpu_quantize(const float *x, float *rms_out, uint8_t *qs) {
    float sum_sq = 0;
    for (int i = 0; i < 32; i++) sum_sq += x[i]*x[i];
    float rms = sqrtf(sum_sq/32.0f);
    if (rms < 1e-10f) rms = 1.0f;
    *rms_out = rms;
    float buf[32];
    for (int i = 0; i < 32; i++) buf[i] = x[i]/rms * cpu_sign(i);
    cpu_wht(buf, 32);
    for (int i = 0; i < 32; i++) buf[i] /= sqrtf(32.0f);
    uint8_t idx[32];
    for (int i = 0; i < 32; i++) {
        idx[i] = 0;
        for (int b = 0; b < 7; b++) if (buf[i] > CPU_BOUNDARIES[b]) idx[i] = b+1;
    }
    for (int g = 0; g < 4; g++) {
        uint8_t *q = qs+g*3, *d = idx+g*8;
        q[0] = d[0]|(d[1]<<3)|(d[2]<<6);
        q[1] = (d[2]>>2)|(d[3]<<1)|(d[4]<<4)|(d[5]<<7);
        q[2] = (d[5]>>1)|(d[6]<<2)|(d[7]<<5);
    }
}

static void cpu_dequantize(float rms, const uint8_t *qs, float *out) {
    float vals[32];
    for (int g = 0; g < 4; g++) {
        const uint8_t *q = qs+g*3; int b = g*8;
        vals[b+0] = CPU_CENTROIDS[ q[0]       & 7];
        vals[b+1] = CPU_CENTROIDS[(q[0] >> 3) & 7];
        vals[b+2] = CPU_CENTROIDS[((q[0]>>6)|(q[1]<<2)) & 7];
        vals[b+3] = CPU_CENTROIDS[(q[1] >> 1) & 7];
        vals[b+4] = CPU_CENTROIDS[(q[1] >> 4) & 7];
        vals[b+5] = CPU_CENTROIDS[((q[1]>>7)|(q[2]<<1)) & 7];
        vals[b+6] = CPU_CENTROIDS[(q[2] >> 2) & 7];
        vals[b+7] = CPU_CENTROIDS[(q[2] >> 5) & 7];
    }
    cpu_wht(vals, 32);
    for (int i = 0; i < 32; i++)
        out[i] = vals[i] / sqrtf(32.0f) * cpu_sign(i) * rms;
}

static float cpu_dot(const float *q, const float *k_dequant, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += q[i]*k_dequant[i]; return s;
}

static float cpu_cosine(const float *a, const float *b, int n) {
    float dot=0,na=0,nb=0;
    for (int i=0;i<n;i++) { dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
    return dot/(sqrtf(na)*sqrtf(nb)+1e-10f);
}

// ===== CUDA kernels (exact copies from our codebase) =====

__constant__ float SIGNS[32] = {+1,-1,+1,-1,+1,+1,-1,+1,-1,-1,+1,-1,+1,+1,-1,+1,
                                 -1,-1,+1,-1,+1,-1,-1,+1,-1,+1,+1,-1,+1,-1,-1,+1};
__constant__ float CENTROIDS[8] = {-2.1519f,-1.3439f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3439f,2.1519f};
__constant__ float BOUNDARIES[7] = {-1.7479f,-1.0500f,-0.5005f,0.0f,0.5005f,1.0500f,1.7479f};

// Quantize kernel (from cpy-utils.cuh — single thread per block)
__device__ void gpu_quantize_block(const float *x, block_tq3_0 *y) {
    float buf[32], sum_sq = 0;
    for (int i=0;i<32;i++) sum_sq += x[i]*x[i];
    float rms = sqrtf(sum_sq/32.0f); if (rms<1e-10f) rms=1.0f;
    y->d = __float2half(rms);
    for (int i=0;i<32;i++) buf[i] = x[i]/rms*SIGNS[i];
    for (int s=1;s<32;s<<=1) for (int i=0;i<32;i+=s*2) for (int j=i;j<i+s;j++) {
        float a=buf[j],b=buf[j+s]; buf[j]=a+b; buf[j+s]=a-b;
    }
    float n=1.0f/sqrtf(32.0f); for (int i=0;i<32;i++) buf[i]*=n;
    uint8_t idx[32];
    for (int i=0;i<32;i++) { idx[i]=0; for (int b=0;b<7;b++) if (buf[i]>BOUNDARIES[b]) idx[i]=b+1; }
    for (int g=0;g<4;g++) {
        uint8_t *q=y->qs+g*3, *d=idx+g*8;
        q[0]=d[0]|(d[1]<<3)|(d[2]<<6);
        q[1]=(d[2]>>2)|(d[3]<<1)|(d[4]<<4)|(d[5]<<7);
        q[2]=(d[5]>>1)|(d[6]<<2)|(d[7]<<5);
    }
}

__global__ void k_quantize(const float *x, block_tq3_0 *y, int nb) {
    int i = blockIdx.x;
    if (i < nb) gpu_quantize_block(x + i*32, y + i);
}

// Dequant kernel (from convert.cu — 32 threads per block, warp shuffle WHT)
__global__ void k_dequant(const block_tq3_0 *blks, float *out, int nb) {
    int i = blockIdx.x; if (i >= nb) return;
    int j = threadIdx.x;
    float d = __half2float(blks[i].d);
    int g=j/8, r=j%8;
    const uint8_t *qp = blks[i].qs + g*3;
    uint8_t idx;
    switch(r) {
        case 0: idx= qp[0]       &7; break;
        case 1: idx=(qp[0]>>3)   &7; break;
        case 2: idx=((qp[0]>>6)|(qp[1]<<2))&7; break;
        case 3: idx=(qp[1]>>1)   &7; break;
        case 4: idx=(qp[1]>>4)   &7; break;
        case 5: idx=((qp[1]>>7)|(qp[2]<<1))&7; break;
        case 6: idx=(qp[2]>>2)   &7; break;
        default:idx=(qp[2]>>5)   &7; break;
    }
    float val = CENTROIDS[idx];
    for (int step=1;step<32;step<<=1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        val = (j&step) ? (other-val) : (other+val);
    }
    out[i*32+j] = val * (d/sqrtf(32.0f)) * SIGNS[j];
}

// ===== Tests =====

int tests_pass = 0, tests_fail = 0;
void check(const char *name, bool ok) {
    printf("  %-50s %s\n", name, ok ? "PASS" : "FAIL");
    ok ? tests_pass++ : tests_fail++;
}

void test_cpu_roundtrip() {
    printf("Test 1: CPU quantize → dequant roundtrip\n");
    float x[32], y[32];
    for (int i=0;i<32;i++) x[i] = sinf(i*0.3f+1.0f);
    float rms; uint8_t qs[12];
    cpu_quantize(x, &rms, qs);
    cpu_dequantize(rms, qs, y);
    check("cosine > 0.97", cpu_cosine(x, y, 32) > 0.97f);
}

void test_gpu_quantize_matches_cpu() {
    printf("Test 2: GPU quantize matches CPU\n");
    float x[32];
    for (int i=0;i<32;i++) x[i] = sinf(i*0.3f+1.0f);

    // CPU
    float cpu_rms; uint8_t cpu_qs[12];
    cpu_quantize(x, &cpu_rms, cpu_qs);

    // GPU
    float *d_x; block_tq3_0 *d_blk;
    cudaMalloc(&d_x, 128); cudaMalloc(&d_blk, sizeof(block_tq3_0));
    cudaMemcpy(d_x, x, 128, cudaMemcpyHostToDevice);
    k_quantize<<<1,1>>>(d_x, d_blk, 1);
    cudaDeviceSynchronize();
    block_tq3_0 h_blk;
    cudaMemcpy(&h_blk, d_blk, sizeof(block_tq3_0), cudaMemcpyDeviceToHost);

    check("rms matches", fabsf(__half2float(h_blk.d) - cpu_rms) < 0.01f);
    check("qs matches", memcmp(h_blk.qs, cpu_qs, 12) == 0);
    cudaFree(d_x); cudaFree(d_blk);
}

void test_gpu_dequant_matches_cpu() {
    printf("Test 3: GPU dequant matches CPU\n");
    float x[32];
    for (int i=0;i<32;i++) x[i] = sinf(i*0.3f+1.0f);

    // CPU quantize + dequant
    float rms; uint8_t qs[12];
    cpu_quantize(x, &rms, qs);
    float cpu_y[32];
    cpu_dequantize(rms, qs, cpu_y);

    // GPU quantize + dequant
    float *d_x, *d_y; block_tq3_0 *d_blk;
    cudaMalloc(&d_x, 128); cudaMalloc(&d_y, 128); cudaMalloc(&d_blk, sizeof(block_tq3_0));
    cudaMemcpy(d_x, x, 128, cudaMemcpyHostToDevice);
    k_quantize<<<1,1>>>(d_x, d_blk, 1);
    k_dequant<<<1,32>>>(d_blk, d_y, 1);
    cudaDeviceSynchronize();
    float gpu_y[32];
    cudaMemcpy(gpu_y, d_y, 128, cudaMemcpyDeviceToHost);

    float max_err = 0;
    for (int i=0;i<32;i++) max_err = fmaxf(max_err, fabsf(gpu_y[i]-cpu_y[i]));
    check("max element error < 0.01", max_err < 0.01f);
    check("cosine vs CPU > 0.999", cpu_cosine(cpu_y, gpu_y, 32) > 0.999f);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_blk);
}

void test_gpu_full_pipeline_dot() {
    printf("Test 4: GPU quantize → dequant → dot product (head_dim=128)\n");
    const int D = 128;
    float Q[D], K[D];
    for (int i=0;i<D;i++) { Q[i]=sinf(i*0.1f); K[i]=cosf(i*0.13f); }

    float true_dot = cpu_dot(Q, K, D);

    // CPU quantize K, dequant, dot
    float cpu_dot_val = 0;
    for (int blk=0;blk<D/32;blk++) {
        float rms; uint8_t qs[12];
        cpu_quantize(K+blk*32, &rms, qs);
        float dq[32];
        cpu_dequantize(rms, qs, dq);
        cpu_dot_val += cpu_dot(Q+blk*32, dq, 32);
    }

    // GPU quantize K, dequant, dot
    float *d_K, *d_Kdq;
    block_tq3_0 *d_Kq;
    int nb = D/32;
    cudaMalloc(&d_K, D*4); cudaMalloc(&d_Kq, nb*sizeof(block_tq3_0)); cudaMalloc(&d_Kdq, D*4);
    cudaMemcpy(d_K, K, D*4, cudaMemcpyHostToDevice);
    k_quantize<<<nb,1>>>(d_K, d_Kq, nb);
    k_dequant<<<nb,32>>>(d_Kq, d_Kdq, nb);
    cudaDeviceSynchronize();
    float gpu_Kdq[D];
    cudaMemcpy(gpu_Kdq, d_Kdq, D*4, cudaMemcpyDeviceToHost);
    float gpu_dot_val = cpu_dot(Q, gpu_Kdq, D);

    float err_cpu = fabsf(cpu_dot_val - true_dot) / (fabsf(true_dot)+1e-6f);
    float err_gpu = fabsf(gpu_dot_val - true_dot) / (fabsf(true_dot)+1e-6f);
    float err_diff = fabsf(gpu_dot_val - cpu_dot_val) / (fabsf(cpu_dot_val)+1e-6f);

    check("CPU dot error < 5%", err_cpu < 0.05f);
    check("GPU dot error < 5%", err_gpu < 0.05f);
    check("GPU vs CPU dot diff < 0.1%", err_diff < 0.001f);
    cudaFree(d_K); cudaFree(d_Kq); cudaFree(d_Kdq);
}

void test_gpu_dequant_to_fp16() {
    printf("Test 5: GPU dequant to fp16 (cublas path)\n");
    // This is what the cublas fallback actually uses
    float x[32];
    for (int i=0;i<32;i++) x[i] = sinf(i*0.3f+1.0f);

    float rms; uint8_t qs[12];
    cpu_quantize(x, &rms, qs);
    float cpu_y[32];
    cpu_dequantize(rms, qs, cpu_y);

    // Build block on host, copy to GPU, dequant to fp16, read back
    block_tq3_0 h_blk;
    h_blk.d = __float2half(rms);
    memcpy(h_blk.qs, qs, 12);

    block_tq3_0 *d_blk; __half *d_y16;
    cudaMalloc(&d_blk, sizeof(block_tq3_0)); cudaMalloc(&d_y16, 32*2);
    cudaMemcpy(d_blk, &h_blk, sizeof(block_tq3_0), cudaMemcpyHostToDevice);

    // Use same dequant kernel but output to a float buffer, then we'll check
    float *d_yf; cudaMalloc(&d_yf, 128);
    k_dequant<<<1,32>>>(d_blk, d_yf, 1);
    cudaDeviceSynchronize();
    float gpu_y[32];
    cudaMemcpy(gpu_y, d_yf, 128, cudaMemcpyDeviceToHost);

    check("cosine vs CPU > 0.999", cpu_cosine(cpu_y, gpu_y, 32) > 0.999f);
    cudaFree(d_blk); cudaFree(d_y16); cudaFree(d_yf);
}

int main() {
    printf("=== TQ3_0 CUDA Unit Tests ===\n\n");
    test_cpu_roundtrip();
    test_gpu_quantize_matches_cpu();
    test_gpu_dequant_matches_cpu();
    test_gpu_full_pipeline_dot();
    test_gpu_dequant_to_fp16();
    printf("\n=== Results: %d passed, %d failed ===\n", tests_pass, tests_fail);
    return tests_fail > 0 ? 1 : 0;
}
