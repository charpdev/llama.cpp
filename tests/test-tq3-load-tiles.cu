// Unit test: TQ3_0 → exact q8_0 requantization
// Verifies the load_tiles_tq3_0 contract: dequant TQ3_0 → quantize to q8_0

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define QK 32
typedef struct { __half d; uint8_t qs[12]; } block_tq3_0;
typedef struct { __half d; int8_t qs[32]; } block_q8_0;

// === CPU reference ===
static float cpu_sign(int i) { return ((((unsigned)i*0x9E3779B9u)>>31)&1)?-1.0f:1.0f; }
static const float C[8]={-2.1519f,-1.3439f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3439f,2.1519f};
static const float B[7]={-1.7479f,-1.0500f,-0.5005f,0.0f,0.5005f,1.0500f,1.7479f};

static void cpu_quantize_tq3(const float *x, block_tq3_0 *blk) {
    float sq=0; for(int i=0;i<32;i++) sq+=x[i]*x[i];
    float rms=sqrtf(sq/32.0f); if(rms<1e-10f) rms=1.0f;
    blk->d=__float2half(rms);
    float buf[32]; for(int i=0;i<32;i++) buf[i]=x[i]/rms*cpu_sign(i);
    for(int s=1;s<32;s<<=1) for(int i=0;i<32;i+=s*2) for(int j=i;j<i+s;j++){float a=buf[j],b=buf[j+s];buf[j]=a+b;buf[j+s]=a-b;}
    for(int i=0;i<32;i++) buf[i]/=sqrtf(32.0f);
    uint8_t idx[32]; for(int i=0;i<32;i++){idx[i]=0;for(int b=0;b<7;b++) if(buf[i]>B[b]) idx[i]=b+1;}
    for(int g=0;g<4;g++){uint8_t*q=blk->qs+g*3,*d=idx+g*8;
        q[0]=d[0]|(d[1]<<3)|(d[2]<<6);q[1]=(d[2]>>2)|(d[3]<<1)|(d[4]<<4)|(d[5]<<7);q[2]=(d[5]>>1)|(d[6]<<2)|(d[7]<<5);}
}

static void cpu_dequant_tq3(const block_tq3_0 *blk, float *out) {
    float rms=__half2float(blk->d);
    float v[32];
    for(int g=0;g<4;g++){const uint8_t*q=blk->qs+g*3;int b=g*8;
        v[b+0]=C[q[0]&7];v[b+1]=C[(q[0]>>3)&7];v[b+2]=C[((q[0]>>6)|(q[1]<<2))&7];
        v[b+3]=C[(q[1]>>1)&7];v[b+4]=C[(q[1]>>4)&7];v[b+5]=C[((q[1]>>7)|(q[2]<<1))&7];
        v[b+6]=C[(q[2]>>2)&7];v[b+7]=C[(q[2]>>5)&7];}
    for(int s=1;s<32;s<<=1) for(int i=0;i<32;i+=s*2) for(int j=i;j<i+s;j++){float a=v[j],b=v[j+s];v[j]=a+b;v[j+s]=a-b;}
    for(int i=0;i<32;i++) out[i]=v[i]/sqrtf(32.0f)*cpu_sign(i)*rms;
}

static void cpu_quantize_q8_0(const float *x, float *d_out, int8_t *qs) {
    float amax=0; for(int i=0;i<32;i++) amax=fmaxf(amax,fabsf(x[i]));
    float d=amax/127.0f;
    float id=d>0?1.0f/d:0.0f;
    *d_out=d;
    for(int i=0;i<32;i++) qs[i]=(int8_t)roundf(x[i]*id);
}

// === GPU kernel: exact TQ3_0 → q8_0 (what load_tiles should do) ===
__constant__ float GPU_C[8]={-2.1519f,-1.3439f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3439f,2.1519f};

__device__ float gpu_sign(int i) { return ((((unsigned)i*0x9E3779B9u)>>31)&1)?-1.0f:1.0f; }

__global__ void tq3_to_q8_kernel(const block_tq3_0 *tq3, float *out_d, int8_t *out_qs) {
    int lane = threadIdx.x; // 0..31

    // 1. Unpack centroid
    const block_tq3_0 *blk = &tq3[blockIdx.x];
    float rms = __half2float(blk->d);
    int g=lane/8, r=lane%8;
    const uint8_t *qp = blk->qs + g*3;
    uint8_t idx;
    switch(r){case 0:idx=qp[0]&7;break;case 1:idx=(qp[0]>>3)&7;break;
    case 2:idx=((qp[0]>>6)|(qp[1]<<2))&7;break;case 3:idx=(qp[1]>>1)&7;break;
    case 4:idx=(qp[1]>>4)&7;break;case 5:idx=((qp[1]>>7)|(qp[2]<<1))&7;break;
    case 6:idx=(qp[2]>>2)&7;break;default:idx=(qp[2]>>5)&7;break;}

    // 2. WHT inverse
    float val = GPU_C[idx];
    for (int step=1; step<32; step<<=1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        val = (lane & step) ? (other - val) : (other + val);
    }

    // 3. Normalize + undo signs + scale
    float x = val / sqrtf(32.0f) * gpu_sign(lane) * rms;

    // 4. Warp reduce amax
    float a = fabsf(x);
    for (int m=16; m>0; m>>=1)
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, m));

    // 5. Compute scale
    float d  = __shfl_sync(0xFFFFFFFF, a / 127.0f, 0);
    float id = __shfl_sync(0xFFFFFFFF, a > 0.0f ? 127.0f / a : 0.0f, 0);

    // 6. Quantize
    int q = (int)roundf(x * id);
    q = max(-127, min(127, q));

    // 7. Write
    out_qs[blockIdx.x * 32 + lane] = (int8_t)q;
    if (lane == 0) out_d[blockIdx.x] = d;
}

int main() {
    printf("=== TQ3_0 → q8_0 requantization test ===\n\n");
    int pass=0, fail=0;

    float inputs[4][32];
    for(int b=0;b<4;b++) for(int i=0;i<32;i++)
        inputs[b][i] = sinf(b*100+i*0.3f+1.0f) * (0.5f + b*0.3f);

    // CPU: quantize to TQ3_0, dequant, requantize to q8_0
    block_tq3_0 cpu_tq3[4];
    float cpu_d[4]; int8_t cpu_qs[4][32];
    for(int b=0;b<4;b++) {
        cpu_quantize_tq3(inputs[b], &cpu_tq3[b]);
        float dq[32]; cpu_dequant_tq3(&cpu_tq3[b], dq);
        cpu_quantize_q8_0(dq, &cpu_d[b], cpu_qs[b]);
    }

    // GPU: same TQ3_0 blocks → q8_0
    block_tq3_0 *d_tq3; float *d_d; int8_t *d_qs;
    cudaMalloc(&d_tq3, 4*sizeof(block_tq3_0));
    cudaMalloc(&d_d, 4*sizeof(float));
    cudaMalloc(&d_qs, 4*32);
    cudaMemcpy(d_tq3, cpu_tq3, 4*sizeof(block_tq3_0), cudaMemcpyHostToDevice);

    tq3_to_q8_kernel<<<4, 32>>>(d_tq3, d_d, d_qs);
    cudaDeviceSynchronize();

    float gpu_d[4]; int8_t gpu_qs[4][32];
    cudaMemcpy(gpu_d, d_d, 4*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_qs, d_qs, 4*32, cudaMemcpyDeviceToHost);

    for(int b=0;b<4;b++) {
        bool d_ok = fabsf(gpu_d[b] - cpu_d[b]) < 1e-5f;
        int qs_diff = 0;
        for(int i=0;i<32;i++) if(gpu_qs[b][i] != cpu_qs[b][i]) qs_diff++;
        bool ok = d_ok && qs_diff == 0;
        printf("Block %d: d cpu=%.6f gpu=%.6f %s | qs mismatches=%d %s\n",
            b, cpu_d[b], gpu_d[b], d_ok?"OK":"FAIL", qs_diff, qs_diff==0?"OK":"FAIL");
        ok ? pass++ : fail++;
    }

    printf("\n%d passed, %d failed\n", pass, fail);
    cudaFree(d_tq3); cudaFree(d_d); cudaFree(d_qs);
    return fail>0 ? 1 : 0;
}
