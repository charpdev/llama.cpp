#include "kv-rotation.h"
#include <stdlib.h>
#include <math.h>

static uint64_t s_rng[4];

static uint64_t rng_next(void) {
    uint64_t r = s_rng[1] * 5;
    r = ((r << 7) | (r >> 57)) * 9;
    uint64_t t = s_rng[1] << 17;
    s_rng[2] ^= s_rng[0]; s_rng[3] ^= s_rng[1];
    s_rng[1] ^= s_rng[2]; s_rng[0] ^= s_rng[3];
    s_rng[2] ^= t;
    s_rng[3] = (s_rng[3] << 45) | (s_rng[3] >> 19);
    return r;
}

static void rng_seed(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s_rng[i] = z ^ (z >> 31);
    }
}

static double rng_normal(void) {
    double u1 = (double)(rng_next() >> 11) / (double)(1ULL << 53);
    double u2 = (double)(rng_next() >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

float * kv_rotation_generate(int head_dim, uint64_t seed) {
    int n = head_dim;
    float * R = (float *)malloc((size_t)n * n * sizeof(float));
    if (!R) return NULL;

    rng_seed(seed);
    for (int i = 0; i < n * n; i++) {
        R[i] = (float)rng_normal();
    }

    /* Modified Gram-Schmidt QR → orthogonal matrix */
    for (int j = 0; j < n; j++) {
        double norm = 0.0;
        for (int i = 0; i < n; i++) norm += (double)R[i*n+j] * R[i*n+j];
        norm = sqrt(norm);
        if (norm < 1e-12) norm = 1e-12;
        for (int i = 0; i < n; i++) R[i*n+j] /= (float)norm;
        for (int k = j+1; k < n; k++) {
            double dot = 0.0;
            for (int i = 0; i < n; i++) dot += (double)R[i*n+j] * R[i*n+k];
            for (int i = 0; i < n; i++) R[i*n+k] -= (float)dot * R[i*n+j];
        }
    }

    /* Transpose in-place: caller gets R^T so ggml_mul_mat(R^T, x) = R*x */
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            float tmp = R[i*n+j];
            R[i*n+j] = R[j*n+i];
            R[j*n+i] = tmp;
        }
    }

    return R;
}
