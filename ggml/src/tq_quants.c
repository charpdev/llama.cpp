/*
 * TurboQuant KV cache quantization
 *
 * Random orthogonal rotation + Q4_0 quantization for KV cache.
 * The rotation Gaussianizes the coordinate distribution, making
 * simple uniform scalar quantization near-optimal.
 *
 * Rotation matrix generated via QR decomposition of a random
 * Gaussian matrix (deterministic from seed). Stored as dense
 * head_dim x head_dim float matrix — small for typical head
 * dims (64: 16KB, 128: 64KB).
 */

#include "tq_quants.h"
#include "ggml-quants.h"
#include "ggml-common.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* rotation state */
static float * g_rotation    = NULL;  /* head_dim x head_dim, row-major */
static float * g_rotation_t  = NULL;  /* transpose (for inverse) */
static int     g_head_dim    = 0;
static float * g_rot_buf     = NULL;  /* scratch for one head */

/* simple xoshiro256** PRNG for reproducible rotation */
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
    /* splitmix64 to fill state */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s_rng[i] = z ^ (z >> 31);
    }
}

static double rng_normal(void) {
    /* Box-Muller */
    double u1 = (double)(rng_next() >> 11) / (double)(1ULL << 53);
    double u2 = (double)(rng_next() >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* In-place QR via modified Gram-Schmidt.
 * Q is n x n, stored row-major in q[]. */
static void qr_gram_schmidt(float * q, int n) {
    for (int j = 0; j < n; j++) {
        /* normalize column j */
        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            norm += (double)q[i * n + j] * q[i * n + j];
        }
        norm = sqrt(norm);
        if (norm < 1e-12) norm = 1e-12;
        for (int i = 0; i < n; i++) {
            q[i * n + j] /= (float)norm;
        }
        /* orthogonalize remaining columns against j */
        for (int k = j + 1; k < n; k++) {
            double dot = 0.0;
            for (int i = 0; i < n; i++) {
                dot += (double)q[i * n + j] * q[i * n + k];
            }
            for (int i = 0; i < n; i++) {
                q[i * n + k] -= (float)dot * q[i * n + j];
            }
        }
    }
}

void tq_init(int head_dim, uint64_t seed) {
    tq_cleanup();

    g_head_dim = head_dim;
    int n = head_dim;

    g_rotation   = (float *)malloc(n * n * sizeof(float));
    g_rotation_t = (float *)malloc(n * n * sizeof(float));
    g_rot_buf    = (float *)malloc(n * sizeof(float));

    /* fill with random Gaussian entries */
    rng_seed(seed);
    for (int i = 0; i < n * n; i++) {
        g_rotation[i] = (float)rng_normal();
    }

    /* QR decomposition -> orthogonal matrix */
    qr_gram_schmidt(g_rotation, n);

    /* precompute transpose */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            g_rotation_t[j * n + i] = g_rotation[i * n + j];
        }
    }
}

void tq_cleanup(void) {
    free(g_rotation);   g_rotation   = NULL;
    free(g_rotation_t); g_rotation_t = NULL;
    free(g_rot_buf);    g_rot_buf    = NULL;
    g_head_dim = 0;
}

float * tq_get_rotation(void) {
    return g_rotation;
}

int tq_get_head_dim(void) {
    return g_head_dim;
}

/* rotate one head: out[i] = sum_j R[i][j] * x[j] */
static void rotate_head(const float * x, float * out, const float * R, int d) {
    for (int i = 0; i < d; i++) {
        double sum = 0.0;
        for (int j = 0; j < d; j++) {
            sum += (double)R[i * d + j] * x[j];
        }
        out[i] = (float)sum;
    }
}

void quantize_row_tq4_0_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(g_rotation != NULL && "tq_init() must be called before quantize");
    assert(k % g_head_dim == 0);

    int d = g_head_dim;
    int n_heads = (int)(k / d);

    /* allocate temp for rotated vector */
    float * rotated = (float *)malloc(k * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        rotate_head(x + h * d, rotated + h * d, g_rotation, d);
    }

    /* quantize the rotated data as standard q4_0 */
    quantize_row_q4_0_ref(rotated, (block_q4_0 *)y, k);

    free(rotated);
}

void quantize_row_tq4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tq4_0_ref(x, y, k);
}

void dequantize_row_tq4_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(g_rotation_t != NULL && "tq_init() must be called before dequantize");
    assert(k % g_head_dim == 0);

    /* first dequantize as q4_0 */
    dequantize_row_q4_0((const block_q4_0 *)x, y, k);

    /* then apply inverse rotation (R^T) per head */
    int d = g_head_dim;
    int n_heads = (int)(k / d);

    for (int h = 0; h < n_heads; h++) {
        float * head = y + h * d;
        /* rotate in-place using scratch buffer */
        rotate_head(head, g_rot_buf, g_rotation_t, d);
        memcpy(head, g_rot_buf, d * sizeof(float));
    }
}
