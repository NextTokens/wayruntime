/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * kernels.c — the CPU compute engine: SIMD matmul/softmax kernels with
 * runtime dispatch, the quantized-weight matmul fast paths, flash
 * attention, the elementwise/normalization ops, and the numeric
 * self-tests, ported from the origin OS's compute backend.
 *
 * Layout of this file:
 *   1. small tensor helpers
 *   2. F32 matmul kernels (scalar / AVX2 / AVX-512 / NEON), in native
 *      row-major and transposed-weight ("GGML dot") flavors
 *   3. softmax kernels (scalar + vectorized exp)
 *   4. SIMD dispatch: atomic publish, acquire snapshots, counters
 *   5. worker-pool plumbing (M/N-split for F32, N-split for quantized)
 *   6. the wri_op_* implementations
 *   7. golden self-tests (driven by test/unit_tests.c)
 *
 * Threading model: all parallelism goes through wr_pool_run (pool.h).
 * The pool contract guarantees each part runs whole on one thread and
 * that at most one job is in flight, which is what the bit-exactness
 * of the quantized N-split rests on — every output column's complete
 * k-reduction happens inside one part, so the float summation order is
 * identical for any worker count.  The origin OS needed two bespoke
 * threading mechanisms (transient kernel threads + a spin/park/steal
 * pool) to get the same guarantee around its scheduler's pathologies;
 * none of that apparatus is ported, only the partitioning math.
 *
 * Scalar math: the wri_* functions from mathx.c only — never libm
 * (mathx.c owns the approx/libm switch).
 */

#include <stdlib.h>
#include <string.h>

#include "core/internal.h"
#include "core/quant.h"

#include "core/mathx.h"

/* --------------------------------------------------------------------------
 * 1. Tensor helpers
 * -------------------------------------------------------------------------- */

static uint64_t tensor_elems(const wr_tensor *t)
{
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->ndim; i++)
        n *= t->shape[i];
    return n;
}

static inline const float *cf32p(const wr_tensor *t)
{
    return (const float *)t->data;
}

static inline float *f32p(wr_tensor *t)
{
    return (float *)t->data;
}

/* Block geometry for the block-quantized dtypes (0 = not block-quantized).
 * The byte counts are the pinned constants from internal.h. */
static uint32_t blk_elems_of(wr_dtype d)
{
    switch (d) {
    case WR_DTYPE_Q4_0:
    case WR_DTYPE_Q8_0: return 32;
    case WR_DTYPE_Q4_K:
    case WR_DTYPE_Q5_K:
    case WR_DTYPE_Q6_K: return WR_QK_K;
    default:            return 0;
    }
}

static uint32_t blk_bytes_of(wr_dtype d)
{
    switch (d) {
    case WR_DTYPE_Q4_0: return WR_Q4_0_BLOCK_BYTES;
    case WR_DTYPE_Q8_0: return WR_Q8_0_BLOCK_BYTES;
    case WR_DTYPE_Q4_K: return WR_Q4_K_BLOCK_BYTES;
    case WR_DTYPE_Q5_K: return WR_Q5_K_BLOCK_BYTES;
    case WR_DTYPE_Q6_K: return WR_Q6_K_BLOCK_BYTES;
    default:            return 0;
    }
}

/* --------------------------------------------------------------------------
 * 2. F32 matmul kernels
 *
 * Two data layouts, three ISA tiers each:
 *   - native row-major:  C[i,j] = sum_k A[i*K+k] * B[k*N+j]
 *   - transposed weight ("GGML dot"): B is W[out][in] stored at j*K+k
 *     (`in` contiguous), so C[i,j] = dot of two CONTIGUOUS rows — the
 *     ideal SIMD case, and the layout every production LLM projection
 *     weight uses (WR_TENSOR_GGML_WEIGHT).
 *
 * The vector kernels use GCC generic vector extensions with per-function
 * target attributes; the compiler emits FMA on its own.
 *
 * ALIGNMENT LANDMINE — do not "clean up" the aligned(4) below.  The
 * vector typedefs deliberately declare 4-byte alignment so every vector
 * load/store is emitted as VMOVUPS (unaligned).  Pointers into plain
 * float arrays are only 4-byte aligned; an aligned VMOVAPS on data that
 * is not actually 32/64-byte aligned raises #GP.  The self-test below
 * runs the dispatched kernel on a deliberately misaligned buffer to
 * guard this exact property.
 * -------------------------------------------------------------------------- */

static void matmul_f32_scalar(const float *a, const float *b, float *c,
                              uint32_t M, uint32_t K, uint32_t N)
{
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0;
            for (uint32_t k = 0; k < K; k++)
                sum += a[(size_t)i * K + k] * b[(size_t)k * N + j];
            c[(size_t)i * N + j] = sum;
        }
    }
}

static void matmul_f32_ggml_scalar(const float *a, const float *b, float *c,
                                   uint32_t M, uint32_t K, uint32_t N)
{
    for (uint32_t i = 0; i < M; i++) {
        const float *arow = a + (size_t)i * K;
        for (uint32_t j = 0; j < N; j++) {
            const float *brow = b + (size_t)j * K;   /* W[out][in] at j*K+k */
            float sum = 0;
            for (uint32_t k = 0; k < K; k++)
                sum += arow[k] * brow[k];
            c[(size_t)i * N + j] = sum;
        }
    }
}

#if defined(__x86_64__)

__attribute__((target("avx2,fma")))
static void matmul_f32_avx2(const float *a, const float *b, float *c,
                            uint32_t M, uint32_t K, uint32_t N)
{
    /* aligned(4): force VMOVUPS — see the alignment note above. */
    typedef float v8sf __attribute__((vector_size(32), aligned(4)));

    for (uint32_t i = 0; i < M; i++) {
        uint32_t j = 0;
        /* Interleave four output vectors: each accumulator's dependency
         * chain is separated by three independent FMAs, hiding FMA
         * latency while one activation broadcast feeds all four. */
        for (; j + 32 <= N; j += 32) {
            v8sf acc0 = { 0 }, acc1 = { 0 }, acc2 = { 0 }, acc3 = { 0 };
            for (uint32_t k = 0; k < K; k++) {
                float scalar = a[(size_t)i * K + k];
                v8sf av = { scalar, scalar, scalar, scalar,
                            scalar, scalar, scalar, scalar };
                const float *brow = b + (size_t)k * N + j;
                acc0 += av * *(const v8sf *)&brow[0];
                acc1 += av * *(const v8sf *)&brow[8];
                acc2 += av * *(const v8sf *)&brow[16];
                acc3 += av * *(const v8sf *)&brow[24];
            }
            float *crow = c + (size_t)i * N + j;
            *(v8sf *)&crow[0] = acc0;
            *(v8sf *)&crow[8] = acc1;
            *(v8sf *)&crow[16] = acc2;
            *(v8sf *)&crow[24] = acc3;
        }
        for (; j + 16 <= N; j += 16) {
            v8sf acc0 = { 0 }, acc1 = { 0 };
            for (uint32_t k = 0; k < K; k++) {
                float scalar = a[(size_t)i * K + k];
                v8sf av = { scalar, scalar, scalar, scalar,
                            scalar, scalar, scalar, scalar };
                const float *brow = b + (size_t)k * N + j;
                acc0 += av * *(const v8sf *)&brow[0];
                acc1 += av * *(const v8sf *)&brow[8];
            }
            float *crow = c + (size_t)i * N + j;
            *(v8sf *)&crow[0] = acc0;
            *(v8sf *)&crow[8] = acc1;
        }
        for (; j + 8 <= N; j += 8) {
            v8sf acc = { 0, 0, 0, 0, 0, 0, 0, 0 };
            for (uint32_t k = 0; k < K; k++) {
                float scalar = a[(size_t)i * K + k];
                v8sf av = { scalar, scalar, scalar, scalar,
                            scalar, scalar, scalar, scalar };
                v8sf bv = *(const v8sf *)&b[(size_t)k * N + j];
                acc += av * bv;          /* VFMADD231PS */
            }
            *(v8sf *)&c[(size_t)i * N + j] = acc;
        }
        for (; j < N; j++) {
            float sum = 0;
            for (uint32_t k = 0; k < K; k++)
                sum += a[(size_t)i * K + k] * b[(size_t)k * N + j];
            c[(size_t)i * N + j] = sum;
        }
    }
}

__attribute__((target("avx512f,avx512dq,fma")))
static void matmul_f32_avx512(const float *a, const float *b, float *c,
                              uint32_t M, uint32_t K, uint32_t N)
{
    typedef float v16sf __attribute__((vector_size(64), aligned(4)));

    for (uint32_t i = 0; i < M; i++) {
        uint32_t j = 0;
        /* Four independent ZMM accumulator chains; the activation
         * broadcast is reused across 64 output columns. */
        for (; j + 64 <= N; j += 64) {
            v16sf acc0 = { 0 }, acc1 = { 0 }, acc2 = { 0 }, acc3 = { 0 };
            for (uint32_t k = 0; k < K; k++) {
                float scalar = a[(size_t)i * K + k];
                v16sf av = { scalar, scalar, scalar, scalar,
                             scalar, scalar, scalar, scalar,
                             scalar, scalar, scalar, scalar,
                             scalar, scalar, scalar, scalar };
                const float *brow = b + (size_t)k * N + j;
                acc0 += av * *(const v16sf *)&brow[0];
                acc1 += av * *(const v16sf *)&brow[16];
                acc2 += av * *(const v16sf *)&brow[32];
                acc3 += av * *(const v16sf *)&brow[48];
            }
            float *crow = c + (size_t)i * N + j;
            *(v16sf *)&crow[0] = acc0;
            *(v16sf *)&crow[16] = acc1;
            *(v16sf *)&crow[32] = acc2;
            *(v16sf *)&crow[48] = acc3;
        }
        for (; j + 32 <= N; j += 32) {
            v16sf acc0 = { 0 }, acc1 = { 0 };
            for (uint32_t k = 0; k < K; k++) {
                float scalar = a[(size_t)i * K + k];
                v16sf av = { scalar, scalar, scalar, scalar,
                             scalar, scalar, scalar, scalar,
                             scalar, scalar, scalar, scalar,
                             scalar, scalar, scalar, scalar };
                const float *brow = b + (size_t)k * N + j;
                acc0 += av * *(const v16sf *)&brow[0];
                acc1 += av * *(const v16sf *)&brow[16];
            }
            float *crow = c + (size_t)i * N + j;
            *(v16sf *)&crow[0] = acc0;
            *(v16sf *)&crow[16] = acc1;
        }
        for (; j + 16 <= N; j += 16) {
            v16sf acc = { 0 };
            for (uint32_t k = 0; k < K; k++) {
                float scalar = a[(size_t)i * K + k];
                v16sf av = { scalar, scalar, scalar, scalar,
                             scalar, scalar, scalar, scalar,
                             scalar, scalar, scalar, scalar,
                             scalar, scalar, scalar, scalar };
                v16sf bv = *(const v16sf *)&b[(size_t)k * N + j];
                acc += av * bv;
            }
            *(v16sf *)&c[(size_t)i * N + j] = acc;
        }
        for (; j < N; j++) {
            float sum = 0;
            for (uint32_t k = 0; k < K; k++)
                sum += a[(size_t)i * K + k] * b[(size_t)k * N + j];
            c[(size_t)i * N + j] = sum;
        }
    }
}

__attribute__((target("avx2,fma")))
static void matmul_f32_ggml_avx2(const float *a, const float *b, float *c,
                                 uint32_t M, uint32_t K, uint32_t N)
{
    typedef float v8sf __attribute__((vector_size(32), aligned(4)));
    for (uint32_t i = 0; i < M; i++) {
        const float *arow = a + (size_t)i * K;
        for (uint32_t j = 0; j < N; j++) {
            const float *brow = b + (size_t)j * K;
            /* Production decode calls this with M=N=1.  Strip-mine the
             * dot into four chains so this consumer, not only the
             * row-major shape, hides FMA latency. */
            v8sf acc0 = { 0 }, acc1 = { 0 }, acc2 = { 0 }, acc3 = { 0 };
            uint32_t k = 0;
            for (; k + 32 <= K; k += 32) {
                acc0 += (*(const v8sf *)&arow[k]) *
                        (*(const v8sf *)&brow[k]);
                acc1 += (*(const v8sf *)&arow[k + 8]) *
                        (*(const v8sf *)&brow[k + 8]);
                acc2 += (*(const v8sf *)&arow[k + 16]) *
                        (*(const v8sf *)&brow[k + 16]);
                acc3 += (*(const v8sf *)&arow[k + 24]) *
                        (*(const v8sf *)&brow[k + 24]);
            }
            v8sf acc = (acc0 + acc1) + (acc2 + acc3);
            for (; k + 8 <= K; k += 8)
                acc += (*(const v8sf *)&arow[k]) *
                       (*(const v8sf *)&brow[k]);
            float sum = acc[0] + acc[1] + acc[2] + acc[3] +
                        acc[4] + acc[5] + acc[6] + acc[7];
            for (; k < K; k++) sum += arow[k] * brow[k];
            c[(size_t)i * N + j] = sum;
        }
    }
}

__attribute__((target("avx512f,avx512dq,fma")))
static void matmul_f32_ggml_avx512(const float *a, const float *b, float *c,
                                   uint32_t M, uint32_t K, uint32_t N)
{
    typedef float v16sf __attribute__((vector_size(64), aligned(4)));
    for (uint32_t i = 0; i < M; i++) {
        const float *arow = a + (size_t)i * K;
        for (uint32_t j = 0; j < N; j++) {
            const float *brow = b + (size_t)j * K;
            v16sf acc0 = { 0 }, acc1 = { 0 }, acc2 = { 0 }, acc3 = { 0 };
            uint32_t k = 0;
            for (; k + 64 <= K; k += 64) {
                acc0 += (*(const v16sf *)&arow[k]) *
                        (*(const v16sf *)&brow[k]);
                acc1 += (*(const v16sf *)&arow[k + 16]) *
                        (*(const v16sf *)&brow[k + 16]);
                acc2 += (*(const v16sf *)&arow[k + 32]) *
                        (*(const v16sf *)&brow[k + 32]);
                acc3 += (*(const v16sf *)&arow[k + 48]) *
                        (*(const v16sf *)&brow[k + 48]);
            }
            v16sf acc = (acc0 + acc1) + (acc2 + acc3);
            for (; k + 16 <= K; k += 16)
                acc += (*(const v16sf *)&arow[k]) *
                       (*(const v16sf *)&brow[k]);
            float sum = 0;
            for (int l = 0; l < 16; l++) sum += acc[l];
            for (; k < K; k++) sum += arow[k] * brow[k];
            c[(size_t)i * N + j] = sum;
        }
    }
}

#endif /* __x86_64__ */

#if defined(__aarch64__)

static void matmul_f32_neon(const float *a, const float *b, float *c,
                            uint32_t M, uint32_t K, uint32_t N)
{
    /* NEON: 4 floats per FMA.  aligned(4) as everywhere. */
    typedef float v4sf __attribute__((vector_size(16), aligned(4)));
    for (uint32_t i = 0; i < M; i++) {
        uint32_t j = 0;
        for (; j + 4 <= N; j += 4) {
            v4sf acc = { 0, 0, 0, 0 };
            for (uint32_t k = 0; k < K; k++) {
                float scalar = a[(size_t)i * K + k];
                v4sf av = { scalar, scalar, scalar, scalar };
                v4sf bv = *(const v4sf *)&b[(size_t)k * N + j];
                acc += av * bv;
            }
            *(v4sf *)&c[(size_t)i * N + j] = acc;
        }
        for (; j < N; j++) {
            float sum = 0;
            for (uint32_t k = 0; k < K; k++)
                sum += a[(size_t)i * K + k] * b[(size_t)k * N + j];
            c[(size_t)i * N + j] = sum;
        }
    }
}

static void matmul_f32_ggml_neon(const float *a, const float *b, float *c,
                                 uint32_t M, uint32_t K, uint32_t N)
{
    typedef float v4sf __attribute__((vector_size(16), aligned(4)));
    for (uint32_t i = 0; i < M; i++) {
        const float *arow = a + (size_t)i * K;
        for (uint32_t j = 0; j < N; j++) {
            const float *brow = b + (size_t)j * K;
            v4sf acc = { 0, 0, 0, 0 };
            uint32_t k = 0;
            for (; k + 4 <= K; k += 4)
                acc += (*(const v4sf *)&arow[k]) * (*(const v4sf *)&brow[k]);
            float sum = acc[0] + acc[1] + acc[2] + acc[3];
            for (; k < K; k++) sum += arow[k] * brow[k];
            c[(size_t)i * N + j] = sum;
        }
    }
}

#endif /* __aarch64__ */

/* --------------------------------------------------------------------------
 * 3. Softmax kernels
 *
 * Three passes over one row: max reduction, shifted exp + sum,
 * normalize by 1/sum.  The vector kernels use range reduction by ln2
 * (x = k*ln2 + r) with a degree-5 Taylor polynomial on the residual,
 * then recover 2^k via an integer add to the IEEE-754 exponent bits —
 * the same algorithm as the scalar wri_exp, so scalar and SIMD agree
 * within the polynomial's tolerance.
 * -------------------------------------------------------------------------- */

static void softmax_f32_scalar(const float *in, float *out, uint64_t n)
{
    if (n == 0) return;
    float max_val = in[0];
    for (uint64_t i = 1; i < n; i++) if (in[i] > max_val) max_val = in[i];

    float sum = 0;
    for (uint64_t i = 0; i < n; i++) {
        float v = wri_exp(in[i] - max_val);
        out[i] = v;
        sum += v;
    }
    if (sum > 0) {
        float inv_sum = 1.0f / sum;
        for (uint64_t i = 0; i < n; i++) out[i] *= inv_sum;
    }
}

#if defined(__x86_64__)

__attribute__((target("avx2,fma")))
static inline void v8sf_exp_clamped(const float *xin, float *yout)
{
    /* 8 lanes at a time; inputs are already clamped to [-88, 88] by the
     * caller (softmax subtracts the max first). */
    typedef float    v8sf __attribute__((vector_size(32)));
    typedef int32_t  v8si __attribute__((vector_size(32)));

    v8sf x = *(const v8sf *)xin;

    const v8sf log2e = { 1.44269504f, 1.44269504f, 1.44269504f, 1.44269504f,
                         1.44269504f, 1.44269504f, 1.44269504f, 1.44269504f };
    const v8sf ln2 = { 0.69314718f, 0.69314718f, 0.69314718f, 0.69314718f,
                       0.69314718f, 0.69314718f, 0.69314718f, 0.69314718f };

    /* k = round(x * log2e); r = x - k*ln2 */
    v8sf z = x * log2e;
    v8si k = __builtin_ia32_cvtps2dq256(z);   /* round-to-nearest */
    v8sf kf;
    v8si kk = k;
    for (int i = 0; i < 8; i++) ((float *)&kf)[i] = (float)kk[i];
    v8sf r = x - kf * ln2;

    /* Degree-5 Taylor at 0, Horner form. */
    v8sf c5 = { 1.0f/120, 1.0f/120, 1.0f/120, 1.0f/120,
                1.0f/120, 1.0f/120, 1.0f/120, 1.0f/120 };
    v8sf c4 = { 1.0f/24,  1.0f/24,  1.0f/24,  1.0f/24,
                1.0f/24,  1.0f/24,  1.0f/24,  1.0f/24  };
    v8sf c3 = { 1.0f/6,   1.0f/6,   1.0f/6,   1.0f/6,
                1.0f/6,   1.0f/6,   1.0f/6,   1.0f/6   };
    v8sf c2 = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
    v8sf one = { 1, 1, 1, 1, 1, 1, 1, 1 };
    v8sf e = r * c5 + c4;
    e = r * e + c3;
    e = r * e + c2;
    e = r * e + one;
    e = r * e + one;

    /* Multiply by 2^k via exponent bit-add. */
    union { v8sf f; v8si i; } u;
    u.f = e;
    v8si shifted = k << 23;
    u.i = u.i + shifted;

    *(v8sf *)yout = u.f;
}

__attribute__((target("avx2,fma")))
static void softmax_f32_avx2(const float *in, float *out, uint64_t n)
{
    if (n == 0) return;

    float max_val = in[0];
    for (uint64_t i = 1; i < n; i++) if (in[i] > max_val) max_val = in[i];

    float sum = 0;
    uint64_t i = 0;
    float tmp_in[8] __attribute__((aligned(32)));
    float tmp_out[8] __attribute__((aligned(32)));
    for (; i + 8 <= n; i += 8) {
        for (int w = 0; w < 8; w++) {
            float v = in[i + w] - max_val;
            if (v < -88.0f) v = -88.0f;
            tmp_in[w] = v;
        }
        v8sf_exp_clamped(tmp_in, tmp_out);
        for (int w = 0; w < 8; w++) {
            out[i + w] = tmp_out[w];
            sum += tmp_out[w];
        }
    }
    for (; i < n; i++) {
        float v = wri_exp(in[i] - max_val);
        out[i] = v;
        sum += v;
    }

    if (sum > 0) {
        float inv_sum = 1.0f / sum;
        for (uint64_t j = 0; j < n; j++) out[j] *= inv_sum;
    }
}

__attribute__((target("avx512f,avx512dq,fma")))
static void softmax_f32_avx512(const float *in, float *out, uint64_t n)
{
    /* DELIBERATE: this is not a true 16-wide AVX-512 softmax — it
     * delegates to the AVX2 kernel.  A real 16-wide exp polynomial was
     * measured a no-win at this op's scale in the origin engine, so the
     * body is shared on purpose; the name reflects the dispatch tier,
     * not a distinct implementation.  Keep the decision. */
    softmax_f32_avx2(in, out, n);
}

#endif /* __x86_64__ */

#if defined(__aarch64__)

/* NEON 4-wide vector exp — same range-reduced polynomial as the AVX2
 * variant so the accuracy contract is identical across architectures. */
static inline void v4sf_exp_clamped(const float *xin, float *yout)
{
    typedef float    v4sf __attribute__((vector_size(16)));
    typedef int32_t  v4si __attribute__((vector_size(16)));

    v4sf x = *(const v4sf *)xin;

    const v4sf log2e = { 1.44269504f, 1.44269504f, 1.44269504f, 1.44269504f };
    const v4sf ln2   = { 0.69314718f, 0.69314718f, 0.69314718f, 0.69314718f };

    v4sf z = x * log2e;
    v4si k;
    v4sf kf;
    for (int i = 0; i < 4; i++) {
        float zf = z[i];
        /* Manual rounding: robust regardless of the host rounding mode. */
        int32_t ki = (int32_t)(zf + (zf >= 0 ? 0.5f : -0.5f));
        k[i]  = ki;
        kf[i] = (float)ki;
    }
    v4sf r = x - kf * ln2;

    v4sf c5 = { 1.0f/120, 1.0f/120, 1.0f/120, 1.0f/120 };
    v4sf c4 = { 1.0f/24,  1.0f/24,  1.0f/24,  1.0f/24  };
    v4sf c3 = { 1.0f/6,   1.0f/6,   1.0f/6,   1.0f/6   };
    v4sf c2 = { 0.5f, 0.5f, 0.5f, 0.5f };
    v4sf one = { 1, 1, 1, 1 };
    v4sf e = r * c5 + c4;
    e = r * e + c3;
    e = r * e + c2;
    e = r * e + one;
    e = r * e + one;

    union { v4sf f; v4si i; } u;
    u.f = e;
    v4si shifted = k << 23;
    u.i = u.i + shifted;

    *(v4sf *)yout = u.f;
}

static void softmax_f32_neon(const float *in, float *out, uint64_t n)
{
    if (n == 0) return;

    float max_val = in[0];
    for (uint64_t i = 1; i < n; i++) if (in[i] > max_val) max_val = in[i];

    float sum = 0;
    uint64_t i = 0;
    float tmp_in[4]  __attribute__((aligned(16)));
    float tmp_out[4] __attribute__((aligned(16)));
    for (; i + 4 <= n; i += 4) {
        for (int w = 0; w < 4; w++) {
            float v = in[i + w] - max_val;
            if (v < -88.0f) v = -88.0f;
            tmp_in[w] = v;
        }
        v4sf_exp_clamped(tmp_in, tmp_out);
        for (int w = 0; w < 4; w++) {
            out[i + w] = tmp_out[w];
            sum += tmp_out[w];
        }
    }
    for (; i < n; i++) {
        float v = wri_exp(in[i] - max_val);
        out[i] = v;
        sum += v;
    }

    if (sum > 0) {
        float inv_sum = 1.0f / sum;
        for (uint64_t j = 0; j < n; j++) out[j] *= inv_sum;
    }
}

#endif /* __aarch64__ */

/* --------------------------------------------------------------------------
 * 4. SIMD dispatch
 *
 * wri_simd_init is a runtime RE-selector, not a boot-only init: the
 * replacement table is built off to the side and then published with
 * release stores; every op takes one acquire snapshot of the function
 * pointer per call.  A re-selection while other threads decode is
 * therefore safe — an in-flight op completes on the variant it
 * snapshotted, and one reduction is never split across two kernels.
 * -------------------------------------------------------------------------- */

typedef void (*wri_matmul_fn)(const float *, const float *, float *,
                              uint32_t, uint32_t, uint32_t);
typedef void (*wri_softmax_fn)(const float *, float *, uint64_t);

static wri_matmul_fn  g_matmul_impl      = matmul_f32_scalar;
static wri_matmul_fn  g_matmul_ggml_impl = matmul_f32_ggml_scalar;
static wri_softmax_fn g_softmax_impl     = softmax_f32_scalar;
static int g_matmul_variant      = WR_SIMD_SCALAR;
static int g_matmul_ggml_variant = WR_SIMD_SCALAR;
static int g_softmax_variant     = WR_SIMD_SCALAR;

static wri_matmul_fn matmul_kernel_snapshot(void)
{
    return __atomic_load_n(&g_matmul_impl, __ATOMIC_ACQUIRE);
}

static wri_matmul_fn matmul_ggml_kernel_snapshot(void)
{
    return __atomic_load_n(&g_matmul_ggml_impl, __ATOMIC_ACQUIRE);
}

static wri_softmax_fn softmax_kernel_snapshot(void)
{
    return __atomic_load_n(&g_softmax_impl, __ATOMIC_ACQUIRE);
}

void wri_simd_init(int force_scalar, int prefer_avx2)
{
    wri_matmul_fn  native = matmul_f32_scalar;
    wri_matmul_fn  ggml   = matmul_f32_ggml_scalar;
    wri_softmax_fn smax   = softmax_f32_scalar;
    int mv = WR_SIMD_SCALAR;
    int sv = WR_SIMD_SCALAR;

    (void)prefer_avx2;   /* meaningful on x86-64 hosts only */

    if (!force_scalar) {
        uint32_t feats = wr_cpu_features();
        (void)feats;
#if defined(__x86_64__)
        /* prefer_avx2 exists because AVX-512 license downclocking was
         * measured SLOWER than AVX2 for small-model decode on the origin
         * hardware; it demotes only the matmul tier choice. */
        if ((feats & WR_CPU_FEAT_AVX512) && !prefer_avx2) {
            native = matmul_f32_avx512;
            ggml   = matmul_f32_ggml_avx512;
            mv     = WR_SIMD_AVX512;
            smax   = softmax_f32_avx512;   /* AVX2 body by design */
            sv     = WR_SIMD_AVX512;
        } else if (feats & WR_CPU_FEAT_AVX2) {
            native = matmul_f32_avx2;
            ggml   = matmul_f32_ggml_avx2;
            mv     = WR_SIMD_AVX2;
            smax   = softmax_f32_avx2;
            sv     = WR_SIMD_AVX2;
        }
#elif defined(__aarch64__)
        if (feats & WR_CPU_FEAT_NEON) {
            native = matmul_f32_neon;
            ggml   = matmul_f32_ggml_neon;
            mv     = WR_SIMD_NEON;
            smax   = softmax_f32_neon;
            sv     = WR_SIMD_NEON;
        }
#endif
    }

    /* Publish: operation tables first, user-visible variants last, all
     * with release stores.  A reader that observes the new variant is
     * then guaranteed to observe the new tables too. */
    __atomic_store_n(&g_matmul_impl, native, __ATOMIC_RELEASE);
    __atomic_store_n(&g_matmul_ggml_impl, ggml, __ATOMIC_RELEASE);
    __atomic_store_n(&g_softmax_impl, smax, __ATOMIC_RELEASE);
    __atomic_store_n(&g_matmul_ggml_variant, mv, __ATOMIC_RELEASE);
    __atomic_store_n(&g_softmax_variant, sv, __ATOMIC_RELEASE);
    __atomic_store_n(&g_matmul_variant, mv, __ATOMIC_RELEASE);
}

int wri_simd_matmul_variant(void)
{
    return __atomic_load_n(&g_matmul_variant, __ATOMIC_ACQUIRE);
}

int wri_simd_softmax_variant(void)
{
    return __atomic_load_n(&g_softmax_variant, __ATOMIC_ACQUIRE);
}

/* Counters.  Accumulator slots are bumped from the ops via WRI_CTR_ADD;
 * the variant slots are materialized here at snapshot time. */
uint64_t wri_g_counters[WR_COUNTER_COUNT];

void wri_counters_reset(void)
{
    for (int i = 0; i < WR_COUNTER_COUNT; i++)
        __atomic_store_n(&wri_g_counters[i], (uint64_t)0, __ATOMIC_RELAXED);
}

void wri_counters_snapshot(uint64_t out[WR_COUNTER_COUNT])
{
    for (int i = 0; i < WR_COUNTER_COUNT; i++)
        out[i] = __atomic_load_n(&wri_g_counters[i], __ATOMIC_RELAXED);
    out[WR_CTR_MATMUL_VARIANT] =
        (uint64_t)__atomic_load_n(&g_matmul_variant, __ATOMIC_ACQUIRE);
    out[WR_CTR_SOFTMAX_VARIANT] =
        (uint64_t)__atomic_load_n(&g_softmax_variant, __ATOMIC_ACQUIRE);
    out[WR_CTR_MATMUL_GGML_VARIANT] =
        (uint64_t)__atomic_load_n(&g_matmul_ggml_variant, __ATOMIC_ACQUIRE);
}

/* --------------------------------------------------------------------------
 * 5. Worker-pool plumbing
 * -------------------------------------------------------------------------- */

static wr_pool *engine_pool(void)
{
    return wri_g_engine ? wri_g_engine->pool : NULL;
}

/* ---- M-split (F32 × F32 matmul, both layouts) -------------------------
 * Split the output rows into one disjoint block per part; every part
 * runs the SAME kernel over full rows, so the result is bit-identical
 * to the serial call regardless of the part count.  Partition math:
 * part p gets base = M/parts rows plus one of the rem = M%parts
 * leftovers, in order. */

typedef struct {
    wri_matmul_fn kernel;
    const float  *a, *b;
    float        *c;
    uint32_t      M, K, N;
} mm_split_job;

static void mm_split_part(void *arg, uint32_t part, uint32_t n_parts)
{
    const mm_split_job *j = (const mm_split_job *)arg;
    uint32_t base  = j->M / n_parts;
    uint32_t rem   = j->M % n_parts;
    uint32_t start = part * base + (part < rem ? part : rem);
    uint32_t cnt   = base + (part < rem ? 1u : 0u);
    if (cnt == 0)
        return;
    j->kernel(j->a + (size_t)start * j->K, j->b,
              j->c + (size_t)start * j->N, cnt, j->K, j->N);
}

static void matmul_f32_run(wri_matmul_fn kernel, const float *a,
                           const float *b, float *c,
                           uint32_t M, uint32_t K, uint32_t N)
{
    wr_pool *pool = engine_pool();
    uint32_t workers = pool ? wr_pool_size(pool) : 1;
    if (M < WR_PARALLEL_THRESHOLD || workers <= 1) {
        kernel(a, b, c, M, K, N);
        return;
    }
    mm_split_job job = { kernel, a, b, c, M, K, N };
    wr_pool_run(pool, mm_split_part, &job, workers);
}

/* ---- N-split (F32 GGML weights, decode M == 1) -----------------------
 * A one-token projection has no M rows to distribute, but a large GGML
 * weight (notably a dequantized tied LM head) has many independent output
 * columns.  Split those contiguous W[out][in] rows across the pool.  Each
 * complete K-dot still runs through the exact same snapshotted kernel, so
 * its reduction order and result bits are identical to the serial call.
 *
 * This specialization is deliberately M == 1.  For M > 1, a sub-call's
 * compact output stride would differ from the parent C stride; the normal
 * M-split above already handles large-M work without that complication. */

typedef struct {
    wri_matmul_fn kernel;
    const float  *a, *b;
    float        *c;
    uint32_t      K, N;
} mm_ggml_n_job;

static void mm_ggml_n_part(void *arg, uint32_t part, uint32_t n_parts)
{
    const mm_ggml_n_job *j = (const mm_ggml_n_job *)arg;
    uint32_t base  = j->N / n_parts;
    uint32_t rem   = j->N % n_parts;
    uint32_t start = part * base + (part < rem ? part : rem);
    uint32_t cnt   = base + (part < rem ? 1u : 0u);
    if (cnt == 0)
        return;
    j->kernel(j->a, j->b + (size_t)start * j->K, j->c + start,
              1, j->K, cnt);
}

static void matmul_f32_ggml_run(wri_matmul_fn kernel, const float *a,
                                const float *b, float *c,
                                uint32_t M, uint32_t K, uint32_t N)
{
    wr_pool *pool = engine_pool();
    uint32_t workers = pool ? wr_pool_size(pool) : 1;
    if (M != 1 || N < WR_PARALLEL_THRESHOLD || workers <= 1) {
        matmul_f32_run(kernel, a, b, c, M, K, N);
        return;
    }
    if (workers > N)
        workers = N;
    mm_ggml_n_job job = { kernel, a, b, c, K, N };
    WRI_CTR_ADD(WR_CTR_MATMUL_PAR_N, 1);
    wr_pool_run(pool, mm_ggml_n_part, &job, workers);
}

/* ---- N-split (kept-quantized GGML weights) ----------------------------
 * Decode one weight row (output column j) into a K-float scratch via
 * the quant.h row decoder, run the GGML dot kernel with N=1 across all
 * M activation rows, copy the column out.  Decode cost is N*K (per
 * weight row) instead of the per-element path's M*N*K, and the dot is
 * vectorized.  Parallelization splits the OUTPUT columns: each dot's
 * complete k-reduction stays inside one part, which makes the result
 * BIT-EXACT versus serial for any worker count (pool.h rule 2).
 *
 * The static scratch below is indexed by PART and only ever touched
 * inside wr_pool_run, whose single-dispatcher rule (pool.h rule 5)
 * serializes concurrent callers — the origin's hand-rolled busy flag
 * is deliberately NOT re-added. */

static float g_qmm_wrow[WR_MAX_WORKERS][WR_QMM_K_MAX];
static float g_qmm_col [WR_MAX_WORKERS][WR_QMM_M_MAX];

typedef void (*wri_row_decode_fn)(const void *, float *, uint64_t);

typedef struct {
    const float   *a;
    const uint8_t *b_base;
    float         *c;
    uint32_t       M, K, N;
    uint32_t       row_bytes;
    wri_row_decode_fn row_decode;
    wri_matmul_fn  kernel;             /* snapshotted ONCE per matmul */
    uint32_t       bnd[WR_MAX_WORKERS + 1];
} qmm_job;

static void qmm_chunk(const qmm_job *g, uint32_t j0, uint32_t j1,
                      float *wrow, float *col)
{
    for (uint32_t j = j0; j < j1; j++) {
        const uint8_t *wj = g->b_base + (size_t)j * g->row_bytes;
        g->row_decode(wj, wrow, g->K);
        g->kernel(g->a, wrow, col, g->M, g->K, 1);
        for (uint32_t i = 0; i < g->M; i++)
            g->c[(size_t)i * g->N + j] = col[i];   /* disjoint j per part */
    }
}

static void qmm_part(void *arg, uint32_t part, uint32_t n_parts)
{
    (void)n_parts;
    const qmm_job *g = (const qmm_job *)arg;
    qmm_chunk(g, g->bnd[part], g->bnd[part + 1],
              g_qmm_wrow[part], g_qmm_col[part]);
}

/* Returns WR_OK, or WR_ERR_NOMEM when no pool exists and heap scratch
 * could not be allocated (the caller then falls back to the per-element
 * path — slower, never wrong). */
static int qmm_ggml_matmul(const float *a, const uint8_t *b_base, float *c,
                           uint32_t M, uint32_t K, uint32_t N,
                           uint32_t blk_bytes, uint32_t blk_elems,
                           wri_row_decode_fn row_decode)
{
    qmm_job job;
    job.a          = a;
    job.b_base     = b_base;
    job.c          = c;
    job.M          = M;
    job.K          = K;
    job.N          = N;
    job.row_bytes  = (K / blk_elems) * blk_bytes;
    job.row_decode = row_decode;
    job.kernel     = matmul_ggml_kernel_snapshot();

    wr_pool *pool = engine_pool();
    if (pool) {
        uint32_t parts = 1;
        if (N >= WR_PARALLEL_THRESHOLD) {
            parts = wr_pool_size(pool);
            if (parts > WR_MAX_WORKERS) parts = WR_MAX_WORKERS;
            if (parts > N) parts = N;
        }
        uint32_t base = N / parts, rem = N % parts, acc = 0;
        for (uint32_t p = 0; p < parts; p++) {
            job.bnd[p] = acc;
            acc += base + (p < rem ? 1u : 0u);
        }
        job.bnd[parts] = N;
        if (parts > 1)
            WRI_CTR_ADD(WR_CTR_MATMUL_PAR_N, 1);
        wr_pool_run(pool, qmm_part, &job, parts);
        return WR_OK;
    }

    /* No engine yet (self-test or pre-create use): private heap
     * scratch, serial. */
    float *wrow = (float *)malloc((size_t)K * sizeof(float));
    float *col  = (float *)malloc((size_t)M * sizeof(float));
    if (!wrow || !col) {
        free(wrow);
        free(col);
        return WR_ERR_NOMEM;
    }
    qmm_chunk(&job, 0, N, wrow, col);
    free(wrow);
    free(col);
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * 6. Compute ops
 * -------------------------------------------------------------------------- */

/* Shared validation shapes.  Capacity checks anchor on size_bytes via
 * wri_tensor_elem_capacity — the raw-pointer fast paths depend on it,
 * and a shape that exceeds the allocation is a caller bug surfaced as
 * WR_ERR_INVAL instead of a silent zero-read. */

static int unary_check(const wr_tensor *A, const wr_tensor *C, uint64_t *n_out)
{
    if (!A || !C || !A->data || !C->data)
        return WR_ERR_INVAL;
    uint64_t n = tensor_elems(A);
    if (n == 0 || tensor_elems(C) < n)
        return WR_ERR_INVAL;
    if (wri_tensor_elem_capacity(A) < n || wri_tensor_elem_capacity(C) < n)
        return WR_ERR_INVAL;
    *n_out = n;
    return WR_OK;
}

static int elemwise2_check(const wr_tensor *A, const wr_tensor *B,
                           const wr_tensor *C, uint64_t *n_out)
{
    if (!A || !B || !C || !A->data || !B->data || !C->data)
        return WR_ERR_INVAL;
    uint64_t n = tensor_elems(A);
    if (n == 0 || tensor_elems(B) != n || tensor_elems(C) < n)
        return WR_ERR_INVAL;
    if (wri_tensor_elem_capacity(A) < n ||
        wri_tensor_elem_capacity(B) < n ||
        wri_tensor_elem_capacity(C) < n)
        return WR_ERR_INVAL;
    *n_out = n;
    return WR_OK;
}

static inline float silu_f(float x)
{
    float sig = 1.0f / (1.0f + wri_exp(-x));
    return x * sig;
}

/* gelu_pytorch_tanh — the exact GeGLU activation:
 * gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))). */
static inline float gelu_tanh_f(float x)
{
    float x3 = x * x * x;
    float z  = 0.7978845608f * (x + 0.044715f * x3);   /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + wri_tanh(z));
}

int wri_op_matmul(const wr_tensor *A, const wr_tensor *B, wr_tensor *C)
{
    if (!A || !B || !C || !A->data || !B->data || !C->data)
        return WR_ERR_INVAL;

    uint32_t M, K, N;
    if (A->ndim >= 2) { M = A->shape[0]; K = A->shape[1]; }
    else { M = 1; K = (uint32_t)tensor_elems(A); }
    N = (B->ndim >= 2) ? B->shape[1] : 1;
    if (M == 0 || K == 0 || N == 0)
        return WR_ERR_INVAL;

    WRI_CTR_ADD(WR_CTR_MATMUL_CALLS, 1);
    WRI_CTR_ADD(WR_CTR_MATMUL_ELEMS, (uint64_t)M * N);

    /* O(1) element-count validation before any raw-pointer path: the
     * fast paths read A[M*K] / B[N*K] and WRITE C[M*N].  Both the
     * logical shape product and the size_bytes-derived capacity must
     * cover the access (the second check also bounds quantized B). */
    if (tensor_elems(A) < (uint64_t)M * K ||
        tensor_elems(B) < (uint64_t)N * K ||
        tensor_elems(C) < (uint64_t)M * N)
        return WR_ERR_INVAL;
    if (wri_tensor_elem_capacity(A) < (uint64_t)M * K ||
        wri_tensor_elem_capacity(B) < (uint64_t)N * K ||
        wri_tensor_elem_capacity(C) < (uint64_t)M * N)
        return WR_ERR_INVAL;

    /* A GGML-layout weight operand stores W[out][in] at out*K+in (`in`
     * contiguous): the correct read is B[j*K+k], not row-major B[k*N+j]. */
    int b_ggml = (B->flags & WR_TENSOR_GGML_WEIGHT) != 0;

    if (A->dtype == WR_DTYPE_F32 && B->dtype == WR_DTYPE_F32 &&
        C->dtype == WR_DTYPE_F32) {
        if (b_ggml) {
            /* Transposed-B is the ideal dot-product SIMD case — both
             * rows stream contiguously in k. */
            WRI_CTR_ADD(WR_CTR_MATMUL_GGML_SIMD, 1);
            matmul_f32_ggml_run(matmul_ggml_kernel_snapshot(),
                                cf32p(A), cf32p(B), f32p(C), M, K, N);
            return WR_OK;
        }
        matmul_f32_run(matmul_kernel_snapshot(),
                       cf32p(A), cf32p(B), f32p(C), M, K, N);
        return WR_OK;
    }

    /* Kept-quantized GGML weight, Q8_0: decode one weight row into a
     * K-float scratch, then the SIMD GGML dot (N=1) across all M
     * activation rows.  Serial by design (matches the origin engine;
     * Q8_0 weights are rare next to the K-quants). */
    if (b_ggml && A->dtype == WR_DTYPE_F32 && C->dtype == WR_DTYPE_F32 &&
        B->dtype == WR_DTYPE_Q8_0 && (K % WR_QK8_0) == 0) {
        float *wrow = (float *)malloc((size_t)K * sizeof(float));
        float *col  = (float *)malloc((size_t)M * sizeof(float));
        if (wrow && col) {
            const uint8_t *b_base = (const uint8_t *)B->data;
            const float   *a      = cf32p(A);
            float         *c      = f32p(C);
            uint32_t blocks_per_row = K / WR_QK8_0;
            uint32_t row_bytes      = blocks_per_row * WR_Q8_0_BLOCK_BYTES;
            wri_matmul_fn ggml_kernel = matmul_ggml_kernel_snapshot();

            WRI_CTR_ADD(WR_CTR_MATMUL_GGML_QUANT, 1);
            for (uint32_t j = 0; j < N; j++) {
                wri_dequant_row_q8_0(b_base + (size_t)j * row_bytes, wrow, K);
                ggml_kernel(a, wrow, col, M, K, 1);
                for (uint32_t i = 0; i < M; i++)
                    c[(size_t)i * N + j] = col[i];
            }
            free(wrow);
            free(col);
            return WR_OK;
        }
        free(wrow);
        free(col);
        /* allocation failure → per-element fallback below */
    }

    /* Kept-quantized GGML weight, K-quants (Q4_K / Q6_K) — the dominant
     * production format.  N-split parallel across the pool; bit-exact
     * versus serial (see the plumbing above).  K beyond the compiled
     * scratch bound falls to the per-element path — counted, never
     * silently wrong. */
    if (b_ggml && A->dtype == WR_DTYPE_F32 && C->dtype == WR_DTYPE_F32 &&
        (B->dtype == WR_DTYPE_Q4_K || B->dtype == WR_DTYPE_Q6_K) &&
        (K % WR_QK_K) == 0 && K <= WR_QMM_K_MAX && M <= WR_QMM_M_MAX) {
        int is_q6 = (B->dtype == WR_DTYPE_Q6_K);
        WRI_CTR_ADD(WR_CTR_MATMUL_GGML_QUANT, 1);
        int rc = qmm_ggml_matmul(cf32p(A), (const uint8_t *)B->data, f32p(C),
                                 M, K, N,
                                 is_q6 ? WR_Q6_K_BLOCK_BYTES
                                       : WR_Q4_K_BLOCK_BYTES,
                                 WR_QK_K,
                                 is_q6 ? wri_dequant_row_q6_k
                                       : wri_dequant_row_q4_k);
        if (rc == WR_OK)
            return WR_OK;
        /* WR_ERR_NOMEM → per-element fallback below */
    }

    /* Q4_K activations × row-major F32 B: decode one A row at a time
     * via the block-aware row decoder, then the plain F32 inner loop. */
    if (A->dtype == WR_DTYPE_Q4_K && B->dtype == WR_DTYPE_F32 &&
        C->dtype == WR_DTYPE_F32 && (K % WR_QK_K) == 0) {
        float *row_scratch = (float *)malloc((size_t)K * sizeof(float));
        if (row_scratch) {
            const uint8_t *a_base = (const uint8_t *)A->data;
            const float   *b      = cf32p(B);
            float         *c      = f32p(C);
            uint32_t blocks_per_row = K / WR_QK_K;
            uint32_t row_bytes      = blocks_per_row * WR_Q4_K_BLOCK_BYTES;

            for (uint32_t i = 0; i < M; i++) {
                wri_dequant_row_q4_k(a_base + (size_t)i * row_bytes,
                                     row_scratch, K);
                for (uint32_t j = 0; j < N; j++) {
                    float sum = 0;
                    for (uint32_t k = 0; k < K; k++)
                        sum += row_scratch[k] * b[(size_t)k * N + j];
                    c[(size_t)i * N + j] = sum;
                }
            }
            free(row_scratch);
            return WR_OK;
        }
        /* allocation failure → per-element fallback below */
    }

    /* Generic fallback: dequant on the fly through the bounds-checked
     * accessors.  ~100x slower than the block paths and counted so a
     * missed fast path during decode is observable, never silent. */
    WRI_CTR_ADD(WR_CTR_MATMUL_PERELEM, 1);
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0;
            for (uint32_t k = 0; k < K; k++)
                sum += wri_read_elem_f32(A, (uint64_t)i * K + k) *
                       wri_read_elem_f32(B, b_ggml
                                            ? (uint64_t)j * K + k
                                            : (uint64_t)k * N + j);
            wri_write_elem_f32(C, (uint64_t)i * N + j, sum);
        }
    }
    return WR_OK;
}

int wri_op_add(const wr_tensor *A, const wr_tensor *B, wr_tensor *C)
{
    uint64_t n;
    int rc = elemwise2_check(A, B, C, &n);
    if (rc != WR_OK)
        return rc;

    if (A->dtype == WR_DTYPE_F32 && B->dtype == WR_DTYPE_F32 &&
        C->dtype == WR_DTYPE_F32) {
        const float *a = cf32p(A);
        const float *b = cf32p(B);
        float *c = f32p(C);
        for (uint64_t i = 0; i < n; i++)
            c[i] = a[i] + b[i];
        return WR_OK;
    }
    for (uint64_t i = 0; i < n; i++)
        wri_write_elem_f32(C, i,
            wri_read_elem_f32(A, i) + wri_read_elem_f32(B, i));
    return WR_OK;
}

int wri_op_mul(const wr_tensor *A, const wr_tensor *B, wr_tensor *C)
{
    uint64_t n;
    int rc = elemwise2_check(A, B, C, &n);
    if (rc != WR_OK)
        return rc;

    for (uint64_t i = 0; i < n; i++)
        wri_write_elem_f32(C, i,
            wri_read_elem_f32(A, i) * wri_read_elem_f32(B, i));
    return WR_OK;
}

int wri_op_relu(const wr_tensor *A, wr_tensor *C)
{
    uint64_t n;
    int rc = unary_check(A, C, &n);
    if (rc != WR_OK)
        return rc;
    for (uint64_t i = 0; i < n; i++) {
        float v = wri_read_elem_f32(A, i);
        wri_write_elem_f32(C, i, v > 0 ? v : 0);
    }
    return WR_OK;
}

int wri_op_gelu(const wr_tensor *A, wr_tensor *C)
{
    uint64_t n;
    int rc = unary_check(A, C, &n);
    if (rc != WR_OK)
        return rc;
    for (uint64_t i = 0; i < n; i++)
        wri_write_elem_f32(C, i, gelu_tanh_f(wri_read_elem_f32(A, i)));
    return WR_OK;
}

int wri_op_silu(const wr_tensor *A, wr_tensor *C)
{
    uint64_t n;
    int rc = unary_check(A, C, &n);
    if (rc != WR_OK)
        return rc;
    for (uint64_t i = 0; i < n; i++)
        wri_write_elem_f32(C, i, silu_f(wri_read_elem_f32(A, i)));
    return WR_OK;
}

int wri_op_softmax(const wr_tensor *A, wr_tensor *C)
{
    uint64_t n;
    int rc = unary_check(A, C, &n);
    if (rc != WR_OK)
        return rc;
    if (A->ndim == 0)
        return WR_ERR_INVAL;
    uint64_t dim = A->shape[A->ndim - 1];
    if (dim == 0)
        return WR_ERR_INVAL;
    uint64_t rows = n / dim;

    WRI_CTR_ADD(WR_CTR_SOFTMAX_CALLS, 1);

    if (A->dtype == WR_DTYPE_F32 && C->dtype == WR_DTYPE_F32) {
        wri_softmax_fn impl = softmax_kernel_snapshot();
        const float *a = cf32p(A);
        float *c = f32p(C);
        for (uint64_t r = 0; r < rows; r++)
            impl(a + r * dim, c + r * dim, dim);
        return WR_OK;
    }

    /* Non-F32: scalar fallback through the accessors, still per row. */
    for (uint64_t r = 0; r < rows; r++) {
        uint64_t base = r * dim;
        float max_val = wri_read_elem_f32(A, base);
        for (uint64_t j = 1; j < dim; j++) {
            float v = wri_read_elem_f32(A, base + j);
            if (v > max_val) max_val = v;
        }
        float sum = 0;
        for (uint64_t j = 0; j < dim; j++) {
            float v = wri_exp(wri_read_elem_f32(A, base + j) - max_val);
            wri_write_elem_f32(C, base + j, v);
            sum += v;
        }
        if (sum > 0) {
            float inv_sum = 1.0f / sum;
            for (uint64_t j = 0; j < dim; j++)
                wri_write_elem_f32(C, base + j,
                    wri_read_elem_f32(C, base + j) * inv_sum);
        }
    }
    return WR_OK;
}

int wri_op_fused_silu_mul(const wr_tensor *A, const wr_tensor *B,
                          wr_tensor *C)
{
    uint64_t n;
    int rc = elemwise2_check(A, B, C, &n);
    if (rc != WR_OK)
        return rc;
    WRI_CTR_ADD(WR_CTR_FUSED_CALLS, 1);
    /* One pass over both inputs — no intermediate activation tensor. */
    for (uint64_t i = 0; i < n; i++) {
        float a = wri_read_elem_f32(A, i);
        float b = wri_read_elem_f32(B, i);
        wri_write_elem_f32(C, i, silu_f(a) * b);
    }
    return WR_OK;
}

int wri_op_fused_gelu_mul(const wr_tensor *A, const wr_tensor *B,
                          wr_tensor *C)
{
    uint64_t n;
    int rc = elemwise2_check(A, B, C, &n);
    if (rc != WR_OK)
        return rc;
    WRI_CTR_ADD(WR_CTR_FUSED_CALLS, 1);
    for (uint64_t i = 0; i < n; i++) {
        float a = wri_read_elem_f32(A, i);
        float b = wri_read_elem_f32(B, i);
        wri_write_elem_f32(C, i, gelu_tanh_f(a) * b);
    }
    return WR_OK;
}

int wri_op_rmsnorm(const wr_tensor *X, const wr_tensor *W, wr_tensor *C,
                   float eps)
{
    uint64_t n;
    int rc = unary_check(X, C, &n);
    if (rc != WR_OK)
        return rc;
    if (X->ndim == 0)
        return WR_ERR_INVAL;
    uint64_t dim = X->shape[X->ndim - 1];
    if (dim == 0)
        return WR_ERR_INVAL;
    uint64_t rows = n / dim;
    uint64_t wn = 0;
    if (W) {
        if (!W->data)
            return WR_ERR_INVAL;
        wn = tensor_elems(W);
    }

    /* The weight is applied DIRECT — no (1+w) transform.  The Gemma
     * variant differs from the Llama-class one only by eps because the
     * loader pre-folds Gemma's (1+w) into the stored norm weights. */
    for (uint64_t r = 0; r < rows; r++) {
        float sumsq = 0;
        for (uint64_t j = 0; j < dim; j++) {
            float v = wri_read_elem_f32(X, r * dim + j);
            sumsq += v * v;
        }
        float inv_rms = 1.0f / wri_sqrt(sumsq / (float)dim + eps);
        for (uint64_t j = 0; j < dim; j++) {
            float v = wri_read_elem_f32(X, r * dim + j);
            float w = (wn > 0) ? wri_read_elem_f32(W, j % wn) : 1.0f;
            wri_write_elem_f32(C, r * dim + j, v * inv_rms * w);
        }
    }
    return WR_OK;
}

int wri_op_layernorm(const wr_tensor *X, const wr_tensor *scale,
                     const wr_tensor *bias, wr_tensor *C, float eps)
{
    uint64_t n;
    int rc = unary_check(X, C, &n);
    if (rc != WR_OK)
        return rc;
    if (X->ndim == 0)
        return WR_ERR_INVAL;
    uint64_t dim = X->shape[X->ndim - 1];
    if (dim == 0)
        return WR_ERR_INVAL;
    uint64_t rows = n / dim;
    uint64_t sn = (scale && scale->data) ? tensor_elems(scale) : 0;
    uint64_t bn = (bias && bias->data) ? tensor_elems(bias) : 0;

    for (uint64_t r = 0; r < rows; r++) {
        uint64_t base = r * dim;
        float mean = 0;
        for (uint64_t j = 0; j < dim; j++)
            mean += wri_read_elem_f32(X, base + j);
        mean /= (float)dim;

        float var = 0;
        for (uint64_t j = 0; j < dim; j++) {
            float diff = wri_read_elem_f32(X, base + j) - mean;
            var += diff * diff;
        }
        var /= (float)dim;
        float inv_std = 1.0f / wri_sqrt(var + eps);

        for (uint64_t j = 0; j < dim; j++) {
            float x = (wri_read_elem_f32(X, base + j) - mean) * inv_std;
            float s = (j < sn) ? wri_read_elem_f32(scale, j) : 1.0f;
            float b = (j < bn) ? wri_read_elem_f32(bias, j) : 0.0f;
            wri_write_elem_f32(C, base + j, x * s + b);
        }
    }
    return WR_OK;
}

int wri_op_rope(const wr_tensor *X, wr_tensor *C, uint32_t pos,
                float theta_base, uint32_t n_rot_pairs)
{
    uint64_t n;
    int rc = unary_check(X, C, &n);
    if (rc != WR_OK)
        return rc;
    if (X->ndim < 2)
        return WR_ERR_INVAL;
    uint64_t hd = X->shape[X->ndim - 1];
    if (hd < 2 || (hd & 1))
        return WR_ERR_INVAL;             /* head_dim must be even */
    uint64_t hd_half = hd / 2;
    uint64_t heads = X->shape[X->ndim - 2];
    if (heads == 0)
        return WR_ERR_INVAL;
    uint64_t seq = n / (heads * hd);

    if (theta_base < 1.0f)
        theta_base = 10000.0f;
    float lnb = wri_log(theta_base);

    /* Partial rotary: rotate only the first n_rot pairs, pass the rest
     * through unrotated (the reference model applies a huge frequency
     * factor to the high pairs, which zeroes their rotation). */
    uint64_t n_rot = hd_half;
    if (n_rot_pairs >= 1 && (uint64_t)n_rot_pairs < hd_half)
        n_rot = n_rot_pairs;

    /* Llama-style non-interleaved pairing: element j pairs with
     * j + hd/2.  C may alias X — each pair is read fully before its
     * two writes, so in-place rotation is well-defined. */
    for (uint64_t s = 0; s < seq; s++) {
        uint64_t p = (uint64_t)pos + s;
        for (uint64_t h = 0; h < heads; h++) {
            uint64_t row = (s * heads + h) * hd;
            for (uint64_t j = 0; j < hd_half; j++) {
                float x0 = wri_read_elem_f32(X, row + j);
                float x1 = wri_read_elem_f32(X, row + j + hd_half);
                if (j >= n_rot) {                    /* pass-through pair */
                    wri_write_elem_f32(C, row + j,           x0);
                    wri_write_elem_f32(C, row + j + hd_half, x1);
                    continue;
                }
                /* freq = 1 / theta_base^(2j/hd), via exp(exponent*ln(base)) */
                float exponent = (float)(2 * j) / (float)hd;
                float powv = wri_exp(exponent * lnb);
                float angle = (float)p / powv;
                float cs = wri_cos(angle);
                float sn = wri_sin(angle);
                wri_write_elem_f32(C, row + j,           x0 * cs - x1 * sn);
                wri_write_elem_f32(C, row + j + hd_half, x0 * sn + x1 * cs);
            }
        }
    }
    return WR_OK;
}

int wri_op_gqa_attention(const wr_tensor *Q, const wr_tensor *K,
                         const wr_tensor *V, wr_tensor *C,
                         float scale_override, uint32_t window)
{
    if (!Q || !K || !V || !C || !Q->data || !K->data || !V->data || !C->data)
        return WR_ERR_INVAL;
    if (Q->ndim < 3 || K->ndim < 3 || V->ndim < 3)
        return WR_ERR_INVAL;

    WRI_CTR_ADD(WR_CTR_ATTENTION_CALLS, 1);

    uint64_t seq_q  = Q->shape[0];
    uint64_t n_q_h  = Q->shape[1];
    uint64_t hd     = Q->shape[2];
    uint64_t n_kv_h = K->shape[1];
    if (n_q_h == 0 || n_kv_h == 0 || hd == 0 || seq_q == 0)
        return WR_ERR_INVAL;
    if (n_q_h % n_kv_h != 0)
        return WR_ERR_INVAL;
    if (K->shape[2] != hd || V->shape[2] != hd || V->shape[1] != n_kv_h)
        return WR_ERR_INVAL;

    /* KV-cache read path: a GROWABLE K/V exposes only valid_rows. */
    uint64_t seq_k = ((K->flags & WR_TENSOR_GROWABLE) && K->valid_rows > 0)
                     ? K->valid_rows : K->shape[0];
    if (seq_k == 0)
        return WR_ERR_INVAL;

    /* The origin engine silently truncated long sequences here; that is
     * banned — the cap is a hard error now. */
    if (seq_k > WR_ATTN_MAX_SEQ)
        return WR_ERR_LIMIT;
    if (hd > WR_FLASH_D_MAX)
        return WR_ERR_LIMIT;

    if (wri_tensor_elem_capacity(Q) < seq_q * n_q_h * hd ||
        wri_tensor_elem_capacity(K) < seq_k * n_kv_h * hd ||
        wri_tensor_elem_capacity(V) < seq_k * n_kv_h * hd ||
        wri_tensor_elem_capacity(C) < seq_q * n_q_h * hd ||
        tensor_elems(C) < seq_q * n_q_h * hd)
        return WR_ERR_INVAL;

    uint64_t group = n_q_h / n_kv_h;
    float scale = (scale_override > 0.0f)
                  ? scale_override
                  : 1.0f / wri_sqrt((float)hd);

    /* Sliding window: keys with index < key_lo are masked out; a query
     * attends only to the last `window` keys. */
    uint64_t key_lo = (window > 0 && seq_k > (uint64_t)window)
                      ? seq_k - (uint64_t)window : 0;

    /* Flash attention v1: process K/V in WR_FLASH_BLOCK tiles with a
     * running max m, denominator l and per-head accumulator o[hd] —
     * O(FLASH_BLOCK + hd) scratch, no O(seq_k) score materialization.
     * Identical to the materialized softmax modulo float associativity
     * (same online-softmax algebra). */
    float *o = (float *)malloc((size_t)hd * sizeof(float));
    if (!o)
        return WR_ERR_NOMEM;

    for (uint64_t qs = 0; qs < seq_q; qs++) {
        for (uint64_t qh = 0; qh < n_q_h; qh++) {
            uint64_t kvh   = qh / group;            /* GQA head mapping */
            uint64_t q_row = (qs * n_q_h + qh) * hd;

            float m = -1e30f, l = 0;
            for (uint64_t d = 0; d < hd; d++) o[d] = 0.0f;

            for (uint64_t tile = 0; tile < seq_k; tile += WR_FLASH_BLOCK) {
                uint64_t Bn = seq_k - tile;
                if (Bn > WR_FLASH_BLOCK) Bn = WR_FLASH_BLOCK;

                float S[WR_FLASH_BLOCK];
                float tmx = -1e30f;
                for (uint64_t b = 0; b < Bn; b++) {
                    /* Masked keys contribute score -inf: their exp is 0,
                     * keeping the online softmax exact. */
                    if (tile + b < key_lo) { S[b] = -1e30f; continue; }
                    uint64_t k_row = ((tile + b) * n_kv_h + kvh) * hd;
                    float dot = 0;
                    for (uint64_t d = 0; d < hd; d++)
                        dot += wri_read_elem_f32(Q, q_row + d) *
                               wri_read_elem_f32(K, k_row + d);
                    S[b] = dot * scale;
                    if (S[b] > tmx) tmx = S[b];
                }

                /* Online softmax: rescale accumulator + denominator. */
                float m_new = (m > tmx) ? m : tmx;
                float a = (m == -1e30f) ? 0.0f : wri_exp(m - m_new);
                for (uint64_t d = 0; d < hd; d++) o[d] *= a;
                l *= a;

                for (uint64_t b = 0; b < Bn; b++) {
                    float pw = wri_exp(S[b] - m_new);
                    l += pw;
                    uint64_t v_row = ((tile + b) * n_kv_h + kvh) * hd;
                    for (uint64_t d = 0; d < hd; d++)
                        o[d] += pw * wri_read_elem_f32(V, v_row + d);
                }
                m = m_new;
            }

            uint64_t out_row = (qs * n_q_h + qh) * hd;
            float inv = (l > 0) ? 1.0f / l : 0.0f;
            for (uint64_t d = 0; d < hd; d++)
                wri_write_elem_f32(C, out_row + d, o[d] * inv);
        }
    }

    free(o);
    return WR_OK;
}

int wri_op_embed(const wr_tensor *table, uint32_t token_id,
                 wr_tensor *out_row)
{
    if (!table || !out_row || !table->data || !out_row->data)
        return WR_ERR_INVAL;
    if (table->ndim < 2)
        return WR_ERR_INVAL;
    uint64_t V = table->shape[0];
    if (V == 0)
        return WR_ERR_INVAL;
    uint64_t D = tensor_elems(table) / V;
    if (D == 0)
        return WR_ERR_INVAL;
    if ((uint64_t)token_id >= V)
        return WR_ERR_INVAL;             /* never clamped */
    if (tensor_elems(out_row) < D ||
        wri_tensor_elem_capacity(out_row) < D ||
        wri_tensor_elem_capacity(table) < V * D)
        return WR_ERR_INVAL;

    uint64_t base = (uint64_t)token_id * D;

    if (out_row->dtype == WR_DTYPE_F32) {
        float *dst = f32p(out_row);
        if (table->dtype == WR_DTYPE_F32) {
            memcpy(dst, cf32p(table) + base, (size_t)D * sizeof(float));
            return WR_OK;
        }
        /* Kept-quantized embedding table: whole-row block decode when
         * the row starts on a block boundary (D % block == 0 — true for
         * every production hidden size). */
        uint32_t be = blk_elems_of((wr_dtype)table->dtype);
        if (be != 0 && (D % be) == 0) {
            const uint8_t *blocks = (const uint8_t *)table->data
                + (base / be) * (uint64_t)blk_bytes_of((wr_dtype)table->dtype);
            return wri_dequant_row((wr_dtype)table->dtype, blocks, dst, D);
        }
        for (uint64_t j = 0; j < D; j++)
            dst[j] = wri_read_elem_f32(table, base + j);
        return WR_OK;
    }

    for (uint64_t j = 0; j < D; j++)
        wri_write_elem_f32(out_row, j, wri_read_elem_f32(table, base + j));
    return WR_OK;
}

void wri_softcap(float *x, uint64_t n, float cap)
{
    if (!x || cap <= 0.0f)
        return;
    /* Divide, never multiply by a reciprocal: 1/cap is inexact and the
     * origin engine (and the documented formula) divides. */
    for (uint64_t i = 0; i < n; i++)
        x[i] = cap * wri_tanh(x[i] / cap);
}

/* --------------------------------------------------------------------------
 * 7. Golden self-tests
 *
 * Ported from the origin engine's boot-time numeric self-tests; the
 * kernel-scheduler/FPU-state legs are gone (meaningless in a hosted
 * process).  Each returns 0 on pass.  test/unit_tests.c drives them.
 * -------------------------------------------------------------------------- */

static int feq(float a, float b, float tol)
{
    float d = a - b;
    if (d < 0) d = -d;
    return d <= tol;
}

/* Fill a stack descriptor over caller-provided storage. */
static void mk_tensor(wr_tensor *t, wr_dtype dt, uint32_t ndim,
                      const uint32_t *shape, void *data)
{
    memset(t, 0, sizeof(*t));
    uint64_t n = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        n *= shape[i];
    }
    t->ndim  = (uint8_t)ndim;
    t->dtype = (uint8_t)dt;
    t->data  = data;
    t->size_bytes = wri_dtype_bytes_for_count(dt, n);
}

int wri_self_test_quant_dequant(void)
{
    int fail = 0;

    /* Q8_0: scale = 2.0 (fp16 0x4000), quants 0..31 → elem i = 2*i. */
    {
        uint8_t buf[WR_Q8_0_BLOCK_BYTES] = {0};
        buf[0] = 0x00; buf[1] = 0x40;
        for (int q = 0; q < 32; q++) buf[2 + q] = (uint8_t)q;
        for (int i = 0; i < 32; i++) {
            if (wri_dequant_q8_0_elem(buf, (uint32_t)i) != (float)i * 2.0f) {
                wri_log_msg(0, "self-test quant: Q8_0 mismatch at %d", i);
                fail++;
                break;
            }
        }
    }

    /* Q4_0 nibble order: d = 1.0, byte 0x29 packs elem0 = 9 (LOW
     * nibble), elem1 = 2 (high) → (9-8)*1 = 1.0 and (2-8)*1 = -6.0. */
    {
        uint8_t buf[WR_Q4_0_BLOCK_BYTES] = {0};
        buf[0] = 0x00; buf[1] = 0x3C;
        buf[2] = 0x29;
        if (wri_dequant_q4_0_elem(buf, 0) != 1.0f ||
            wri_dequant_q4_0_elem(buf, 1) != -6.0f) {
            wri_log_msg(0, "self-test quant: Q4_0 nibble order broken");
            fail++;
        }
    }

    /* Q4_K: d=1.0, dmin=0, scales[0]=63 (sc=63, m=0 for sub-block 0),
     * qs[0] low nibble = 5 → 1.0 * 63 * 5 = 315.0. */
    {
        uint8_t buf[WR_Q4_K_BLOCK_BYTES] = {0};
        buf[0] = 0x00; buf[1] = 0x3C;
        buf[4] = 63;
        buf[16] = 0x05;
        if (wri_dequant_q4_k_elem(buf, 0) != 315.0f) {
            wri_log_msg(0, "self-test quant: Q4_K expect 315 got %d",
                    (int)wri_dequant_q4_k_elem(buf, 0));
            fail++;
        }
    }

    /* Q5_K: d=1.0, scales[0]=20, qs[0]=0x07, qh[0] bit0 = 1 →
     * q = 7 | 16 = 23 → 20 * 23 = 460. */
    {
        uint8_t buf[WR_Q5_K_BLOCK_BYTES] = {0};
        buf[0] = 0x00; buf[1] = 0x3C;
        buf[4] = 20;
        buf[16] = 0x01;
        buf[48] = 0x07;
        if (wri_dequant_q5_k_elem(buf, 0) != 460.0f) {
            wri_log_msg(0, "self-test quant: Q5_K expect 460 got %d",
                    (int)wri_dequant_q5_k_elem(buf, 0));
            fail++;
        }
    }

    /* Q6_K: d=1.0, scales[0]=10, ql[0] low nibble = 5, qh[0] low 2
     * bits = 1 → q = 21, centered 21-32 = -11 → 10 * -11 = -110. */
    {
        uint8_t buf[WR_Q6_K_BLOCK_BYTES] = {0};
        buf[0]   = 0x05;
        buf[128] = 0x01;
        buf[192] = 10;
        buf[208] = 0x00; buf[209] = 0x3C;
        if (wri_dequant_q6_k_elem(buf, 0) != -110.0f) {
            wri_log_msg(0, "self-test quant: Q6_K expect -110 got %d",
                    (int)wri_dequant_q6_k_elem(buf, 0));
            fail++;
        }
    }

    /* Q6_K whole-row decode bit-equal to per-element × 256, on a block
     * with every ql/qh byte and all 16 scales distinct so any element-
     * order or scale-index bug shows up (the LM-head fast path decodes
     * rows through this). */
    {
        uint8_t blk[WR_Q6_K_BLOCK_BYTES] = {0};
        for (int i = 0; i < 128; i++) blk[i]       = (uint8_t)(i ^ 0x5A);
        for (int i = 0; i < 64;  i++) blk[128 + i] = (uint8_t)(i * 7 + 1);
        for (int i = 0; i < 16;  i++) blk[192 + i] = (uint8_t)(i - 8);
        blk[208] = 0x00; blk[209] = 0x3C;
        float be[256], bb[256];
        for (int i = 0; i < 256; i++)
            be[i] = wri_dequant_q6_k_elem(blk, (uint32_t)i);
        wri_dequant_row_q6_k(blk, bb, 256);
        if (memcmp(be, bb, sizeof(be)) != 0) {
            wri_log_msg(0, "self-test quant: Q6_K row decode != elem decode");
            fail++;
        }
    }

    /* Q4_K whole-row decode bit-equal to per-element × 256. */
    {
        uint8_t blk[WR_Q4_K_BLOCK_BYTES] = {0};
        blk[0] = 0x00; blk[1] = 0x3C;   /* d    = 1.0 */
        blk[2] = 0x00; blk[3] = 0x38;   /* dmin = 0.5 */
        for (int i = 0; i < 12;  i++) blk[4 + i]  = (uint8_t)(0x33 + i);
        for (int i = 0; i < 128; i++) blk[16 + i] = (uint8_t)(i ^ 0xA5);
        float be[256], bb[256];
        for (int i = 0; i < 256; i++)
            be[i] = wri_dequant_q4_k_elem(blk, (uint32_t)i);
        wri_dequant_row_q4_k(blk, bb, 256);
        if (memcmp(be, bb, sizeof(be)) != 0) {
            wri_log_msg(0, "self-test quant: Q4_K row decode != elem decode");
            fail++;
        }
    }

    return fail ? -1 : 0;
}

int wri_self_test_matmul_simd(void)
{
    /* N=127 crosses the 4-, 2-, 1-vector and scalar-tail paths of the
     * AVX2 and AVX-512 kernels.  Small integral values keep every sum
     * exactly representable, so equality is strict, not tolerant. */
    enum { RM_M = 2, RM_K = 5, RM_N = 127 };
    float a[RM_M * RM_K];
    float b[RM_K * RM_N];
    float c_scalar[RM_M * RM_N];
    float c_disp[RM_M * RM_N];
    wri_matmul_fn native_kernel = matmul_kernel_snapshot();
    wri_matmul_fn ggml_kernel   = matmul_ggml_kernel_snapshot();

    for (uint32_t i = 0; i < RM_M * RM_K; i++)
        a[i] = (float)((int)(i % 7) - 3);
    for (uint32_t i = 0; i < RM_K * RM_N; i++)
        b[i] = (float)((int)(i % 5) - 2);

    matmul_f32_scalar(a, b, c_scalar, RM_M, RM_K, RM_N);
    native_kernel(a, b, c_disp, RM_M, RM_K, RM_N);
    if (memcmp(c_scalar, c_disp, sizeof(c_scalar)) != 0) {
        wri_log_msg(0, "self-test matmul: native kernel != scalar (variant=%d)",
                wri_simd_matmul_variant());
        return -1;
    }

    /* Misaligned-buffer guard for the aligned(4)/VMOVUPS decision: run
     * the dispatched kernel on pointers that are 4-byte aligned but
     * deliberately NOT 16/32/64-byte aligned.  A kernel that regressed
     * to aligned vector moves faults here instead of in production. */
    {
        char *raw_a = (char *)malloc(sizeof(a) + 4);
        char *raw_b = (char *)malloc(sizeof(b) + 4);
        char *raw_c = (char *)malloc(sizeof(c_disp) + 4);
        if (!raw_a || !raw_b || !raw_c) {
            free(raw_a); free(raw_b); free(raw_c);
            wri_log_msg(0, "self-test matmul: misaligned fixture alloc failed");
            return -1;
        }
        float *ma = (float *)(raw_a + 4);
        float *mb = (float *)(raw_b + 4);
        float *mc = (float *)(raw_c + 4);
        memcpy(ma, a, sizeof(a));
        memcpy(mb, b, sizeof(b));
        native_kernel(ma, mb, mc, RM_M, RM_K, RM_N);
        int bad = memcmp(c_scalar, mc, sizeof(c_scalar)) != 0;
        free(raw_a); free(raw_b); free(raw_c);
        if (bad) {
            wri_log_msg(0, "self-test matmul: misaligned run != scalar");
            return -1;
        }
    }

    /* The quantized decode path invokes the GGML kernel with N=1 after
     * decoding a weight row.  K=65 forces the four-chain body plus its
     * scalar tail; N=2 also checks row selection. */
    enum { GG_M = 1, GG_K = 65, GG_N = 2 };
    float ga[GG_M * GG_K];
    float gb[GG_N * GG_K];
    float gc_scalar[GG_M * GG_N];
    float gc_disp[GG_M * GG_N];
    for (uint32_t i = 0; i < GG_M * GG_K; i++)
        ga[i] = (float)((int)(i % 7) - 3);
    for (uint32_t i = 0; i < GG_N * GG_K; i++)
        gb[i] = (float)((int)(i % 5) - 2);
    matmul_f32_ggml_scalar(ga, gb, gc_scalar, GG_M, GG_K, GG_N);
    ggml_kernel(ga, gb, gc_disp, GG_M, GG_K, GG_N);
    if (memcmp(gc_scalar, gc_disp, sizeof(gc_scalar)) != 0) {
        wri_log_msg(0, "self-test matmul: GGML kernel != scalar (variant=%d)",
                wri_simd_matmul_variant());
        return -1;
    }

    /* M-split partitioning bit-equality: compute the same matmul in 4
     * sequential row blocks (the exact split the pool uses) and require
     * bit-identity with the single-call result. */
    {
        enum { PM = 12, PK = 8, PN = 10 };
        float pa[PM * PK], pb[PK * PN];
        float c_serial[PM * PN], c_split[PM * PN];
        for (int i = 0; i < PM * PK; i++) pa[i] = (float)((i * 7) % 19) - 9.0f;
        for (int i = 0; i < PK * PN; i++) pb[i] = (float)((i * 5) % 17) - 8.0f;
        native_kernel(pa, pb, c_serial, PM, PK, PN);
        uint32_t base = PM / 4, rem = PM % 4, cursor = 0;
        for (uint32_t w = 0; w < 4; w++) {
            uint32_t cnt = base + (w < rem ? 1u : 0u);
            native_kernel(pa + (size_t)cursor * PK, pb,
                          c_split + (size_t)cursor * PN, cnt, PK, PN);
            cursor += cnt;
        }
        if (memcmp(c_serial, c_split, sizeof(c_serial)) != 0) {
            wri_log_msg(0, "self-test matmul: row-split != serial");
            return -1;
        }
    }

    return 0;
}

int wri_self_test_softmax(void)
{
    static const float input[16] = {
        0.5f, -1.2f, 3.4f, 0.0f,  -0.5f, 2.1f, -3.0f, 1.7f,
        0.1f,  0.2f, 0.3f, 0.4f,   0.5f, 0.6f, 0.7f,  0.8f,
    };
    float out_scalar[16], out_simd[16];
    softmax_f32_scalar(input, out_scalar, 16);
    softmax_kernel_snapshot()(input, out_simd, 16);

    /* The polynomial accumulates differently when vectorized; 1e-4 is
     * generous for that and tight enough to catch real bugs. */
    for (int i = 0; i < 16; i++) {
        if (!feq(out_scalar[i], out_simd[i], 1e-4f)) {
            wri_log_msg(0, "self-test softmax: lane %d diverges (variant=%d)",
                    i, wri_simd_softmax_variant());
            return -1;
        }
    }

    /* Per-row semantics of the op: each row of a [2,8] input must
     * normalize independently to sum 1. */
    {
        float x[16], y[16];
        for (int i = 0; i < 8; i++)  x[i] = (float)i * 0.25f;
        for (int i = 8; i < 16; i++) x[i] = (float)(15 - i) * 0.5f;
        uint32_t shp[2] = { 2, 8 };
        wr_tensor A, C;
        mk_tensor(&A, WR_DTYPE_F32, 2, shp, x);
        mk_tensor(&C, WR_DTYPE_F32, 2, shp, y);
        if (wri_op_softmax(&A, &C) != WR_OK) {
            wri_log_msg(0, "self-test softmax: op returned error");
            return -1;
        }
        float s0 = 0, s1 = 0;
        for (int i = 0; i < 8; i++)  s0 += y[i];
        for (int i = 8; i < 16; i++) s1 += y[i];
        if (!feq(s0, 1.0f, 1e-4f) || !feq(s1, 1.0f, 1e-4f)) {
            wri_log_msg(0, "self-test softmax: rows do not normalize independently");
            return -1;
        }
    }
    return 0;
}

/* Materialized softmax-attention reference (MHA layout: one head).
 * scores[] must hold SK floats. */
static void attn_reference(const float *q, const float *k, const float *v,
                           float *out, uint32_t SQ, uint32_t SK, uint32_t D,
                           float scale, uint32_t key_lo, float *scores)
{
    for (uint32_t qi = 0; qi < SQ; qi++) {
        float mx = -1e30f;
        for (uint32_t ki = 0; ki < SK; ki++) {
            if (ki < key_lo) { scores[ki] = -1e30f; continue; }
            float dot = 0;
            for (uint32_t d = 0; d < D; d++)
                dot += q[qi * D + d] * k[ki * D + d];
            scores[ki] = dot * scale;
            if (scores[ki] > mx) mx = scores[ki];
        }
        float sum = 0;
        for (uint32_t ki = 0; ki < SK; ki++) {
            scores[ki] = wri_exp(scores[ki] - mx);
            sum += scores[ki];
        }
        float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
        for (uint32_t d = 0; d < D; d++) {
            float acc = 0;
            for (uint32_t ki = 0; ki < SK; ki++)
                acc += scores[ki] * v[ki * D + d];
            out[qi * D + d] = acc * inv;
        }
    }
}

int wri_self_test_flash_attn(void)
{
    /* Small fixture: seq_q=2, seq_k=8 (single tile), d=4; MHA as the
     * degenerate GQA (1 query head == 1 kv head). */
    enum { SQ = 2, SK = 8, D = 4 };
    static const float q_data[SQ * D] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
    };
    float k_data[SK * D];
    float v_data[SK * D];
    float ref_out[SQ * D];
    float op_out[SQ * D];
    float scores[SK];
    for (int i = 0; i < SK * D; i++) {
        k_data[i] = (float)((i * 7 + 3) % 13) * 0.1f;
        v_data[i] = (float)((i * 5 + 1) % 11) * 0.1f;
    }

    uint32_t qsh[3] = { SQ, 1, D };
    uint32_t ksh[3] = { SK, 1, D };
    wr_tensor Q, K, V, C;
    mk_tensor(&Q, WR_DTYPE_F32, 3, qsh, (void *)q_data);
    mk_tensor(&K, WR_DTYPE_F32, 3, ksh, k_data);
    mk_tensor(&V, WR_DTYPE_F32, 3, ksh, v_data);
    mk_tensor(&C, WR_DTYPE_F32, 3, qsh, op_out);

    float scale = 1.0f / wri_sqrt((float)D);
    attn_reference(q_data, k_data, v_data, ref_out, SQ, SK, D, scale, 0,
                   scores);
    if (wri_op_gqa_attention(&Q, &K, &V, &C, 0.0f, 0) != WR_OK) {
        wri_log_msg(0, "self-test flash: op returned error");
        return -1;
    }
    for (int i = 0; i < SQ * D; i++) {
        if (!feq(ref_out[i], op_out[i], 1e-4f)) {
            wri_log_msg(0, "self-test flash: small case idx %d diverges", i);
            return -1;
        }
    }

    /* Explicit attention-scale override. */
    attn_reference(q_data, k_data, v_data, ref_out, SQ, SK, D, 0.25f, 0,
                   scores);
    if (wri_op_gqa_attention(&Q, &K, &V, &C, 0.25f, 0) != WR_OK) {
        wri_log_msg(0, "self-test flash: scale-override op error");
        return -1;
    }
    for (int i = 0; i < SQ * D; i++) {
        if (!feq(ref_out[i], op_out[i], 1e-4f)) {
            wri_log_msg(0, "self-test flash: scale override idx %d diverges", i);
            return -1;
        }
    }

    /* Sliding-window masking: only the last 4 keys attendable. */
    attn_reference(q_data, k_data, v_data, ref_out, SQ, SK, D, scale,
                   SK - 4, scores);
    if (wri_op_gqa_attention(&Q, &K, &V, &C, 0.0f, 4) != WR_OK) {
        wri_log_msg(0, "self-test flash: window op error");
        return -1;
    }
    for (int i = 0; i < SQ * D; i++) {
        if (!feq(ref_out[i], op_out[i], 1e-4f)) {
            wri_log_msg(0, "self-test flash: window idx %d diverges", i);
            return -1;
        }
    }

    /* Long sequences: SK in {64, 256, 1024, 4096} through a GROWABLE
     * K/V (the KV-cache read path — valid_rows, not shape[0], bounds
     * the scan).  Deterministic fills that keep exp() ranges sane. */
    {
        enum { LQ = 2, LD = 8, SK_MAX = 4096 };
        static const uint32_t sk_set[] = { 64, 256, 1024, 4096 };
        float *k_buf   = (float *)malloc((size_t)SK_MAX * LD * sizeof(float));
        float *v_buf   = (float *)malloc((size_t)SK_MAX * LD * sizeof(float));
        float *sc_buf  = (float *)malloc((size_t)SK_MAX * sizeof(float));
        float  q_buf[LQ * LD];
        float  ref_buf[LQ * LD];
        float  out_buf[LQ * LD];
        if (!k_buf || !v_buf || !sc_buf) {
            free(k_buf); free(v_buf); free(sc_buf);
            wri_log_msg(0, "self-test flash: long fixture alloc failed");
            return -1;
        }
        for (uint32_t i = 0; i < LQ * LD; i++)
            q_buf[i] = (float)((i * 11 + 3) % 17) * 0.05f;
        for (uint32_t i = 0; i < SK_MAX * LD; i++) {
            uint32_t s = i * 2654435761u + 0x9E3779B9u;
            k_buf[i] = ((float)(int32_t)(s & 0xFFFF) / 32768.0f - 1.0f) * 0.3f;
            v_buf[i] = ((float)(int32_t)((s >> 16) & 0xFFFF) / 32768.0f - 1.0f)
                       * 0.3f;
        }
        float lscale = 1.0f / wri_sqrt((float)LD);

        uint32_t lqsh[3] = { LQ, 1, LD };
        uint32_t lksh[3] = { SK_MAX, 1, LD };
        wr_tensor LQt, LKt, LVt, LCt;
        mk_tensor(&LQt, WR_DTYPE_F32, 3, lqsh, q_buf);
        mk_tensor(&LKt, WR_DTYPE_F32, 3, lksh, k_buf);
        mk_tensor(&LVt, WR_DTYPE_F32, 3, lksh, v_buf);
        mk_tensor(&LCt, WR_DTYPE_F32, 3, lqsh, out_buf);
        LKt.flags |= WR_TENSOR_GROWABLE;
        LVt.flags |= WR_TENSOR_GROWABLE;

        for (unsigned t = 0; t < sizeof(sk_set) / sizeof(sk_set[0]); t++) {
            uint32_t SKl = sk_set[t];
            LKt.valid_rows = SKl;
            LVt.valid_rows = SKl;
            attn_reference(q_buf, k_buf, v_buf, ref_buf, LQ, SKl, LD,
                           lscale, 0, sc_buf);
            if (wri_op_gqa_attention(&LQt, &LKt, &LVt, &LCt, 0.0f, 0)
                    != WR_OK) {
                wri_log_msg(0, "self-test flash: long op error at SK=%u",
                        (unsigned)SKl);
                free(k_buf); free(v_buf); free(sc_buf);
                return -1;
            }
            for (uint32_t i = 0; i < LQ * LD; i++) {
                if (!feq(ref_buf[i], out_buf[i], 1e-4f)) {
                    wri_log_msg(0, "self-test flash: SK=%u idx %u diverges",
                            (unsigned)SKl, (unsigned)i);
                    free(k_buf); free(v_buf); free(sc_buf);
                    return -1;
                }
            }
        }
        free(k_buf); free(v_buf); free(sc_buf);
    }

    /* The sequence cap is a hard error, never a silent truncation. */
    {
        enum { OV = WR_ATTN_MAX_SEQ + 1 };
        float *kv = (float *)malloc((size_t)OV * sizeof(float) * 2);
        if (kv) {
            float qv[1] = { 1.0f };
            float ov[1];
            uint32_t qsh1[3] = { 1, 1, 1 };
            uint32_t ksh1[3] = { OV, 1, 1 };
            wr_tensor Qt, Kt, Vt, Ct;
            mk_tensor(&Qt, WR_DTYPE_F32, 3, qsh1, qv);
            mk_tensor(&Kt, WR_DTYPE_F32, 3, ksh1, kv);
            mk_tensor(&Vt, WR_DTYPE_F32, 3, ksh1, kv + OV);
            mk_tensor(&Ct, WR_DTYPE_F32, 3, qsh1, ov);
            int rc = wri_op_gqa_attention(&Qt, &Kt, &Vt, &Ct, 0.0f, 0);
            free(kv);
            if (rc != WR_ERR_LIMIT) {
                wri_log_msg(0, "self-test flash: over-cap seq_k not refused");
                return -1;
            }
        }
    }

    return 0;
}

int wri_self_test_fused(void)
{
    float gate[4] = { 1.0f, 2.0f, -1.0f, 0.5f };
    float up[4]   = { 0.5f, 0.5f, 2.0f,  4.0f };
    float out[4];
    uint32_t shp[1] = { 4 };
    wr_tensor G, U, O;
    mk_tensor(&G, WR_DTYPE_F32, 1, shp, gate);
    mk_tensor(&U, WR_DTYPE_F32, 1, shp, up);
    mk_tensor(&O, WR_DTYPE_F32, 1, shp, out);

    if (wri_op_fused_silu_mul(&G, &U, &O) != WR_OK) {
        wri_log_msg(0, "self-test fused: silu_mul returned error");
        return -1;
    }
    for (int i = 0; i < 4; i++) {
        float sig = 1.0f / (1.0f + wri_exp(-gate[i]));
        float ref = gate[i] * sig * up[i];
        if (out[i] != ref) {
            wri_log_msg(0, "self-test fused: silu_mul idx %d diverges", i);
            return -1;
        }
    }

    if (wri_op_fused_gelu_mul(&G, &U, &O) != WR_OK) {
        wri_log_msg(0, "self-test fused: gelu_mul returned error");
        return -1;
    }
    for (int i = 0; i < 4; i++) {
        float x  = gate[i];
        float x3 = x * x * x;
        float z  = 0.7978845608f * (x + 0.044715f * x3);
        float ref = 0.5f * x * (1.0f + wri_tanh(z)) * up[i];
        if (out[i] != ref) {
            wri_log_msg(0, "self-test fused: gelu_mul idx %d diverges", i);
            return -1;
        }
    }

    /* Shape mismatch must be refused, not truncated. */
    {
        uint32_t shp3[1] = { 3 };
        wr_tensor U3;
        mk_tensor(&U3, WR_DTYPE_F32, 1, shp3, up);
        if (wri_op_fused_silu_mul(&G, &U3, &O) != WR_ERR_INVAL) {
            wri_log_msg(0, "self-test fused: shape mismatch not refused");
            return -1;
        }
    }
    return 0;
}

static int ops_test_rmsnorm(void)
{
    /* Row [1,2,3,4]: mean(x^2) = 7.5, rms ~ 2.7386; all-ones weight →
     * out[i] = x[i] / 2.7386. */
    float x[4] = { 1, 2, 3, 4 };
    float w[4] = { 1, 1, 1, 1 };
    float y[4];
    uint32_t shp[1] = { 4 };
    wr_tensor X, W, C;
    mk_tensor(&X, WR_DTYPE_F32, 1, shp, x);
    mk_tensor(&W, WR_DTYPE_F32, 1, shp, w);
    mk_tensor(&C, WR_DTYPE_F32, 1, shp, y);
    if (wri_op_rmsnorm(&X, &W, &C, 1e-5f) != WR_OK) return -1;
    if (!feq(y[0], 1.0f / 2.7386f, 0.01f) ||
        !feq(y[3], 4.0f / 2.7386f, 0.02f))
        return -1;

    /* Weightless variant (W == NULL — the Gemma V-norm shape). */
    float y2[4];
    wr_tensor C2;
    mk_tensor(&C2, WR_DTYPE_F32, 1, shp, y2);
    if (wri_op_rmsnorm(&X, NULL, &C2, 1e-6f) != WR_OK) return -1;
    for (int i = 0; i < 4; i++)
        if (!feq(y2[i], x[i] / 2.7386f, 0.02f)) return -1;
    return 0;
}

static int ops_test_silu(void)
{
    float in[4] = { -2.0f, 0.0f, 1.0f, 4.0f };
    float out[4];
    uint32_t shp[1] = { 4 };
    wr_tensor X, C;
    mk_tensor(&X, WR_DTYPE_F32, 1, shp, in);
    mk_tensor(&C, WR_DTYPE_F32, 1, shp, out);
    if (wri_op_silu(&X, &C) != WR_OK) return -1;
    /* silu(0)=0; silu(1)~0.7311; silu(4)~3.928 */
    if (!feq(out[1], 0.0f, 0.01f) ||
        !feq(out[2], 0.7311f, 0.02f) ||
        !feq(out[3], 3.928f, 0.05f))
        return -1;
    return 0;
}

static int ops_test_rope(void)
{
    /* [seq=2, heads=1, head_dim=4], values 1..8, position offset 0. */
    float x[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    float y[8];
    uint32_t shp[3] = { 2, 1, 4 };
    wr_tensor X, C;
    mk_tensor(&X, WR_DTYPE_F32, 3, shp, x);
    mk_tensor(&C, WR_DTYPE_F32, 3, shp, y);
    if (wri_op_rope(&X, &C, 0, 10000.0f, 0) != WR_OK) return -1;
    /* pos=0 → angle 0 → identity for the first row. */
    if (!feq(y[0], 1.0f, 0.01f) || !feq(y[2], 3.0f, 0.01f)) return -1;
    /* pos=1, pair j=0: angle 1.0 → 5*cos(1) - 7*sin(1) ~ -3.193. */
    if (!feq(y[4], 5.0f * 0.5403f - 7.0f * 0.8415f, 0.05f)) return -1;

    /* Partial rotary: n_rot_pairs=1 leaves pair j=1 untouched. */
    float y2[8];
    wr_tensor C2;
    mk_tensor(&C2, WR_DTYPE_F32, 3, shp, y2);
    if (wri_op_rope(&X, &C2, 0, 10000.0f, 1) != WR_OK) return -1;
    if (y2[5] != 6.0f || y2[7] != 8.0f) return -1;
    if (!feq(y2[4], 5.0f * 0.5403f - 7.0f * 0.8415f, 0.05f)) return -1;
    return 0;
}

static int ops_test_gqa(void)
{
    /* Q [1,4,2], K/V [1,2,2]: group=2 — q heads 0,1 read kv head 0 and
     * q heads 2,3 read kv head 1.  With one key the softmax is 1.0, so
     * each output equals the mapped V row. */
    float q[8], k[4];
    float v[4] = { 10, 20, 30, 40 };
    float o[8];
    for (int i = 0; i < 8; i++) q[i] = 1.0f;
    for (int i = 0; i < 4; i++) k[i] = 1.0f;
    uint32_t qsh[3] = { 1, 4, 2 };
    uint32_t ksh[3] = { 1, 2, 2 };
    wr_tensor Q, K, V, C;
    mk_tensor(&Q, WR_DTYPE_F32, 3, qsh, q);
    mk_tensor(&K, WR_DTYPE_F32, 3, ksh, k);
    mk_tensor(&V, WR_DTYPE_F32, 3, ksh, v);
    mk_tensor(&C, WR_DTYPE_F32, 3, qsh, o);
    if (wri_op_gqa_attention(&Q, &K, &V, &C, 0.0f, 0) != WR_OK) return -1;
    if (!feq(o[0], 10.0f, 0.5f) || !feq(o[1], 20.0f, 0.5f) ||
        !feq(o[2], 10.0f, 0.5f) || !feq(o[4], 30.0f, 0.5f) ||
        !feq(o[6], 30.0f, 0.5f))
        return -1;
    return 0;
}

static int ops_test_q4_0_read(void)
{
    /* One Q4_0 block, scale 0.5 (fp16 0x3800), nibbles packed low-first
     * as the format requires; element i decodes to ((i & 0xF) - 8)*0.5. */
    uint8_t raw[WR_Q4_0_BLOCK_BYTES];
    raw[0] = 0x00;
    raw[1] = 0x38;
    for (int b = 0; b < 16; b++) {
        uint8_t lo = (uint8_t)((2 * b) & 0xF);
        uint8_t hi = (uint8_t)((2 * b + 1) & 0xF);
        raw[2 + b] = (uint8_t)((hi << 4) | lo);
    }
    uint32_t shp[1] = { 32 };
    wr_tensor T;
    mk_tensor(&T, WR_DTYPE_Q4_0, 1, shp, raw);
    for (int i = 0; i < 32; i++) {
        float expect = (float)((i & 0xF) - 8) * 0.5f;
        if (!feq(wri_read_elem_f32(&T, (uint64_t)i), expect, 0.01f))
            return -1;
    }
    return 0;
}

static int ops_test_embed(void)
{
    /* F32 table. */
    float table[12];
    float out[3];
    for (int i = 0; i < 12; i++) table[i] = (float)i;
    uint32_t tsh[2] = { 4, 3 };
    uint32_t osh[1] = { 3 };
    wr_tensor T, O;
    mk_tensor(&T, WR_DTYPE_F32, 2, tsh, table);
    mk_tensor(&O, WR_DTYPE_F32, 1, osh, out);
    if (wri_op_embed(&T, 2, &O) != WR_OK) return -1;
    if (out[0] != 6.0f || out[1] != 7.0f || out[2] != 8.0f) return -1;
    if (wri_op_embed(&T, 5, &O) != WR_ERR_INVAL) return -1;

    /* Kept-quantized table (Q8_0, D == one block): the whole-row decode
     * path.  Row 0 scale 1.0 quants=i; row 1 scale 2.0 quants=i. */
    {
        uint8_t qt[2 * WR_Q8_0_BLOCK_BYTES];
        qt[0] = 0x00; qt[1] = 0x3C;                       /* d = 1.0 */
        for (int i = 0; i < 32; i++) qt[2 + i] = (uint8_t)i;
        uint8_t *r1 = qt + WR_Q8_0_BLOCK_BYTES;
        r1[0] = 0x00; r1[1] = 0x40;                       /* d = 2.0 */
        for (int i = 0; i < 32; i++) r1[2 + i] = (uint8_t)i;

        float qout[32];
        uint32_t qtsh[2] = { 2, 32 };
        uint32_t qosh[1] = { 32 };
        wr_tensor QT, QO;
        mk_tensor(&QT, WR_DTYPE_Q8_0, 2, qtsh, qt);
        mk_tensor(&QO, WR_DTYPE_F32, 1, qosh, qout);
        if (wri_op_embed(&QT, 1, &QO) != WR_OK) return -1;
        for (int i = 0; i < 32; i++)
            if (qout[i] != (float)i * 2.0f) return -1;
    }
    return 0;
}

static int ops_test_add_softcap(void)
{
    float a[2] = { 1, 2 };
    float b[2] = { 3, 4 };
    float c[2];
    uint32_t shp[1] = { 2 };
    wr_tensor A, B, C;
    mk_tensor(&A, WR_DTYPE_F32, 1, shp, a);
    mk_tensor(&B, WR_DTYPE_F32, 1, shp, b);
    mk_tensor(&C, WR_DTYPE_F32, 1, shp, c);
    if (wri_op_add(&A, &B, &C) != WR_OK) return -1;
    if (c[0] != 4.0f || c[1] != 6.0f) return -1;

    float x[3] = { 0.0f, 30.0f, -300.0f };
    float ref[3];
    for (int i = 0; i < 3; i++)
        ref[i] = 30.0f * wri_tanh(x[i] / 30.0f);
    wri_softcap(x, 3, 30.0f);
    for (int i = 0; i < 3; i++)
        if (x[i] != ref[i]) return -1;
    return 0;
}

static int ops_test_matmul_validation(void)
{
    /* An under-sized operand must be refused, never silently zero-read. */
    float a[4] = { 1, 2, 3, 4 };
    float b[6] = { 1, 0, 0, 1, 1, 1 };
    float c[3];
    uint32_t ash[2] = { 1, 4 };
    uint32_t bsh[2] = { 2, 3 };       /* claims K=2 rows: 6 < N*K = 12 */
    uint32_t csh[2] = { 1, 3 };
    wr_tensor A, B, C;
    mk_tensor(&A, WR_DTYPE_F32, 2, ash, a);
    mk_tensor(&B, WR_DTYPE_F32, 2, bsh, b);
    mk_tensor(&C, WR_DTYPE_F32, 2, csh, c);
    if (wri_op_matmul(&A, &B, &C) != WR_ERR_INVAL) return -1;
    return 0;
}

int wri_self_test_ops(void)
{
    int rms = ops_test_rmsnorm();
    int sil = ops_test_silu();
    int rop = ops_test_rope();
    int gqa = ops_test_gqa();
    int q40 = ops_test_q4_0_read();
    int emb = ops_test_embed();
    int ads = ops_test_add_softcap();
    int val = ops_test_matmul_validation();
    if (rms == 0 && sil == 0 && rop == 0 && gqa == 0 && q40 == 0 &&
        emb == 0 && ads == 0 && val == 0)
        return 0;
    wri_log_msg(0, "self-test ops: rmsnorm=%d silu=%d rope=%d gqa=%d q4_0=%d "
               "embed=%d add/softcap=%d validation=%d",
            rms, sil, rop, gqa, q40, emb, ads, val);
    return -1;
}

int wri_self_test_qmm_bit_equality(void)
{
    /* Prove the pool N-split of the K-quant matmul is BIT-EXACT to the
     * serial decode+dot, and stays so across 200 sequential
     * re-dispatches (the pattern real decode traffic produces).  A
     * third, independently accumulated double-precision oracle
     * identifies which side is wrong if they ever differ. */
    enum { QK = 256, QN = 512, REPS = 200 };
    uint8_t *wq = (uint8_t *)malloc((size_t)QN * WR_Q4_K_BLOCK_BYTES);
    if (!wq) {
        wri_log_msg(0, "self-test qmm: fixture alloc failed");
        return -1;
    }
    float act[QK];
    float c_ser[QN], c_par[QN], c_ref[QN];
    float wrow[QK], col[1];

    for (uint32_t j = 0; j < QN; j++) {
        uint8_t *blk = wq + (size_t)j * WR_Q4_K_BLOCK_BYTES;
        memset(blk, 0, WR_Q4_K_BLOCK_BYTES);
        blk[0] = 0x00; blk[1] = 0x3C;   /* d    = 1.0 */
        blk[2] = 0x00; blk[3] = 0x38;   /* dmin = 0.5 */
        for (uint32_t i = 0; i < 12; i++)
            blk[4 + i] = (uint8_t)(0x33 + i + j);
        for (uint32_t i = 0; i < 128; i++)
            blk[16 + i] = (uint8_t)((i ^ (j * 13 + 0x5A)) & 0xFF);
    }
    for (uint32_t k = 0; k < QK; k++)
        act[k] = (float)((k * 7) % 23) * 0.03f - 0.3f;

    /* Serial reference: the exact decode+dot every pool part runs. */
    wri_matmul_fn kern = matmul_ggml_kernel_snapshot();
    for (uint32_t j = 0; j < QN; j++) {
        wri_dequant_row_q4_k(wq + (size_t)j * WR_Q4_K_BLOCK_BYTES, wrow, QK);
        kern(act, wrow, col, 1, QK, 1);
        c_ser[j] = col[0];
    }

    /* Independent oracle: per-element decoder + double accumulator.  A
     * tolerance is intentional — SIMD/FMA rounding need not match a
     * double reference bit-for-bit, while a missing/zeroed range is
     * orders of magnitude outside it. */
    for (uint32_t j = 0; j < QN; j++) {
        double sum = 0.0;
        const uint8_t *wj = wq + (size_t)j * WR_Q4_K_BLOCK_BYTES;
        for (uint32_t k = 0; k < QK; k++)
            sum += (double)act[k] * (double)wri_dequant_q4_k_elem(wj, k);
        c_ref[j] = (float)sum;
        float d = c_ser[j] - c_ref[j];
        if (d < 0.0f) d = -d;
        float mag = c_ref[j];
        if (mag < 0.0f) mag = -mag;
        if (d > 0.001f + mag * 0.0001f) {
            wri_log_msg(0, "self-test qmm: serial diverges from oracle at j=%u",
                    (unsigned)j);
            free(wq);
            return -1;
        }
    }

    /* Op-level dispatch (N >= the parallel threshold, so with a live
     * engine pool this exercises the N-split), 200 re-dispatches. */
    wr_tensor A, B, C;
    uint32_t ash[2] = { 1, QK };
    uint32_t bsh[2] = { QK, QN };      /* logical [K,N]; data W[out][in] */
    uint32_t csh[2] = { 1, QN };
    mk_tensor(&A, WR_DTYPE_F32, 2, ash, act);
    mk_tensor(&B, WR_DTYPE_Q4_K, 2, bsh, wq);
    mk_tensor(&C, WR_DTYPE_F32, 2, csh, c_par);
    B.flags |= WR_TENSOR_GGML_WEIGHT;

    for (int rep = 0; rep < REPS; rep++) {
        memset(c_par, 0, sizeof(c_par));
        if (wri_op_matmul(&A, &B, &C) != WR_OK) {
            wri_log_msg(0, "self-test qmm: matmul error at rep %d", rep);
            free(wq);
            return -1;
        }
        if (memcmp(c_ser, c_par, sizeof(c_ser)) != 0) {
            for (uint32_t j = 0; j < QN; j++) {
                if (c_ser[j] != c_par[j]) {
                    wri_log_msg(0, "self-test qmm: rep %d j=%u parallel != serial",
                            rep, (unsigned)j);
                    break;
                }
            }
            free(wq);
            return -1;
        }
    }

    free(wq);
    return 0;
}

int wri_self_test_f32_ggml_nsplit(void)
{
    /* The dimensions cross both the pool threshold and SIMD tail paths;
     * N is intentionally not a multiple of a typical worker count. */
    enum { K = 259, N = WR_PARALLEL_THRESHOLD + 13 };
    float *a = (float *)malloc((size_t)K * sizeof(float));
    float *b = (float *)malloc((size_t)N * K * sizeof(float));
    float *serial = (float *)malloc((size_t)N * sizeof(float));
    float *split  = (float *)malloc((size_t)N * sizeof(float));
    if (!a || !b || !serial || !split) {
        free(a); free(b); free(serial); free(split);
        wri_log_msg(0, "self-test f32 nsplit: fixture alloc failed");
        return -1;
    }

    for (uint32_t k = 0; k < K; k++)
        a[k] = (float)((k * 17u) % 41u) * 0.025f - 0.5f;
    for (uint32_t j = 0; j < N; j++)
        for (uint32_t k = 0; k < K; k++)
            b[(size_t)j * K + k] =
                (float)(((j + 3u) * 29u + k * 11u) % 67u) * 0.015f - 0.4f;

    wri_matmul_fn kernel = matmul_ggml_kernel_snapshot();
    kernel(a, b, serial, 1, K, N);

    wr_tensor A, B, C;
    uint32_t ash[2] = { 1, K };
    uint32_t bsh[2] = { K, N };
    uint32_t csh[2] = { 1, N };
    mk_tensor(&A, WR_DTYPE_F32, 2, ash, a);
    mk_tensor(&B, WR_DTYPE_F32, 2, bsh, b);
    mk_tensor(&C, WR_DTYPE_F32, 2, csh, split);
    B.flags |= WR_TENSOR_GGML_WEIGHT;
    memset(split, 0xA5, (size_t)N * sizeof(float));

    uint64_t before = __atomic_load_n(
        &wri_g_counters[WR_CTR_MATMUL_PAR_N], __ATOMIC_RELAXED);
    if (wri_op_matmul(&A, &B, &C) != WR_OK) {
        free(a); free(b); free(serial); free(split);
        wri_log_msg(0, "self-test f32 nsplit: op failed");
        return -1;
    }
    uint64_t after = __atomic_load_n(
        &wri_g_counters[WR_CTR_MATMUL_PAR_N], __ATOMIC_RELAXED);

    int fail = memcmp(serial, split, (size_t)N * sizeof(float)) != 0;
    wr_pool *pool = engine_pool();
    if (pool && wr_pool_size(pool) > 1 && after != before + 1u)
        fail = 1;  /* counter witness: the parallel branch really ran */
    if (serial[0] == 0.0f && serial[N / 2] == 0.0f && serial[N - 1] == 0.0f)
        fail = 1;  /* fixture witness: equality cannot pass on zero output */

    if (fail)
        wri_log_msg(0, "self-test f32 nsplit: split is not bit-exact/witnessed");
    free(a); free(b); free(serial); free(split);
    return fail ? -1 : 0;
}
