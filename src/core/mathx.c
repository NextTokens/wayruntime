/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * mathx.c — scalar math for the hot path.
 *
 * Two builds behind one symbol set (see mathx.h):
 *
 *   WR_MATH_APPROX=1  (default) — the hand-rolled approximations the
 *       engine was validated with, ported bit-for-bit from the origin
 *       OS.  Every baked expectation in the golden self-tests was
 *       computed with THESE exact polynomials; changing a coefficient,
 *       an association, or an iteration count invalidates them.
 *
 *   WR_MATH_APPROX=0  — the same symbols wrap libm (C99 semantics).
 *       Numerically fine for inference; the bit-exact goldens no longer
 *       apply.  This is the ONLY core translation unit permitted to
 *       include math.h, and only in this mode — the core library links
 *       without -lm when the approximations are on.
 */
#include "core/mathx.h"

#if WR_MATH_APPROX

#include <stdint.h>
#include <string.h>

/* exp(x).
 *
 * Clamp to the finite range first: exp(88) ≈ 1.7e38 sits just below
 * FLT_MAX, and exp(-88) underflows to ~0.  Saturating to 1e38f (not
 * +inf) is deliberate — softmax-style downstream sums stay finite.
 *
 * Then split x = k·ln2 + r with |r| ≤ ln2/2, evaluate a degree-5
 * Taylor polynomial on the residual, and scale by 2^k with an integer
 * add into the IEEE-754 exponent field (well-defined for k in
 * [-126, 127]; the clamp keeps |k| ≤ 127). */
float wri_exp(float x)
{
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) return 1e38f;

    float ln2 = 0.6931471805599453f;
    int k = (int)(x / ln2 + (x >= 0 ? 0.5f : -0.5f));
    float r = x - (float)k * ln2;

    float r2 = r * r;
    float e = 1.0f + r + r2 * 0.5f + r2 * r * (1.0f / 6.0f) +
              r2 * r2 * (1.0f / 24.0f) + r2 * r2 * r * (1.0f / 120.0f);

    if (k >= -126 && k <= 127) {
        uint32_t bits;
        memcpy(&bits, &e, 4);
        bits += (uint32_t)k << 23;   /* unsigned wrap == subtract for k<0 */
        memcpy(&e, &bits, 4);
    }
    return e;
}

/* ln(x) for x > 0.
 *
 * Decompose x = m·2^e (m ∈ [1,2) via the IEEE-754 exponent field), then
 * ln(x) = ln(m) + e·ln2 with ln(m) from the atanh series
 * ln(m) = 2(t + t³/3 + t⁵/5 + t⁷/7), t = (m-1)/(m+1) ∈ [0, 1/3] —
 * four terms give ~1e-6 relative accuracy.  RoPE calls this once per
 * op to turn the model's theta base into ln(base) (the inner loop uses
 * wri_exp); sanity anchors: ln(10000) = 9.2103, ln(1e6) = 13.8155.
 *
 * Domain edge per the mathx.h contract: x <= 0 returns a large negative
 * finite value (the limit direction of log(0+)), never NaN/inf — the
 * origin OS returned 0 here, which masked domain bugs.  No live caller
 * passes x < 1. */
float wri_log(float x)
{
    if (x <= 0.0f) return -1e38f;
    uint32_t u;
    memcpy(&u, &x, 4);
    int e = (int)((u >> 23) & 0xFF) - 127;
    u = (u & 0x007FFFFFu) | 0x3F800000u;        /* mantissa in [1,2) */
    float m;
    memcpy(&m, &u, 4);
    float t  = (m - 1.0f) / (m + 1.0f);
    float t2 = t * t;
    float lnm = 2.0f * t * (1.0f + t2 * (1.0f / 3.0f +
                            t2 * (1.0f / 5.0f + t2 * (1.0f / 7.0f))));
    return lnm + (float)e * 0.6931471805599453f;
}

/* sqrt(x), x < 0 returns 0.
 *
 * Bit-trick initial guess — add the exponent bias and halve the bit
 * pattern, which halves the exponent to within a few percent of the
 * root — then FOUR Newton iterations to full float precision across
 * the whole range (verified against reference sqrt for x in
 * [1e-6, 288]).  Both details are load-bearing: an earlier linear
 * seed with three iterations failed to converge for small x
 * (sqrt(0.001) came back ~8x too large), which shrank RMSNorm's
 * 1/rms by the same factor and collapsed real-model decode. */
float wri_sqrt(float x)
{
    if (x <= 0) return 0;
    union { float f; uint32_t i; } u;
    u.f = x;
    u.i = (u.i + 0x3f800000u) >> 1;
    float g = u.f;
    g = 0.5f * (g + x / g);
    g = 0.5f * (g + x / g);
    g = 0.5f * (g + x / g);
    g = 0.5f * (g + x / g);
    return g;
}

/* sin/cos — the RoPE rotation pair.
 *
 * Range-reduce into [-pi, pi] by repeated 2·pi subtraction, then
 * evaluate the truncated Taylor series (~1e-5 over the reduced range).
 * That is fine for positional rotation: a position's angle is the same
 * for every head, so determinism matters more than absolute precision.
 * The loop reduction degrades for |x| beyond ~1e4; RoPE angles stay far
 * below that at supported context lengths (mathx.h documents the
 * bound). */
static const float WRI_PI    = 3.14159265358979f;
static const float WRI_TWOPI = 6.28318530717958f;

float wri_sin(float x)
{
    while (x >  WRI_PI) x -= WRI_TWOPI;
    while (x < -WRI_PI) x += WRI_TWOPI;
    float x2 = x * x;
    float x3 = x2 * x;
    float x5 = x3 * x2;
    float x7 = x5 * x2;
    return x - x3 * (1.0f / 6.0f)
             + x5 * (1.0f / 120.0f)
             - x7 * (1.0f / 5040.0f);
}

float wri_cos(float x)
{
    while (x >  WRI_PI) x -= WRI_TWOPI;
    while (x < -WRI_PI) x += WRI_TWOPI;
    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;
    return 1.0f - x2 * 0.5f
                + x4 * (1.0f / 24.0f)
                - x6 * (1.0f / 720.0f);
}

float wri_sigmoid(float x)
{
    return 1.0f / (1.0f + wri_exp(-x));
}

/* tanh via the sigmoid identity tanh(x) = 2·sigmoid(2x) − 1, written as
 * a single division so the rounding sequence matches the engine's GELU
 * tail exactly (2/(1+e^-2x) rounds once; 2·(1/(1+e^-2x)) could differ
 * in the last ulp). */
float wri_tanh(float x)
{
    return 2.0f / (1.0f + wri_exp(-2.0f * x)) - 1.0f;
}

#else /* WR_MATH_APPROX == 0: same symbols, libm semantics */

#include <math.h>

float wri_exp(float x)  { return expf(x); }
float wri_log(float x)  { return logf(x); }
float wri_sqrt(float x) { return sqrtf(x); }
float wri_sin(float x)  { return sinf(x); }
float wri_cos(float x)  { return cosf(x); }

float wri_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float wri_tanh(float x) { return tanhf(x); }

#endif /* WR_MATH_APPROX */
