/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * mathx.h — scalar math for the hot path.
 *
 * WR_MATH_APPROX (default ON) selects the hand-rolled approximations the
 * engine was validated with (~1e-5..1e-6 relative error): the ported
 * golden self-tests and their baked expectations assume these EXACTLY —
 * building with WR_MATH_APPROX=0 links the same names to libm and is
 * numerically fine for inference, but the bit-exact goldens then no
 * longer apply (run the tolerance-based tests only).
 *
 * The switch lives in mathx.c: the prototypes below never change, so no
 * other translation unit rebuilds differently.  The library core links
 * WITHOUT -lm when WR_MATH_APPROX is on; the CLI and tests always link
 * -lm (they may use libm freely).
 *
 * Build:  make MATH_APPROX=0  →  -DWR_MATH_APPROX=0 on mathx.c.
 */
#ifndef WR_MATHX_H
#define WR_MATHX_H

#ifndef WR_MATH_APPROX
#define WR_MATH_APPROX 1
#endif

/* Domains/edge cases (approx mode):
 *   wri_exp:  input clamped to [-88, 88] (finite output guaranteed)
 *   wri_log:  x <= 0 returns a large negative finite value, never NaN/inf
 *   wri_sqrt: x < 0 returns 0
 *   wri_sin/wri_cos: argument-reduced; accuracy degrades for |x| > ~1e4
 *                    (RoPE angles stay far below that at supported
 *                    contexts)
 * The libm mode follows C99 semantics instead; both modes are branchless
 * enough for the inner loops. */
float wri_exp(float x);
float wri_log(float x);
float wri_sqrt(float x);
float wri_sin(float x);
float wri_cos(float x);

/* tanh via sigmoid identity (approx mode): tanh(x) = 2*sigmoid(2x) - 1. */
float wri_tanh(float x);
float wri_sigmoid(float x);

#endif /* WR_MATHX_H */
