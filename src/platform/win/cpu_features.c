/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * cpu_features.c (Windows/mingw-w64) — one-shot CPU feature probe for
 * the SIMD dispatcher.
 *
 * The bit semantics mirror the origin OS's dispatch predicates exactly:
 *   WR_CPU_FEAT_AVX2   = AVX2 AND FMA        (the AVX2 kernels use FMA)
 *   WR_CPU_FEAT_AVX512 = AVX-512F AND -512DQ (the AVX-512 kernel set)
 *   WR_CPU_FEAT_NEON   = arm64 ASIMD
 * kernels.c maps these to the WR_SIMD_* variants; keep the pairings —
 * reporting AVX2 without FMA (or F without DQ) would bind kernels the
 * host cannot run.
 *
 * mingw-w64 builds use gcc, so on x86 the probe is the same
 * __builtin_cpu_supports as the POSIX side — libgcc's feature init also
 * honors OSXSAVE/XCR0, so "supported" means usable from userland (no
 * XCR0 poking here, the OS owns extended state).  On arm64 Windows the
 * probe is IsProcessorFeaturePresent.  Anything else probes to 0 and
 * the dispatcher stays scalar.
 */
#include "platform/platform.h"

#if defined(__x86_64__) || defined(__i386__)

static uint32_t wri_probe_features(void)
{
    uint32_t f = 0;

    /* Idempotent; guarantees the libgcc CPU model is populated even if
     * the platform's startup constructors have not run yet. */
    __builtin_cpu_init();

    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        f |= WR_CPU_FEAT_AVX2;
    if (__builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512dq"))
        f |= WR_CPU_FEAT_AVX512;
    return f;
}

#elif defined(__aarch64__) || defined(_M_ARM64)

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

static uint32_t wri_probe_features(void)
{
    uint32_t f = 0;

    /* ASIMD is architecturally mandatory in ARMv8-A, and arm64 Windows
     * requires ARMv8 — but ask the OS rather than assume. */
    if (IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE))
        f |= WR_CPU_FEAT_NEON;
    return f;
}

#else

/* Unknown architecture: scalar-only by design. */
static uint32_t wri_probe_features(void)
{
    return 0;
}

#endif

/* Probe once, cache forever.  Concurrent first calls may both probe;
 * the result is identical, so the race is benign.  The ready flag is
 * published with release/acquire so the cached value is visible. */
static uint32_t wri_g_features;
static int      wri_g_features_ready;

uint32_t wr_cpu_features(void)
{
    uint32_t f;

    if (__atomic_load_n(&wri_g_features_ready, __ATOMIC_ACQUIRE))
        return __atomic_load_n(&wri_g_features, __ATOMIC_RELAXED);

    f = wri_probe_features();
    __atomic_store_n(&wri_g_features, f, __ATOMIC_RELAXED);
    __atomic_store_n(&wri_g_features_ready, 1, __ATOMIC_RELEASE);
    return f;
}
