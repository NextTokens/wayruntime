/* SPDX-License-Identifier: Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * check_api.c — SDK-boundary gate (`make check-api`).
 *
 * Compiled with -std=c11 -pedantic -Wall -Wextra -Werror and -Iinclude
 * ONLY: proves the public header is self-contained (stdint/stddef only),
 * C11-clean, and that every public type and function is declared.  This
 * file is compiled, never linked.
 */
#include <wayruntime/wayruntime.h>

/* Instantiate every public struct so incomplete types fail here. */
static const wr_engine_config    g_cfg;
static const wr_model_params     g_mp;
static const wr_model_info       g_mi;
static const wr_session_params   g_sp;
static const wr_sample_params    g_smp;
static const wr_generate_params  g_gp;
static const wr_generate_result  g_gr;

/* Reference every public function; the count keeps them all "used".
 * Generic-function-pointer casts are pedantic-legal (the pointers are
 * never called through the generic type). */
typedef void (*wr_genfn)(void);

unsigned long long wr_api_surface(void);
unsigned long long wr_api_surface(void)
{
    const wr_genfn fns[] = {
        (wr_genfn)wr_version,
        (wr_genfn)wr_status_str,
        (wr_genfn)wr_engine_create,
        (wr_genfn)wr_engine_destroy,
        (wr_genfn)wr_engine_simd_variant,
        (wr_genfn)wr_engine_thread_count,
        (wr_genfn)wr_engine_set_simd,
        (wr_genfn)wr_engine_counters,
        (wr_genfn)wr_model_load,
        (wr_genfn)wr_model_free,
        (wr_genfn)wr_model_get_info,
        (wr_genfn)wr_tokenizer_from_model,
        (wr_genfn)wr_tokenizer_free,
        (wr_genfn)wr_tokenize,
        (wr_genfn)wr_tokenize_ex,
        (wr_genfn)wr_detokenize,
        (wr_genfn)wr_token_piece,
        (wr_genfn)wr_token_id,
        (wr_genfn)wr_token_eos,
        (wr_genfn)wr_token_bos,
        (wr_genfn)wr_session_create,
        (wr_genfn)wr_session_destroy,
        (wr_genfn)wr_prefill,
        (wr_genfn)wr_step,
        (wr_genfn)wr_batch_step,
        (wr_genfn)wr_session_pos,
        (wr_genfn)wr_session_max_context,
        (wr_genfn)wr_session_status,
        (wr_genfn)wr_sample_params_default,
        (wr_genfn)wr_sampler_create,
        (wr_genfn)wr_sampler_free,
        (wr_genfn)wr_sampler_reset,
        (wr_genfn)wr_sample,
        (wr_genfn)wr_sampler_set_mask,
        (wr_genfn)wr_generate,
        (wr_genfn)wr_free,
    };
    const void *objs[] = { &g_cfg, &g_mp, &g_mi, &g_sp, &g_smp, &g_gp, &g_gr };
    return (unsigned long long)(sizeof(fns) / sizeof(fns[0]))
         + (unsigned long long)(sizeof(objs) / sizeof(objs[0]))
         + (unsigned long long)(objs[0] != (const void *)0);
}

/* Compile-time checks on ABI-stable constants. */
_Static_assert(WR_OK == 0, "WR_OK must be 0");
_Static_assert(WR_ERR_CTX_FULL < 0, "errors are negative");
_Static_assert(WR_COUNTER_COUNT == 12, "counters ABI is 12 slots");
_Static_assert(WR_BATCH_MAX == 16, "batch cap is 16");
_Static_assert(WR_MMAP_DISABLED == -1 && WR_MMAP_AUTO == 0 &&
               WR_MMAP_ENABLED == 1,
               "mmap policy ids are stable ABI");
_Static_assert(WR_SIMD_SCALAR == 0 && WR_SIMD_AVX2 == 1 &&
               WR_SIMD_AVX512 == 2 && WR_SIMD_NEON == 3,
               "SIMD variant ids are stable ABI");
