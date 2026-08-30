/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * unit_tests.c — wayruntime unit test runner.
 *
 * Two layers:
 *   1. CONTRACT TESTS (below, live now): pin the numeric constants and
 *      struct-layout invariants the whole port hangs on.  These compile
 *      and run against the headers plus the smallest implementation
 *      surface and must never be weakened.
 *   2. PORTED GOLDEN SELF-TESTS: the origin engine's numeric self-tests
 *      (quant/dequant, matmul SIMD-vs-scalar oracle, softmax, flash
 *      attention, fused ops, the 200-rep quantized-matmul re-dispatch
 *      bit-equality, the tiny-model decode fixture, batched-vs-serial
 *      step equality) arrive with their modules and are wired into
 *      run_all() via the wri_self_test_* entry points in internal.h.
 *
 * Exit code 0 = all pass.  Output: one line per test, TAP-ish.
 */
#include <stdio.h>
#include <string.h>

#include <wayruntime/wayruntime.h>

#include "core/internal.h"
#include "core/quant.h"

static int g_fail, g_run;

#define CHECK(name, cond)                                        \
    do {                                                         \
        g_run++;                                                 \
        if (cond) {                                              \
            printf("ok  %s\n", name);                            \
        } else {                                                 \
            printf("FAIL %s (%s:%d)\n", name, __FILE__, __LINE__);\
            g_fail++;                                            \
        }                                                        \
    } while (0)

/* ---- 1. dtype enum values are pinned (GGUF mapping + goldens) --------- */
static void test_dtype_values(void)
{
    CHECK("dtype.f32",  WR_DTYPE_F32  == 0);
    CHECK("dtype.f16",  WR_DTYPE_F16  == 1);
    CHECK("dtype.bf16", WR_DTYPE_BF16 == 2);
    CHECK("dtype.int8", WR_DTYPE_INT8 == 3);
    CHECK("dtype.int4", WR_DTYPE_INT4 == 4);
    CHECK("dtype.q4_0", WR_DTYPE_Q4_0 == 5);
    CHECK("dtype.q8_0", WR_DTYPE_Q8_0 == 6);
    CHECK("dtype.q4_k", WR_DTYPE_Q4_K == 7);
    CHECK("dtype.q5_k", WR_DTYPE_Q5_K == 8);
    CHECK("dtype.q6_k", WR_DTYPE_Q6_K == 9);
    CHECK("dtype.count", WR_DTYPE_COUNT == 10);
}

/* ---- 2. block-format constants (GGML byte-layout compatibility) ------ */
static void test_block_constants(void)
{
    CHECK("blk.q4_0", WR_QK4_0 == 32 && WR_Q4_0_BLOCK_BYTES == 18);
    CHECK("blk.q8_0", WR_QK8_0 == 32 && WR_Q8_0_BLOCK_BYTES == 34);
    CHECK("blk.qk_k", WR_QK_K == 256);
    CHECK("blk.q4_k", WR_Q4_K_BLOCK_BYTES == 144);
    CHECK("blk.q5_k", WR_Q5_K_BLOCK_BYTES == 176);
    CHECK("blk.q6_k", WR_Q6_K_BLOCK_BYTES == 210);
    CHECK("blk.q4_1", WR_Q4_1_BLOCK_BYTES == 20);
    CHECK("blk.q5_0", WR_Q5_0_BLOCK_BYTES == 22);
    CHECK("blk.q5_1", WR_Q5_1_BLOCK_BYTES == 24);
}

/* ---- 3. sizing helpers round partial blocks up ------------------------ */
static void test_dtype_sizing(void)
{
    CHECK("size.f32",   wri_dtype_bytes_for_count(WR_DTYPE_F32, 7) == 28);
    CHECK("size.f16",   wri_dtype_bytes_for_count(WR_DTYPE_F16, 7) == 14);
    CHECK("size.q4_0",  wri_dtype_bytes_for_count(WR_DTYPE_Q4_0, 33) == 36);
    CHECK("size.q8_0",  wri_dtype_bytes_for_count(WR_DTYPE_Q8_0, 32) == 34);
    CHECK("size.q4_k",  wri_dtype_bytes_for_count(WR_DTYPE_Q4_K, 257) == 288);
    CHECK("size.q6_k",  wri_dtype_bytes_for_count(WR_DTYPE_Q6_K, 256) == 210);
    CHECK("size.fixed0", wri_dtype_size(WR_DTYPE_Q4_K) == 0);
}

/* ---- 4. op ids + flags are pinned ------------------------------------- */
static void test_op_and_flag_values(void)
{
    CHECK("op.matmul",  WR_OP_MATMUL == 0);
    CHECK("op.rmsnorm", WR_OP_RMSNORM == 16);
    CHECK("op.gqa",     WR_OP_GQA_ATTENTION == 19);
    CHECK("op.silu_mul", WR_OP_FUSED_SILU_MUL == 0x84);
    CHECK("op.gelu_mul", WR_OP_FUSED_GELU_MUL == 0x85);
    CHECK("op.rms_gemma", WR_OP_RMSNORM_GEMMA == 0x86);
    CHECK("flag.growable", WR_TENSOR_GROWABLE == (1u << 1));
    CHECK("flag.ggml",     WR_TENSOR_GGML_WEIGHT == (1u << 3));
}

/* ---- 5. f16/bf16 conversions (parity: TRUNCATING, not RTN) ------------ */
static void test_float_conversions(void)
{
    CHECK("f16.one",   wri_f16_to_f32(0x3C00) == 1.0f);
    CHECK("f16.neg2",  wri_f16_to_f32(0xC000) == -2.0f);
    CHECK("f16.zero",  wri_f16_to_f32(0x0000) == 0.0f);
    CHECK("f16.rt1",   wri_f32_to_f16(1.0f) == 0x3C00);
    CHECK("bf16.rt",   wri_bf16_to_f32(wri_f32_to_bf16(1.5f)) == 1.5f);
    /* Truncation witness: 1 + 3*2^-12 truncates DOWN to 1.0 (0x3C00);
     * round-to-nearest would give 0x3C01.  Guards the parity decision. */
    CHECK("f16.trunc", wri_f32_to_f16(1.000732421875f) == 0x3C00);
}

/* ---- 6. Q4_0 vs INT4 nibble order (do-not-unify landmine) ------------- */
static void test_nibble_order(void)
{
    /* One Q4_0 block: scale = 1.0 (fp16 0x3C00), first byte packs
     * element0 = low nibble, element1 = high nibble.  q=9 → (9-8)*1 = 1. */
    uint8_t blk[WR_Q4_0_BLOCK_BYTES] = {0};
    blk[0] = 0x00; blk[1] = 0x3C;      /* d = 1.0 */
    blk[2] = 0x29;                      /* elem0 = 9 (low), elem1 = 2 (high) */
    CHECK("q4_0.lo_first",
          wri_dequant_q4_0_elem(blk, 0) == 1.0f &&
          wri_dequant_q4_0_elem(blk, 1) == -6.0f);
}

/* ---- 7. status strings exist for every code --------------------------- */
static void test_status_strings(void)
{
    CHECK("status.ok",  strcmp(wr_status_str(WR_OK), "") != 0);
    CHECK("status.ctx", strcmp(wr_status_str(WR_ERR_CTX_FULL), "") != 0);
    CHECK("status.int", strcmp(wr_status_str(WR_ERR_INTERNAL), "") != 0);
}

/* ---- ported golden self-tests (wired as modules land) ----------------- */
static void test_ported_goldens(void)
{
    CHECK("golden.quant_dequant",  wri_self_test_quant_dequant() == 0);
    CHECK("golden.matmul_simd",    wri_self_test_matmul_simd() == 0);
    CHECK("golden.softmax",        wri_self_test_softmax() == 0);
    CHECK("golden.flash_attn",     wri_self_test_flash_attn() == 0);
    CHECK("golden.fused",          wri_self_test_fused() == 0);
    CHECK("golden.ops",            wri_self_test_ops() == 0);
    CHECK("golden.qmm_bit_equal",  wri_self_test_qmm_bit_equality() == 0);
    CHECK("golden.llm_step",       wri_self_test_llm_step() == 0);
    CHECK("golden.batch_step",     wri_self_test_batch_step() == 0);
}

int main(void)
{
    /* Engine up first: SIMD dispatch + pool are live for the goldens. */
    wr_engine *e = NULL;
    wr_status st = wr_engine_create(NULL, &e);
    CHECK("engine.create", st == WR_OK && e != NULL);

    test_dtype_values();
    test_block_constants();
    test_dtype_sizing();
    test_op_and_flag_values();
    test_float_conversions();
    test_nibble_order();
    test_status_strings();
    test_ported_goldens();

    wr_engine_destroy(e);

    printf("%d/%d passed\n", g_run - g_fail, g_run);
    return g_fail ? 1 : 0;
}
