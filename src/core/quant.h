/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * quant.h — dtype sizing, float conversions, block dequantization and the
 * bounds-checked element accessors.
 *
 * Block layouts are byte-identical to the GGML formats (llama.cpp
 * ggml-quants.c); the golden vectors in test/unit_tests.c hold that line.
 */
#ifndef WR_QUANT_H
#define WR_QUANT_H

#include "core/internal.h"

/* --------------------------------------------------------------------------
 * Sizing
 * -------------------------------------------------------------------------- */

/* Fixed per-element byte size, or 0 for block-quantized dtypes (use
 * wri_dtype_bytes_for_count — the two-shot API forces quantized callers
 * to think in whole blocks).  INT4 reports 1 (packed pair per byte; the
 * capacity math accounts for it separately). */
uint32_t wri_dtype_size(wr_dtype dtype);

/* Total storage bytes for n_elements of dtype, rounding partial blocks
 * up.  Defined for every wr_dtype. */
uint64_t wri_dtype_bytes_for_count(wr_dtype dtype, uint64_t n_elements);

/* Static dtype name ("Q4_K", "F16", ...). */
const char *wri_dtype_name(wr_dtype dtype);

/* --------------------------------------------------------------------------
 * Scalar float conversions
 * -------------------------------------------------------------------------- */

float    wri_f16_to_f32(uint16_t h);

/* TRUNCATING conversion (round toward zero on the dropped mantissa bits,
 * no round-to-nearest-even).  This matches the origin engine bit-for-bit
 * and therefore the ported KV-cache goldens.  It deliberately does NOT
 * match hardware F16 rounding; if that trade is ever revisited, every
 * F16 golden regenerates. */
uint16_t wri_f32_to_f16(float f);

float    wri_bf16_to_f32(uint16_t bh);
uint16_t wri_f32_to_bf16(float f);   /* truncating (drop low 16 bits) */

/* --------------------------------------------------------------------------
 * Block dequantization
 *
 * `block` points at ONE block (WR_Q*_BLOCK_BYTES bytes); idx is the
 * element index within the block.  No bounds checks here — these are the
 * innermost primitives; callers guarantee the block pointer (the element
 * accessors below and the row decoders do the checking).
 * -------------------------------------------------------------------------- */

float wri_dequant_q4_0_elem(const uint8_t *block, uint32_t idx); /* idx < 32  */
float wri_dequant_q8_0_elem(const uint8_t *block, uint32_t idx); /* idx < 32  */
float wri_dequant_q4_k_elem(const uint8_t *block, uint32_t idx); /* idx < 256 */
float wri_dequant_q5_k_elem(const uint8_t *block, uint32_t idx); /* idx < 256 */
float wri_dequant_q6_k_elem(const uint8_t *block, uint32_t idx); /* idx < 256 */

/* Whole-row decode: out[0..n) from consecutive blocks at `blocks`.
 * n need not be a block multiple only for the FINAL row of a tensor;
 * weight-row decode in the quantized matmul always passes whole rows.
 * These match llama.cpp's dequantize_row_* outputs exactly. */
void wri_dequant_row_q4_0(const void *blocks, float *out, uint64_t n);
void wri_dequant_row_q8_0(const void *blocks, float *out, uint64_t n);
void wri_dequant_row_q4_k(const void *blocks, float *out, uint64_t n);
void wri_dequant_row_q5_k(const void *blocks, float *out, uint64_t n);
void wri_dequant_row_q6_k(const void *blocks, float *out, uint64_t n);

/* Dispatch on dtype (quantized dtypes only). WR_ERR_UNSUPPORTED otherwise. */
int wri_dequant_row(wr_dtype dtype, const void *blocks, float *out,
                    uint64_t n);

/* --------------------------------------------------------------------------
 * Element accessors — THE bounds contract
 *
 * Capacity is derived from t->size_bytes for EVERY dtype, including the
 * block-quantized ones (the origin left quantized capacity unchecked;
 * that hole is closed — a GGUF is untrusted input and these accessors
 * are the last line):
 *   fixed-stride: cap = size_bytes / elem_size   (INT4: size_bytes * 2)
 *   block-quant:  cap = (size_bytes / block_bytes) * block_elems
 *
 *   read  OOB  -> returns 0.0f (defensive; shape validation upstream
 *                should have rejected the op — WRI_ASSERT in debug)
 *   write OOB  -> dropped + WRI_ASSERT in debug
 *   write to a block-quantized dtype -> unsupported by design (weights
 *                are immutable); dropped + WRI_ASSERT — the origin
 *                dropped these SILENTLY, which hid real bugs.
 * -------------------------------------------------------------------------- */

/* Element capacity of t under the rules above. */
uint64_t wri_tensor_elem_capacity(const wr_tensor *t);

float wri_read_elem_f32(const wr_tensor *t, uint64_t i);
void  wri_write_elem_f32(wr_tensor *t, uint64_t i, float val);

#endif /* WR_QUANT_H */
