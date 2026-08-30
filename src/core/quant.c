/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * quant.c — dtype sizing, scalar F16/BF16 conversions, GGML block
 * dequantization and the bounds-checked element accessors.
 *
 * Ported from the origin OS's compute engine.  The block layouts are
 * byte-identical to the GGML formats (llama.cpp ggml-quants.c), so GGUF
 * tensor bytes load/mmap verbatim; the decode order of every routine
 * here is pinned by golden vectors in test/unit_tests.c.
 *
 * Two hardening changes versus the origin (both mandated by quant.h):
 *   - element capacity is derived from size_bytes for EVERY dtype,
 *     including the block-quantized ones (the origin left quantized
 *     indices unchecked);
 *   - writes to packed/quantized dtypes are dropped WITH a debug assert
 *     (the origin dropped K-quant writes silently, and supported a
 *     Q4_0 scale-reusing write path that no longer exists — weights
 *     are immutable after commit).
 */
#include "core/quant.h"

#include <string.h>

/* --------------------------------------------------------------------------
 * Sizing
 * -------------------------------------------------------------------------- */

uint32_t wri_dtype_size(wr_dtype dtype)
{
    switch (dtype) {
    case WR_DTYPE_F32:  return 4;
    case WR_DTYPE_F16:  return 2;
    case WR_DTYPE_BF16: return 2;
    case WR_DTYPE_INT8: return 1;
    case WR_DTYPE_INT4: return 1;   /* packed pair per byte; capacity math
                                     * accounts for the doubling         */
    default:            return 0;   /* block-quantized (or invalid)      */
    }
}

uint64_t wri_dtype_bytes_for_count(wr_dtype dtype, uint64_t n_elements)
{
    switch (dtype) {
    case WR_DTYPE_F32:  return n_elements * 4;
    case WR_DTYPE_F16:  return n_elements * 2;
    case WR_DTYPE_BF16: return n_elements * 2;
    case WR_DTYPE_INT8: return n_elements;
    case WR_DTYPE_INT4: return (n_elements + 1) / 2;
    case WR_DTYPE_Q4_0:
        return ((n_elements + WR_QK4_0 - 1) / WR_QK4_0) * WR_Q4_0_BLOCK_BYTES;
    case WR_DTYPE_Q8_0:
        return ((n_elements + WR_QK8_0 - 1) / WR_QK8_0) * WR_Q8_0_BLOCK_BYTES;
    case WR_DTYPE_Q4_K:
        return ((n_elements + WR_QK_K - 1) / WR_QK_K) * WR_Q4_K_BLOCK_BYTES;
    case WR_DTYPE_Q5_K:
        return ((n_elements + WR_QK_K - 1) / WR_QK_K) * WR_Q5_K_BLOCK_BYTES;
    case WR_DTYPE_Q6_K:
        return ((n_elements + WR_QK_K - 1) / WR_QK_K) * WR_Q6_K_BLOCK_BYTES;
    default:
        return 0;
    }
}

const char *wri_dtype_name(wr_dtype dtype)
{
    switch (dtype) {
    case WR_DTYPE_F32:  return "F32";
    case WR_DTYPE_F16:  return "F16";
    case WR_DTYPE_BF16: return "BF16";
    case WR_DTYPE_INT8: return "INT8";
    case WR_DTYPE_INT4: return "INT4";
    case WR_DTYPE_Q4_0: return "Q4_0";
    case WR_DTYPE_Q8_0: return "Q8_0";
    case WR_DTYPE_Q4_K: return "Q4_K";
    case WR_DTYPE_Q5_K: return "Q5_K";
    case WR_DTYPE_Q6_K: return "Q6_K";
    default:            return "UNKNOWN";
    }
}

/* --------------------------------------------------------------------------
 * Scalar float conversions (software; no hardware F16C dependency)
 * -------------------------------------------------------------------------- */

float wri_f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;                       /* signed zero */
        } else {
            /* Subnormal: renormalize the mantissa into the implicit-bit
             * form.  exp may wrap below zero here; the biased sum below
             * is computed mod 2^32 and lands on the correct field. */
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);   /* inf / NaN */
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &f, 4);
    return result;
}

/* TRUNCATING encode — see the quant.h contract.  Drops the low mantissa
 * bits with no rounding, flushes exponents <= 0 (F16 subnormal range)
 * to signed zero, saturates overflow to infinity.  NaN inputs collapse
 * to infinity too (mantissa is truncated before the exponent test) —
 * origin-faithful; nothing on the KV path produces NaN. */
uint16_t wri_f32_to_f16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, 4);

    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
    int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;

    if (exp <= 0) {
        return sign;
    } else if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

float wri_bf16_to_f32(uint16_t bh)
{
    uint32_t f = (uint32_t)bh << 16;
    float result;
    memcpy(&result, &f, 4);
    return result;
}

uint16_t wri_f32_to_bf16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);   /* truncate: keep the high half */
}

/* --------------------------------------------------------------------------
 * INT4 nibble unpack (engine-native packed format — NOT Q4_0)
 *
 * Even element index = HIGH nibble of the byte, odd = LOW nibble, and
 * the value is offset-binary (nibble − 8 → [−8, 7]).  This is the
 * OPPOSITE nibble order to Q4_0 (GGML file format, even = low) — see
 * the do-not-unify note in internal.h.
 * -------------------------------------------------------------------------- */

static inline int8_t wri_int4_unpack_hi(uint8_t packed)
{
    return (int8_t)((packed >> 4) - 8);
}

static inline int8_t wri_int4_unpack_lo(uint8_t packed)
{
    return (int8_t)((packed & 0xF) - 8);
}

/* --------------------------------------------------------------------------
 * Per-element block dequantization
 *
 * `block` points at one block; idx is the element index inside it.
 * No bounds checks — innermost primitives (the accessors and row
 * decoders below do the checking).
 * -------------------------------------------------------------------------- */

/* Q4_0: [2 B fp16 scale | 16 B nibbles].  Element 2k = LOW nibble of
 * byte k, 2k+1 = HIGH nibble; value = (q − 8)·scale. */
float wri_dequant_q4_0_elem(const uint8_t *block, uint32_t idx)
{
    uint16_t scale_h;
    memcpy(&scale_h, block, 2);
    float scale = wri_f16_to_f32(scale_h);
    uint8_t byte = block[2 + idx / 2];
    int q = (idx & 1) ? (byte >> 4) : (byte & 0xF);
    return ((float)q - 8.0f) * scale;
}

/* Q8_0: [2 B fp16 scale | 32 × int8]; value = q·scale. */
float wri_dequant_q8_0_elem(const uint8_t *block, uint32_t idx)
{
    uint16_t scale_h;
    memcpy(&scale_h, block, 2);
    float scale = wri_f16_to_f32(scale_h);
    int8_t q = (int8_t)block[2 + idx];
    return (float)q * scale;
}

/* Q4_K and Q5_K share the 6+6-bit packed scale/min encoding: 12 bytes
 * hold eight (scale, min) pairs for the eight 32-element sub-blocks. */
static inline void q_k_unpack_scale_min(const uint8_t *scales, uint32_t j,
                                        uint8_t *sc_out, uint8_t *m_out)
{
    if (j < 4) {
        *sc_out = (uint8_t)(scales[j]   & 63);
        *m_out  = (uint8_t)(scales[j+4] & 63);
    } else {
        *sc_out = (uint8_t)((scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4));
        *m_out  = (uint8_t)((scales[j+4] >> 4)   | ((scales[j  ] >> 6) << 4));
    }
}

/* Q4_K super-block: [fp16 d | fp16 dmin | 12 B scales | 128 B nibbles].
 *
 * Standard GGUF element order (matches llama.cpp dequantize_row_q4_K):
 * a 64-element span uses 32 bytes — the first 32 elements are the LOW
 * nibbles of those bytes, the next 32 the HIGH nibbles.  So sub-blocks
 * j and j+1 SHARE the 32-byte chunk qs[(j>>1)*32 ..]; even j reads the
 * low nibble, odd j the high.  value = d·sc·q − dmin·m. */
float wri_dequant_q4_k_elem(const uint8_t *block, uint32_t idx)
{
    uint16_t dh, dmh;
    memcpy(&dh,  block + 0, 2);
    memcpy(&dmh, block + 2, 2);
    float d    = wri_f16_to_f32(dh);
    float dmin = wri_f16_to_f32(dmh);
    const uint8_t *scales = block + 4;
    const uint8_t *qs     = block + 16;

    uint32_t j  = idx / 32;
    uint32_t kk = idx % 32;
    uint8_t sc, m;
    q_k_unpack_scale_min(scales, j, &sc, &m);

    const uint8_t *qchunk = qs + (j >> 1) * 32;
    uint8_t byte = qchunk[kk];
    int q = (j & 1) ? (byte >> 4) : (byte & 0xF);
    return d * (float)sc * (float)q - dmin * (float)m;
}

/* Q5_K super-block: [fp16 d | fp16 dmin | 12 B scales | 32 B qh | 128 B qs].
 *
 * Canonical llama.cpp layout: the same low/high nibble packing as Q4_K
 * (sub-block pair shares a 32-byte qs chunk), and the 5th bit of element
 * l in sub-block `is` is bit `is` of qh[l].  value = d·sc·q − dmin·m.
 * The layout is easy to get wrong in ways that only corrupt SOME models
 * (the origin OS shipped a wrong variant once, surfacing only when a
 * Q5_K per-layer embedding was first exercised) — the golden vectors
 * pin this decode. */
float wri_dequant_q5_k_elem(const uint8_t *block, uint32_t idx)
{
    uint16_t dh, dmh;
    memcpy(&dh,  block + 0, 2);
    memcpy(&dmh, block + 2, 2);
    float d    = wri_f16_to_f32(dh);
    float dmin = wri_f16_to_f32(dmh);
    const uint8_t *scales = block + 4;
    const uint8_t *qh     = block + 16;
    const uint8_t *qs     = block + 48;

    uint32_t is = idx / 32;   /* sub-block index 0..7      */
    uint32_t l  = idx % 32;   /* position in sub-block     */
    uint8_t sc, m;
    q_k_unpack_scale_min(scales, is, &sc, &m);

    const uint8_t *qchunk = qs + (is >> 1) * 32;
    uint8_t byte = qchunk[l];
    int qlo = (is & 1) ? (byte >> 4) : (byte & 0xF);
    int qhi = (qh[l] >> is) & 1;
    int q = qlo | (qhi << 4);
    return d * (float)sc * (float)q - dmin * (float)m;
}

/* Q6_K super-block: [128 B ql | 64 B qh | 16 B int8 scales | fp16 d].
 *
 * Standard GGUF element order (matches llama.cpp dequantize_row_q6_K):
 * 256 elements = two 128-element halves; within a half, for l in 0..31
 * the four 32-element groups decode as
 *   y[l+ 0] = (ql[l]    & 0xF) | ((qh[l]>>0 & 3)<<4), scale sc[l/16 + 0]
 *   y[l+32] = (ql[l+32] & 0xF) | ((qh[l]>>2 & 3)<<4), scale sc[l/16 + 2]
 *   y[l+64] = (ql[l]    >> 4)  | ((qh[l]>>4 & 3)<<4), scale sc[l/16 + 4]
 *   y[l+96] = (ql[l+32] >> 4)  | ((qh[l]>>6 & 3)<<4), scale sc[l/16 + 6]
 * with ql += 64, qh += 32, sc += 8 per half.  value = d·sc·(q − 32). */
float wri_dequant_q6_k_elem(const uint8_t *block, uint32_t idx)
{
    const uint8_t *ql  = block + 0;
    const uint8_t *qh  = block + 128;
    const int8_t  *scs = (const int8_t *)(block + 192);
    uint16_t dh;
    memcpy(&dh, block + 208, 2);
    float d = wri_f16_to_f32(dh);

    uint32_t e     = idx;         /* 0..255 */
    uint32_t half  = e >> 7;      /* 0..1   */
    uint32_t wh    = e & 127;     /* 0..127 */
    uint32_t group = wh >> 5;     /* 0..3   */
    uint32_t l     = wh & 31;     /* 0..31  */
    const uint8_t *qlb = ql  + half * 64;
    const uint8_t *qhb = qh  + half * 32;
    const int8_t  *scb = scs + half * 8;
    uint8_t qhbyte = qhb[l];
    int qv, sc;
    switch (group) {
    case 0:  qv = (qlb[l]      & 0xF) | (((qhbyte >> 0) & 3) << 4); sc = scb[l/16 + 0]; break;
    case 1:  qv = (qlb[l + 32] & 0xF) | (((qhbyte >> 2) & 3) << 4); sc = scb[l/16 + 2]; break;
    case 2:  qv = (qlb[l]      >> 4)  | (((qhbyte >> 4) & 3) << 4); sc = scb[l/16 + 4]; break;
    default: qv = (qlb[l + 32] >> 4)  | (((qhbyte >> 6) & 3) << 4); sc = scb[l/16 + 6]; break;
    }
    return d * (float)sc * ((float)qv - 32.0f);
}

/* --------------------------------------------------------------------------
 * Whole-block decoders (internal)
 *
 * One full block per call, hoisting the fp16 header decode and the
 * per-sub-block scale/min unpack out of the inner loop.  Each is
 * BIT-IDENTICAL to block_elems × the per-element decoder above: the
 * hoisted products keep the same left-to-right association (d·sc first,
 * dmin·m first), so the float rounding sequence is unchanged.  The
 * quant-dequant golden self-test holds that equivalence.
 * -------------------------------------------------------------------------- */

static void dequant_block_q4_0(const uint8_t *block, float *out)
{
    uint16_t scale_h;
    memcpy(&scale_h, block, 2);
    float scale = wri_f16_to_f32(scale_h);
    for (uint32_t kk = 0; kk < WR_QK4_0; kk++) {
        uint8_t byte = block[2 + kk / 2];
        int q = (kk & 1) ? (byte >> 4) : (byte & 0xF);
        out[kk] = ((float)q - 8.0f) * scale;
    }
}

static void dequant_block_q8_0(const uint8_t *block, float *out)
{
    uint16_t scale_h;
    memcpy(&scale_h, block, 2);
    float scale = wri_f16_to_f32(scale_h);
    for (uint32_t kk = 0; kk < WR_QK8_0; kk++)
        out[kk] = (float)(int8_t)block[2 + kk] * scale;
}

static void dequant_block_q4_k(const uint8_t *block, float *out)
{
    uint16_t dh, dmh;
    memcpy(&dh,  block + 0, 2);
    memcpy(&dmh, block + 2, 2);
    float d    = wri_f16_to_f32(dh);
    float dmin = wri_f16_to_f32(dmh);
    const uint8_t *scales = block + 4;
    const uint8_t *qs     = block + 16;

    for (uint32_t j = 0; j < 8; j++) {
        uint8_t sc, m;
        q_k_unpack_scale_min(scales, j, &sc, &m);
        float dsc = d * (float)sc;
        float dmm = dmin * (float)m;
        const uint8_t *qchunk = qs + (j >> 1) * 32;
        int hi = (int)(j & 1);
        float *outsub = out + j * 32;

        for (uint32_t kk = 0; kk < 32; kk++) {
            uint8_t byte = qchunk[kk];
            int q = hi ? ((byte >> 4) & 0xF) : (byte & 0xF);
            outsub[kk] = dsc * (float)q - dmm;
        }
    }
}

static void dequant_block_q5_k(const uint8_t *block, float *out)
{
    uint16_t dh, dmh;
    memcpy(&dh,  block + 0, 2);
    memcpy(&dmh, block + 2, 2);
    float d    = wri_f16_to_f32(dh);
    float dmin = wri_f16_to_f32(dmh);
    const uint8_t *scales = block + 4;
    const uint8_t *qh     = block + 16;
    const uint8_t *qs     = block + 48;

    for (uint32_t is = 0; is < 8; is++) {
        uint8_t sc, m;
        q_k_unpack_scale_min(scales, is, &sc, &m);
        float dsc = d * (float)sc;
        float dmm = dmin * (float)m;
        const uint8_t *qchunk = qs + (is >> 1) * 32;
        int hi = (int)(is & 1);
        float *outsub = out + is * 32;

        for (uint32_t l = 0; l < 32; l++) {
            uint8_t byte = qchunk[l];
            int qlo = hi ? (byte >> 4) : (byte & 0xF);
            int q = qlo | (((qh[l] >> is) & 1) << 4);
            outsub[l] = dsc * (float)q - dmm;
        }
    }
}

static void dequant_block_q6_k(const uint8_t *block, float *out)
{
    const uint8_t *ql  = block + 0;
    const uint8_t *qh  = block + 128;
    const int8_t  *scs = (const int8_t *)(block + 192);
    uint16_t dh;
    memcpy(&dh, block + 208, 2);
    float d = wri_f16_to_f32(dh);
    for (uint32_t half = 0; half < 2; half++) {
        const uint8_t *qlb = ql  + half * 64;
        const uint8_t *qhb = qh  + half * 32;
        const int8_t  *scb = scs + half * 8;
        for (uint32_t l = 0; l < 32; l++) {
            uint8_t qhbyte = qhb[l];
            int s0 = scb[l / 16 + 0], s2 = scb[l / 16 + 2];
            int s4 = scb[l / 16 + 4], s6 = scb[l / 16 + 6];
            int q0 = (qlb[l]      & 0xF) | (((qhbyte >> 0) & 3) << 4);
            int q1 = (qlb[l + 32] & 0xF) | (((qhbyte >> 2) & 3) << 4);
            int q2 = (qlb[l]      >> 4)  | (((qhbyte >> 4) & 3) << 4);
            int q3 = (qlb[l + 32] >> 4)  | (((qhbyte >> 6) & 3) << 4);
            uint32_t eb = half * 128 + l;   /* e = half*128 + group*32 + l */
            out[eb + 0]  = d * (float)s0 * ((float)q0 - 32.0f);
            out[eb + 32] = d * (float)s2 * ((float)q1 - 32.0f);
            out[eb + 64] = d * (float)s4 * ((float)q2 - 32.0f);
            out[eb + 96] = d * (float)s6 * ((float)q3 - 32.0f);
        }
    }
}

/* --------------------------------------------------------------------------
 * Whole-row decoders
 *
 * Full blocks go through the hoisted block decoders; a trailing partial
 * block (legal only for the final row of a tensor) decodes per-element.
 * -------------------------------------------------------------------------- */

void wri_dequant_row_q4_0(const void *blocks, float *out, uint64_t n)
{
    const uint8_t *b = (const uint8_t *)blocks;
    uint64_t nb = n / WR_QK4_0;
    for (uint64_t bi = 0; bi < nb; bi++)
        dequant_block_q4_0(b + bi * WR_Q4_0_BLOCK_BYTES, out + bi * WR_QK4_0);
    for (uint64_t i = nb * WR_QK4_0; i < n; i++)
        out[i] = wri_dequant_q4_0_elem(b + (i / WR_QK4_0) * WR_Q4_0_BLOCK_BYTES,
                                       (uint32_t)(i % WR_QK4_0));
}

void wri_dequant_row_q8_0(const void *blocks, float *out, uint64_t n)
{
    const uint8_t *b = (const uint8_t *)blocks;
    uint64_t nb = n / WR_QK8_0;
    for (uint64_t bi = 0; bi < nb; bi++)
        dequant_block_q8_0(b + bi * WR_Q8_0_BLOCK_BYTES, out + bi * WR_QK8_0);
    for (uint64_t i = nb * WR_QK8_0; i < n; i++)
        out[i] = wri_dequant_q8_0_elem(b + (i / WR_QK8_0) * WR_Q8_0_BLOCK_BYTES,
                                       (uint32_t)(i % WR_QK8_0));
}

void wri_dequant_row_q4_k(const void *blocks, float *out, uint64_t n)
{
    const uint8_t *b = (const uint8_t *)blocks;
    uint64_t nb = n / WR_QK_K;
    for (uint64_t bi = 0; bi < nb; bi++)
        dequant_block_q4_k(b + bi * WR_Q4_K_BLOCK_BYTES, out + bi * WR_QK_K);
    for (uint64_t i = nb * WR_QK_K; i < n; i++)
        out[i] = wri_dequant_q4_k_elem(b + (i / WR_QK_K) * WR_Q4_K_BLOCK_BYTES,
                                       (uint32_t)(i % WR_QK_K));
}

void wri_dequant_row_q5_k(const void *blocks, float *out, uint64_t n)
{
    const uint8_t *b = (const uint8_t *)blocks;
    uint64_t nb = n / WR_QK_K;
    for (uint64_t bi = 0; bi < nb; bi++)
        dequant_block_q5_k(b + bi * WR_Q5_K_BLOCK_BYTES, out + bi * WR_QK_K);
    for (uint64_t i = nb * WR_QK_K; i < n; i++)
        out[i] = wri_dequant_q5_k_elem(b + (i / WR_QK_K) * WR_Q5_K_BLOCK_BYTES,
                                       (uint32_t)(i % WR_QK_K));
}

void wri_dequant_row_q6_k(const void *blocks, float *out, uint64_t n)
{
    const uint8_t *b = (const uint8_t *)blocks;
    uint64_t nb = n / WR_QK_K;
    for (uint64_t bi = 0; bi < nb; bi++)
        dequant_block_q6_k(b + bi * WR_Q6_K_BLOCK_BYTES, out + bi * WR_QK_K);
    for (uint64_t i = nb * WR_QK_K; i < n; i++)
        out[i] = wri_dequant_q6_k_elem(b + (i / WR_QK_K) * WR_Q6_K_BLOCK_BYTES,
                                       (uint32_t)(i % WR_QK_K));
}

int wri_dequant_row(wr_dtype dtype, const void *blocks, float *out, uint64_t n)
{
    switch (dtype) {
    case WR_DTYPE_Q4_0: wri_dequant_row_q4_0(blocks, out, n); return WR_OK;
    case WR_DTYPE_Q8_0: wri_dequant_row_q8_0(blocks, out, n); return WR_OK;
    case WR_DTYPE_Q4_K: wri_dequant_row_q4_k(blocks, out, n); return WR_OK;
    case WR_DTYPE_Q5_K: wri_dequant_row_q5_k(blocks, out, n); return WR_OK;
    case WR_DTYPE_Q6_K: wri_dequant_row_q6_k(blocks, out, n); return WR_OK;
    default:            return WR_ERR_UNSUPPORTED;
    }
}

/* --------------------------------------------------------------------------
 * Element accessors — the bounds contract (quant.h)
 *
 * Capacity derives from size_bytes for EVERY dtype.  The origin engine
 * returned capacity 0 ("unchecked") for the block-quantized dtypes;
 * that hole is closed here: a GGUF is untrusted input, and after the
 * loader's structural validation these accessors are the last line
 * against a bad element index.
 * -------------------------------------------------------------------------- */

uint64_t wri_tensor_elem_capacity(const wr_tensor *t)
{
    switch ((wr_dtype)t->dtype) {
    case WR_DTYPE_F32:  return t->size_bytes / 4;
    case WR_DTYPE_F16:
    case WR_DTYPE_BF16: return t->size_bytes / 2;
    case WR_DTYPE_INT8: return t->size_bytes;
    case WR_DTYPE_INT4: return t->size_bytes * 2;   /* two per byte */
    case WR_DTYPE_Q4_0:
        return (t->size_bytes / WR_Q4_0_BLOCK_BYTES) * WR_QK4_0;
    case WR_DTYPE_Q8_0:
        return (t->size_bytes / WR_Q8_0_BLOCK_BYTES) * WR_QK8_0;
    case WR_DTYPE_Q4_K:
        return (t->size_bytes / WR_Q4_K_BLOCK_BYTES) * WR_QK_K;
    case WR_DTYPE_Q5_K:
        return (t->size_bytes / WR_Q5_K_BLOCK_BYTES) * WR_QK_K;
    case WR_DTYPE_Q6_K:
        return (t->size_bytes / WR_Q6_K_BLOCK_BYTES) * WR_QK_K;
    default:
        return 0;   /* invalid dtype: nothing is in bounds */
    }
}

float wri_read_elem_f32(const wr_tensor *t, uint64_t i)
{
    if (i >= wri_tensor_elem_capacity(t)) {
        /* Upstream shape validation should make this unreachable; an
         * OOB read here is a library bug, not an input error. */
        WRI_ASSERT(!"wri_read_elem_f32: index out of bounds");
        return 0.0f;
    }
    const uint8_t *data = (const uint8_t *)t->data;
    switch ((wr_dtype)t->dtype) {
    case WR_DTYPE_F32:
        return ((const float *)t->data)[i];
    case WR_DTYPE_F16:
        return wri_f16_to_f32(((const uint16_t *)t->data)[i]);
    case WR_DTYPE_BF16:
        return wri_bf16_to_f32(((const uint16_t *)t->data)[i]);
    case WR_DTYPE_INT8:
        return (float)((const int8_t *)t->data)[i];
    case WR_DTYPE_INT4: {
        /* Engine-native packing: even index = HIGH nibble. */
        uint8_t byte = data[i / 2];
        return (float)((i & 1) ? wri_int4_unpack_lo(byte)
                               : wri_int4_unpack_hi(byte));
    }
    case WR_DTYPE_Q4_0:
        return wri_dequant_q4_0_elem(
            data + (i / WR_QK4_0) * WR_Q4_0_BLOCK_BYTES,
            (uint32_t)(i % WR_QK4_0));
    case WR_DTYPE_Q8_0:
        return wri_dequant_q8_0_elem(
            data + (i / WR_QK8_0) * WR_Q8_0_BLOCK_BYTES,
            (uint32_t)(i % WR_QK8_0));
    case WR_DTYPE_Q4_K:
        return wri_dequant_q4_k_elem(
            data + (i / WR_QK_K) * WR_Q4_K_BLOCK_BYTES,
            (uint32_t)(i % WR_QK_K));
    case WR_DTYPE_Q5_K:
        return wri_dequant_q5_k_elem(
            data + (i / WR_QK_K) * WR_Q5_K_BLOCK_BYTES,
            (uint32_t)(i % WR_QK_K));
    case WR_DTYPE_Q6_K:
        return wri_dequant_q6_k_elem(
            data + (i / WR_QK_K) * WR_Q6_K_BLOCK_BYTES,
            (uint32_t)(i % WR_QK_K));
    default:
        /* Unreachable: capacity 0 rejected the index above. */
        return 0.0f;
    }
}

void wri_write_elem_f32(wr_tensor *t, uint64_t i, float val)
{
    if (i >= wri_tensor_elem_capacity(t)) {
        WRI_ASSERT(!"wri_write_elem_f32: index out of bounds");
        return;
    }
    switch ((wr_dtype)t->dtype) {
    case WR_DTYPE_F32:
        ((float *)t->data)[i] = val;
        return;
    case WR_DTYPE_F16:
        ((uint16_t *)t->data)[i] = wri_f32_to_f16(val);   /* truncating */
        return;
    case WR_DTYPE_BF16:
        ((uint16_t *)t->data)[i] = wri_f32_to_bf16(val);  /* truncating */
        return;
    case WR_DTYPE_INT8: {
        int32_t v = (int32_t)val;
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        ((int8_t *)t->data)[i] = (int8_t)v;
        return;
    }
    default:
        /* INT4 and every block-quantized dtype are read-only through
         * this interface: weights are immutable after model commit, and
         * a lossy re-quantizing write has no legitimate caller.  The
         * origin engine dropped K-quant writes SILENTLY (and kept a
         * scale-reusing Q4_0 write path with no remaining consumer) —
         * an assert makes the misuse loud in debug builds; release
         * builds drop the write. */
        WRI_ASSERT(!"wri_write_elem_f32: dtype is not writable");
        return;
    }
}
