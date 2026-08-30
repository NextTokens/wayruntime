/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * gguf.c — hardened GGUF container parser.
 *
 * This file is the library's trust boundary: every byte of a GGUF file
 * is attacker-controlled until it passes the V1-V6 validation pass
 * documented in gguf.h.  The rules of this file:
 *
 *   - Every length, count, offset and type id read from the file is
 *     checked BEFORE it is used.  The parser tracks a logical cursor
 *     against the fstat'd file size; no read or skip may move past EOF,
 *     so no uint64 arithmetic on untrusted lengths can wrap.
 *   - Every validation failure is a distinct, logged, honest error.
 *     There are no partial-success states: wri_gguf_open either returns
 *     a fully validated handle or frees everything and returns a
 *     negative wr_status.
 *   - Multi-byte integers are assembled byte-at-a-time little-endian.
 *     This is deliberate (it decouples parsing from host struct layout
 *     and alignment) — do not "optimize" it into raw struct reads.
 *
 * The parser descends from the origin OS's streaming GGUF reader, which
 * hand-rolled a 64 KB userland read buffer over raw syscalls.  Hosted
 * stdio buffering makes that apparatus redundant, so file access here is
 * plain FILE* + wr_fseek64/wr_file_size (platform.h).  The hardening —
 * version gate, EOF/overlap bounds, pow2 alignment, tensor-count hard
 * limit, cursor-wrap guards, BF16 sizing, float metadata kept as float,
 * dynamic "<architecture>.<suffix>" key matching — is new in this port.
 */

/* wr_fseek64 (platform.h) expands to fseeko/off_t on POSIX, which strict
 * -std=c11 hides unless a feature-test macro is visible before the first
 * system include.  Must precede every #include in this file. */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _FILE_OFFSET_BITS 64
#endif

#include "core/gguf.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * GGUF metadata value type ids (wire numbering — never renumber)
 * -------------------------------------------------------------------------- */

#define WRI_GGUF_V_UINT8    0u
#define WRI_GGUF_V_INT8     1u
#define WRI_GGUF_V_UINT16   2u
#define WRI_GGUF_V_INT16    3u
#define WRI_GGUF_V_UINT32   4u
#define WRI_GGUF_V_INT32    5u
#define WRI_GGUF_V_FLOAT32  6u
#define WRI_GGUF_V_BOOL     7u
#define WRI_GGUF_V_STRING   8u
#define WRI_GGUF_V_ARRAY    9u
#define WRI_GGUF_V_UINT64   10u
#define WRI_GGUF_V_INT64    11u
#define WRI_GGUF_V_FLOAT64  12u

#define WRI_GGUF_DEFAULT_ALIGNMENT 32u

/* Per-string byte cap for heap-captured metadata strings.  Real vocab
 * entries are tens of bytes; 64 KB rejects garbage lengths while leaving
 * generous headroom. */
#define WRI_GGUF_STR_MAX 65536u

/* Element cap for captured metadata arrays (vocab, merges, token types).
 * Covers the largest real vocabularies (262144 tokens in current
 * large-vocab models) with 2x headroom while rejecting absurd counts. */
#define WRI_GGUF_ARRAY_MAX 524288u

/* Maximum metadata array nesting depth. */
#define WRI_GGUF_ARRAY_DEPTH_MAX 4

/* --------------------------------------------------------------------------
 * Type tables
 * -------------------------------------------------------------------------- */

int wri_ggml_type_to_dtype(uint32_t ggml_type)
{
    switch (ggml_type) {
    case WR_GGML_F32:  return WR_DTYPE_F32;
    case WR_GGML_F16:  return WR_DTYPE_F16;
    case WR_GGML_BF16: return WR_DTYPE_BF16;
    case WR_GGML_Q4_0: return WR_DTYPE_Q4_0;
    case WR_GGML_Q8_0: return WR_DTYPE_Q8_0;
    case WR_GGML_Q4_K: return WR_DTYPE_Q4_K;
    case WR_GGML_Q5_K: return WR_DTYPE_Q5_K;
    case WR_GGML_Q6_K: return WR_DTYPE_Q6_K;
    default:           return -1;  /* parse-only (Q4_1/Q5_0/Q5_1) and
                                    * unknown ids: the loader reports
                                    * WR_ERR_UNSUPPORTED naming the tensor
                                    * and type if the tensor is needed */
    }
}

/* Overflow-checked block-format sizing: bytes for `elements` packed into
 * blk_elems-sized blocks of blk_bytes each.  0 on uint64 overflow. */
static uint64_t wri_block_bytes(uint64_t elements, uint64_t blk_elems,
                                uint64_t blk_bytes)
{
    uint64_t blocks;
    if (elements > UINT64_MAX - (blk_elems - 1))
        return 0;
    blocks = (elements + blk_elems - 1) / blk_elems;
    if (blocks > UINT64_MAX / blk_bytes)
        return 0;
    return blocks * blk_bytes;
}

uint64_t wri_ggml_type_storage_size(uint32_t ggml_type, uint64_t elements)
{
    if (elements == 0)
        return 0;

    switch (ggml_type) {
    case WR_GGML_F32:
        return (elements > UINT64_MAX / 4) ? 0 : elements * 4;
    case WR_GGML_F16:
    case WR_GGML_BF16:  /* 2 bytes/elem: high 16 bits of an f32 */
        return (elements > UINT64_MAX / 2) ? 0 : elements * 2;
    case WR_GGML_Q4_0:
        return wri_block_bytes(elements, WR_QK4_0, WR_Q4_0_BLOCK_BYTES);
    case WR_GGML_Q4_1:
        return wri_block_bytes(elements, WR_QK4_1, WR_Q4_1_BLOCK_BYTES);
    case WR_GGML_Q5_0:
        return wri_block_bytes(elements, WR_QK5_0, WR_Q5_0_BLOCK_BYTES);
    case WR_GGML_Q5_1:
        return wri_block_bytes(elements, WR_QK5_1, WR_Q5_1_BLOCK_BYTES);
    case WR_GGML_Q8_0:
        return wri_block_bytes(elements, WR_QK8_0, WR_Q8_0_BLOCK_BYTES);
    case WR_GGML_Q4_K:
        return wri_block_bytes(elements, WR_QK_K, WR_Q4_K_BLOCK_BYTES);
    case WR_GGML_Q5_K:
        return wri_block_bytes(elements, WR_QK_K, WR_Q5_K_BLOCK_BYTES);
    case WR_GGML_Q6_K:
        return wri_block_bytes(elements, WR_QK_K, WR_Q6_K_BLOCK_BYTES);
    default:
        return 0;   /* unknown / unsized wire type */
    }
}

const char *wri_ggml_type_name(uint32_t ggml_type)
{
    /* Full GGML wire-type name table for diagnostics; ids this parser
     * cannot size still get their real name in error messages.  Log
     * sites print the numeric id alongside, so ids beyond the table
     * return a plain "unknown". */
    switch (ggml_type) {
    case WR_GGML_F32:  return "F32";
    case WR_GGML_F16:  return "F16";
    case WR_GGML_Q4_0: return "Q4_0";
    case WR_GGML_Q4_1: return "Q4_1";
    case WR_GGML_Q5_0: return "Q5_0";
    case WR_GGML_Q5_1: return "Q5_1";
    case WR_GGML_Q8_0: return "Q8_0";
    case 9u:           return "Q8_1";
    case 10u:          return "Q2_K";
    case 11u:          return "Q3_K";
    case WR_GGML_Q4_K: return "Q4_K";
    case WR_GGML_Q5_K: return "Q5_K";
    case WR_GGML_Q6_K: return "Q6_K";
    case 15u:          return "Q8_K";
    case 16u:          return "IQ2_XXS";
    case 17u:          return "IQ2_XS";
    case 18u:          return "IQ3_XXS";
    case 19u:          return "IQ1_S";
    case 20u:          return "IQ4_NL";
    case 21u:          return "IQ3_S";
    case 22u:          return "IQ2_S";
    case 23u:          return "IQ4_XS";
    case 24u:          return "I8";
    case 25u:          return "I16";
    case 26u:          return "I32";
    case 27u:          return "I64";
    case 28u:          return "F64";
    case 29u:          return "IQ1_M";
    case WR_GGML_BF16: return "BF16";
    default:           return "unknown";
    }
}

uint64_t wri_gguf_tensor_elements(const wr_gguf_tensor_info *t)
{
    uint64_t elems = 1;
    uint32_t i;

    if (!t || t->n_dims == 0 || t->n_dims > WR_GGUF_MAX_DIMS)
        return 0;
    for (i = 0; i < t->n_dims; i++) {
        if (t->dims[i] == 0)
            return 0;
        if (elems > UINT64_MAX / t->dims[i])
            return 0;   /* element count would overflow uint64 */
        elems *= t->dims[i];
    }
    return elems;
}

/* --------------------------------------------------------------------------
 * Bounded reader
 *
 * All parsing goes through this cursor.  Invariant: cur <= size at all
 * times, and no primitive advances unless the full advance fits — which
 * is exactly the V4 guarantee (a crafted u64 length cannot wrap the
 * cursor or run past EOF).
 * -------------------------------------------------------------------------- */

typedef struct wri_gguf_rd {
    FILE    *f;
    uint64_t size;   /* fstat'd file size */
    uint64_t cur;    /* logical cursor == stdio position */
} wri_gguf_rd;

static int wri_rd_bytes(wri_gguf_rd *rd, void *dst, uint64_t n)
{
    if (n > rd->size - rd->cur) {
        wri_log_msg(0, "gguf: truncated: %" PRIu64 " bytes needed at offset %"
                   PRIu64 " but file size is %" PRIu64,
                n, rd->cur, rd->size);
        return WR_ERR_FORMAT;
    }
    if (n > 0 && fread(dst, 1, (size_t)n, rd->f) != (size_t)n) {
        wri_log_msg(0, "gguf: read failure at offset %" PRIu64, rd->cur);
        return WR_ERR_IO;
    }
    rd->cur += n;
    return WR_OK;
}

static int wri_rd_skip(wri_gguf_rd *rd, uint64_t n)
{
    if (n > rd->size - rd->cur) {
        wri_log_msg(0, "gguf: skip of %" PRIu64 " bytes at offset %" PRIu64
                   " runs past file size %" PRIu64,
                n, rd->cur, rd->size);
        return WR_ERR_FORMAT;
    }
    if (n == 0)
        return WR_OK;
    if (n <= 512) {
        /* Read-and-discard small skips: keeps stdio's buffer streaming
         * instead of forcing a buffer flush per tiny metadata seek. */
        uint8_t scratch[512];
        if (fread(scratch, 1, (size_t)n, rd->f) != (size_t)n) {
            wri_log_msg(0, "gguf: read failure at offset %" PRIu64, rd->cur);
            return WR_ERR_IO;
        }
    } else {
        if (wr_fseek64(rd->f, (int64_t)(rd->cur + n), SEEK_SET) != 0) {
            wri_log_msg(0, "gguf: seek failure to offset %" PRIu64, rd->cur + n);
            return WR_ERR_IO;
        }
    }
    rd->cur += n;
    return WR_OK;
}

static uint32_t wri_le32(const uint8_t *b)
{
    return (uint32_t)b[0] |
           ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
}

static uint64_t wri_le64(const uint8_t *b)
{
    return (uint64_t)b[0] |
           ((uint64_t)b[1] << 8) |
           ((uint64_t)b[2] << 16) |
           ((uint64_t)b[3] << 24) |
           ((uint64_t)b[4] << 32) |
           ((uint64_t)b[5] << 40) |
           ((uint64_t)b[6] << 48) |
           ((uint64_t)b[7] << 56);
}

static int wri_rd_u32(wri_gguf_rd *rd, uint32_t *out)
{
    uint8_t b[4];
    int rc = wri_rd_bytes(rd, b, 4);
    if (rc != WR_OK)
        return rc;
    *out = wri_le32(b);
    return WR_OK;
}

static int wri_rd_u64(wri_gguf_rd *rd, uint64_t *out)
{
    uint8_t b[8];
    int rc = wri_rd_bytes(rd, b, 8);
    if (rc != WR_OK)
        return rc;
    *out = wri_le64(b);
    return WR_OK;
}

/* Length-prefixed string into a fixed buffer; longer strings are
 * truncated (NUL always written) and the remainder consumed so the
 * cursor stays honest.  full_len (optional) receives the on-disk length
 * so callers can detect truncation. */
static int wri_rd_string_fixed(wri_gguf_rd *rd, char *dst, uint32_t cap,
                               uint64_t *full_len)
{
    uint64_t len = 0, copy;
    int rc = wri_rd_u64(rd, &len);
    if (rc != WR_OK)
        return rc;
    if (len > rd->size - rd->cur) {
        wri_log_msg(0, "gguf: string length %" PRIu64 " at offset %" PRIu64
                   " runs past EOF", len, rd->cur);
        return WR_ERR_FORMAT;
    }
    copy = len;
    if (copy > (uint64_t)cap - 1)
        copy = (uint64_t)cap - 1;
    if (copy > 0 && (rc = wri_rd_bytes(rd, dst, copy)) != WR_OK)
        return rc;
    dst[copy] = '\0';
    if (len > copy && (rc = wri_rd_skip(rd, len - copy)) != WR_OK)
        return rc;
    if (full_len)
        *full_len = len;
    return WR_OK;
}

static int wri_rd_string_skip(wri_gguf_rd *rd)
{
    uint64_t len = 0;
    int rc = wri_rd_u64(rd, &len);
    if (rc != WR_OK)
        return rc;
    if (len > rd->size - rd->cur) {
        wri_log_msg(0, "gguf: string length %" PRIu64 " at offset %" PRIu64
                   " runs past EOF", len, rd->cur);
        return WR_ERR_FORMAT;
    }
    return wri_rd_skip(rd, len);
}

/* Length-prefixed string into a fresh heap buffer (NUL-terminated). */
static int wri_rd_string_alloc(wri_gguf_rd *rd, char **out)
{
    uint64_t len = 0;
    char *s;
    int rc = wri_rd_u64(rd, &len);
    if (rc != WR_OK)
        return rc;
    if (len > rd->size - rd->cur) {
        wri_log_msg(0, "gguf: string length %" PRIu64 " at offset %" PRIu64
                   " runs past EOF", len, rd->cur);
        return WR_ERR_FORMAT;
    }
    if (len > WRI_GGUF_STR_MAX) {
        wri_log_msg(0, "gguf: metadata string of %" PRIu64
                   " bytes exceeds the %u-byte cap", len, WRI_GGUF_STR_MAX);
        return WR_ERR_FORMAT;
    }
    s = (char *)malloc((size_t)len + 1);
    if (!s)
        return WR_ERR_NOMEM;
    if (len > 0 && (rc = wri_rd_bytes(rd, s, len)) != WR_OK) {
        free(s);
        return rc;
    }
    s[len] = '\0';
    *out = s;
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Metadata values
 *
 * A scalar value is carried in all three representations so integer
 * fields, signed token ids and float hyperparameters each read the view
 * they need.  Float metadata STAYS float — there is no fixed-point or
 * truncate-to-integer plumbing anywhere in this parser.
 * -------------------------------------------------------------------------- */

typedef struct wri_val {
    uint64_t u;         /* raw little-endian bits, zero-extended  */
    int64_t  i;         /* sign-extended view                     */
    double   f;         /* numeric view (exact for float types)   */
    uint8_t  is_float;
} wri_val;

/* Saturating float→integer casts: a direct C cast of an out-of-range
 * float is undefined behavior, and every float here is untrusted. */
static uint64_t wri_f64_to_u64_sat(double f)
{
    if (!(f > 0.0))
        return 0;                       /* NaN and <= 0 */
    if (f >= 18446744073709551616.0)    /* 2^64 */
        return UINT64_MAX;
    return (uint64_t)f;
}

static int64_t wri_f64_to_i64_sat(double f)
{
    if (f != f)
        return 0;                       /* NaN */
    if (f >= 9223372036854775808.0)     /* 2^63 */
        return INT64_MAX;
    if (f <= -9223372036854775808.0)
        return INT64_MIN;
    return (int64_t)f;
}

/* Size in bytes of a scalar metadata value type; 0 for STRING/ARRAY and
 * unknown ids. */
static uint32_t wri_value_scalar_size(uint32_t vtype)
{
    switch (vtype) {
    case WRI_GGUF_V_UINT8:
    case WRI_GGUF_V_INT8:
    case WRI_GGUF_V_BOOL:
        return 1;
    case WRI_GGUF_V_UINT16:
    case WRI_GGUF_V_INT16:
        return 2;
    case WRI_GGUF_V_UINT32:
    case WRI_GGUF_V_INT32:
    case WRI_GGUF_V_FLOAT32:
        return 4;
    case WRI_GGUF_V_UINT64:
    case WRI_GGUF_V_INT64:
    case WRI_GGUF_V_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static int wri_rd_value_scalar(wri_gguf_rd *rd, uint32_t vtype, wri_val *out)
{
    uint8_t b[8];
    int rc;

    out->u = 0;
    out->i = 0;
    out->f = 0.0;
    out->is_float = 0;

    switch (vtype) {
    case WRI_GGUF_V_UINT8:
    case WRI_GGUF_V_BOOL:
        if ((rc = wri_rd_bytes(rd, b, 1)) != WR_OK)
            return rc;
        out->u = b[0];
        out->i = (int64_t)b[0];
        out->f = (double)b[0];
        return WR_OK;
    case WRI_GGUF_V_INT8:
        if ((rc = wri_rd_bytes(rd, b, 1)) != WR_OK)
            return rc;
        out->i = (int8_t)b[0];
        out->u = b[0];
        out->f = (double)out->i;
        return WR_OK;
    case WRI_GGUF_V_UINT16: {
        uint32_t v;
        if ((rc = wri_rd_bytes(rd, b, 2)) != WR_OK)
            return rc;
        v = (uint32_t)b[0] | ((uint32_t)b[1] << 8);
        out->u = v;
        out->i = (int64_t)v;
        out->f = (double)v;
        return WR_OK;
    }
    case WRI_GGUF_V_INT16: {
        uint32_t v;
        if ((rc = wri_rd_bytes(rd, b, 2)) != WR_OK)
            return rc;
        v = (uint32_t)b[0] | ((uint32_t)b[1] << 8);
        out->i = (int16_t)v;
        out->u = v;
        out->f = (double)out->i;
        return WR_OK;
    }
    case WRI_GGUF_V_UINT32: {
        uint32_t v;
        if ((rc = wri_rd_u32(rd, &v)) != WR_OK)
            return rc;
        out->u = v;
        out->i = (int64_t)v;
        out->f = (double)v;
        return WR_OK;
    }
    case WRI_GGUF_V_INT32: {
        uint32_t v;
        if ((rc = wri_rd_u32(rd, &v)) != WR_OK)
            return rc;
        out->i = (int32_t)v;
        out->u = v;
        out->f = (double)out->i;
        return WR_OK;
    }
    case WRI_GGUF_V_FLOAT32: {
        uint32_t v;
        float fv;
        if ((rc = wri_rd_u32(rd, &v)) != WR_OK)
            return rc;
        memcpy(&fv, &v, 4);
        out->is_float = 1;
        out->f = (double)fv;
        out->u = wri_f64_to_u64_sat(out->f);
        out->i = wri_f64_to_i64_sat(out->f);
        return WR_OK;
    }
    case WRI_GGUF_V_UINT64: {
        uint64_t v;
        if ((rc = wri_rd_u64(rd, &v)) != WR_OK)
            return rc;
        out->u = v;
        out->i = (int64_t)v;
        out->f = (double)v;
        return WR_OK;
    }
    case WRI_GGUF_V_INT64: {
        uint64_t v;
        if ((rc = wri_rd_u64(rd, &v)) != WR_OK)
            return rc;
        out->i = (int64_t)v;
        out->u = v;
        out->f = (double)out->i;
        return WR_OK;
    }
    case WRI_GGUF_V_FLOAT64: {
        uint64_t v;
        double dv;
        if ((rc = wri_rd_u64(rd, &v)) != WR_OK)
            return rc;
        memcpy(&dv, &v, 8);
        out->is_float = 1;
        out->f = dv;
        out->u = wri_f64_to_u64_sat(dv);
        out->i = wri_f64_to_i64_sat(dv);
        return WR_OK;
    }
    default:
        wri_log_msg(0, "gguf: unknown metadata value type %u at offset %" PRIu64,
                vtype, rd->cur);
        return WR_ERR_FORMAT;
    }
}

/* Skip the body of an array whose (elem_type, count) header has already
 * been read.  Scalar element arrays skip in one bounded jump; string and
 * nested-array elements are walked (each iteration consumes bytes or
 * errors, so the loops are bounded by the file size). */
static int wri_skip_array_body(wri_gguf_rd *rd, uint32_t elem_type,
                               uint64_t count, int depth)
{
    uint32_t esz = wri_value_scalar_size(elem_type);
    uint64_t k;
    int rc;

    if (esz > 0) {
        if (count > (rd->size - rd->cur) / esz) {
            wri_log_msg(0, "gguf: array count %" PRIu64
                       " exceeds the remaining file size", count);
            return WR_ERR_FORMAT;
        }
        return wri_rd_skip(rd, count * (uint64_t)esz);
    }
    if (elem_type == WRI_GGUF_V_STRING) {
        /* each string carries at least an 8-byte length prefix */
        if (count > (rd->size - rd->cur) / 8) {
            wri_log_msg(0, "gguf: string-array count %" PRIu64
                       " exceeds the remaining file size", count);
            return WR_ERR_FORMAT;
        }
        for (k = 0; k < count; k++)
            if ((rc = wri_rd_string_skip(rd)) != WR_OK)
                return rc;
        return WR_OK;
    }
    if (elem_type == WRI_GGUF_V_ARRAY) {
        if (depth + 1 >= WRI_GGUF_ARRAY_DEPTH_MAX) {
            wri_log_msg(0, "gguf: metadata array nested deeper than %d levels",
                    WRI_GGUF_ARRAY_DEPTH_MAX);
            return WR_ERR_FORMAT;
        }
        /* each nested array carries at least a 12-byte header */
        if (count > (rd->size - rd->cur) / 12) {
            wri_log_msg(0, "gguf: nested-array count %" PRIu64
                       " exceeds the remaining file size", count);
            return WR_ERR_FORMAT;
        }
        for (k = 0; k < count; k++) {
            uint32_t et2;
            uint64_t n2;
            if ((rc = wri_rd_u32(rd, &et2)) != WR_OK)
                return rc;
            if ((rc = wri_rd_u64(rd, &n2)) != WR_OK)
                return rc;
            if ((rc = wri_skip_array_body(rd, et2, n2, depth + 1)) != WR_OK)
                return rc;
        }
        return WR_OK;
    }
    wri_log_msg(0, "gguf: unknown metadata array element type %u", elem_type);
    return WR_ERR_FORMAT;
}

/* Skip one metadata value of any type. */
static int wri_skip_value(wri_gguf_rd *rd, uint32_t vtype, int depth)
{
    uint32_t esz = wri_value_scalar_size(vtype);
    if (esz > 0)
        return wri_rd_skip(rd, esz);
    if (vtype == WRI_GGUF_V_STRING)
        return wri_rd_string_skip(rd);
    if (vtype == WRI_GGUF_V_ARRAY) {
        uint32_t et;
        uint64_t n;
        int rc;
        if (depth >= WRI_GGUF_ARRAY_DEPTH_MAX) {
            wri_log_msg(0, "gguf: metadata array nested deeper than %d levels",
                    WRI_GGUF_ARRAY_DEPTH_MAX);
            return WR_ERR_FORMAT;
        }
        if ((rc = wri_rd_u32(rd, &et)) != WR_OK)
            return rc;
        if ((rc = wri_rd_u64(rd, &n)) != WR_OK)
            return rc;
        return wri_skip_array_body(rd, et, n, depth);
    }
    wri_log_msg(0, "gguf: unknown metadata value type %u at offset %" PRIu64,
            vtype, rd->cur);
    return WR_ERR_FORMAT;
}

/* Read one metadata value as a scalar field.  Scalars are taken as-is;
 * an array of scalars yields its FIRST element (some writers emit
 * per-layer arrays for hyperparameters that are uniform across layers,
 * e.g. feed_forward_length — the loader re-derives true per-layer dims
 * from the tensors themselves); strings and non-scalar arrays are
 * consumed with *got left 0 so the field stays "absent" and downstream
 * validation fails honestly on the missing hyperparameter. */
static int wri_rd_field_val(wri_gguf_rd *rd, uint32_t vtype, wri_val *out,
                            int *got)
{
    *got = 0;
    out->u = 0;
    out->i = 0;
    out->f = 0.0;
    out->is_float = 0;

    if (wri_value_scalar_size(vtype) > 0) {
        int rc = wri_rd_value_scalar(rd, vtype, out);
        if (rc == WR_OK)
            *got = 1;
        return rc;
    }
    if (vtype == WRI_GGUF_V_STRING)
        return wri_rd_string_skip(rd);
    if (vtype == WRI_GGUF_V_ARRAY) {
        uint32_t et, esz;
        uint64_t n;
        int rc;
        if ((rc = wri_rd_u32(rd, &et)) != WR_OK)
            return rc;
        if ((rc = wri_rd_u64(rd, &n)) != WR_OK)
            return rc;
        esz = wri_value_scalar_size(et);
        if (esz == 0)
            return wri_skip_array_body(rd, et, n, 0);
        if (n == 0)
            return WR_OK;
        if (n > (rd->size - rd->cur) / esz) {
            wri_log_msg(0, "gguf: array count %" PRIu64
                       " exceeds the remaining file size", n);
            return WR_ERR_FORMAT;
        }
        if ((rc = wri_rd_value_scalar(rd, et, out)) != WR_OK)
            return rc;
        *got = 1;
        return wri_rd_skip(rd, (n - 1) * (uint64_t)esz);
    }
    wri_log_msg(0, "gguf: unknown metadata value type %u at offset %" PRIu64,
            vtype, rd->cur);
    return WR_ERR_FORMAT;
}

static int wri_rd_field_u32(wri_gguf_rd *rd, uint32_t vtype, uint32_t *out)
{
    wri_val v;
    int got = 0;
    int rc = wri_rd_field_val(rd, vtype, &v, &got);
    if (rc == WR_OK && got)
        *out = v.is_float ? (uint32_t)wri_f64_to_u64_sat(v.f) : (uint32_t)v.u;
    return rc;
}

static int wri_rd_field_f32(wri_gguf_rd *rd, uint32_t vtype, float *out)
{
    wri_val v;
    int got = 0;
    int rc = wri_rd_field_val(rd, vtype, &v, &got);
    if (rc == WR_OK && got)
        *out = v.is_float ? (float)v.f : (float)v.i;
    return rc;
}

static int wri_rd_field_i32(wri_gguf_rd *rd, uint32_t vtype, int32_t *out)
{
    wri_val v;
    int got = 0;
    int rc = wri_rd_field_val(rd, vtype, &v, &got);
    if (rc == WR_OK && got)
        *out = v.is_float ? (int32_t)wri_f64_to_i64_sat(v.f) : (int32_t)v.i;
    return rc;
}

static int wri_rd_field_bool(wri_gguf_rd *rd, uint32_t vtype, uint8_t *out,
                             uint8_t *present)
{
    wri_val v;
    int got = 0;
    int rc = wri_rd_field_val(rd, vtype, &v, &got);
    if (rc == WR_OK && got) {
        *out = (uint8_t)((v.is_float ? (v.f != 0.0) : (v.u != 0)) ? 1 : 0);
        *present = 1;
    }
    return rc;
}

/* --------------------------------------------------------------------------
 * Captured metadata arrays (vocab / merges / token types)
 * -------------------------------------------------------------------------- */

/* Capture `count` length-prefixed strings into a heap char** owned by
 * the handle.  The (possibly partial) array is assigned to the handle
 * BEFORE filling, so the single cleanup path in wri_gguf_close frees
 * whatever was allocated when a later element fails. */
static int wri_capture_string_array(wri_gguf_rd *rd, uint64_t count,
                                    char ***out_arr, uint32_t *out_n,
                                    const char *keyname)
{
    char **arr;
    uint64_t k;
    int rc;

    if (count > WRI_GGUF_ARRAY_MAX) {
        wri_log_msg(0, "gguf: %s: %" PRIu64 " entries exceeds the %u cap",
                keyname, count, WRI_GGUF_ARRAY_MAX);
        return WR_ERR_FORMAT;
    }
    /* each string carries at least an 8-byte length prefix */
    if (count > (rd->size - rd->cur) / 8) {
        wri_log_msg(0, "gguf: %s: %" PRIu64
                   " entries exceeds the remaining file size", keyname, count);
        return WR_ERR_FORMAT;
    }
    if (count == 0)
        return WR_OK;   /* left NULL — treated as absent */

    arr = (char **)calloc((size_t)count, sizeof *arr);
    if (!arr)
        return WR_ERR_NOMEM;
    *out_arr = arr;
    *out_n = (uint32_t)count;

    for (k = 0; k < count; k++)
        if ((rc = wri_rd_string_alloc(rd, &arr[k])) != WR_OK)
            return rc;
    return WR_OK;
}

static int wri_rd_kv_string_array(wri_gguf_rd *rd, uint32_t vtype,
                                  char ***arr, uint32_t *count,
                                  const char *keyname)
{
    uint32_t et;
    uint64_t n;
    int rc;

    if (vtype != WRI_GGUF_V_ARRAY)
        return wri_skip_value(rd, vtype, 0);
    if ((rc = wri_rd_u32(rd, &et)) != WR_OK)
        return rc;
    if ((rc = wri_rd_u64(rd, &n)) != WR_OK)
        return rc;
    if (et != WRI_GGUF_V_STRING)
        return wri_skip_array_body(rd, et, n, 0);
    if (*arr) {
        wri_log_msg(0, "gguf: duplicate %s key", keyname);
        return WR_ERR_FORMAT;
    }
    return wri_capture_string_array(rd, n, arr, count, keyname);
}

static int wri_rd_kv_token_types(wri_gguf_rd *rd, uint32_t vtype,
                                 wr_gguf_tokenizer *tok)
{
    uint32_t et, esz;
    uint64_t n, k;
    int32_t *tt;
    int rc;

    if (vtype != WRI_GGUF_V_ARRAY)
        return wri_skip_value(rd, vtype, 0);
    if ((rc = wri_rd_u32(rd, &et)) != WR_OK)
        return rc;
    if ((rc = wri_rd_u64(rd, &n)) != WR_OK)
        return rc;
    esz = wri_value_scalar_size(et);
    if (esz == 0)
        return wri_skip_array_body(rd, et, n, 0);
    if (tok->token_type) {
        wri_log_msg(0, "gguf: duplicate tokenizer.ggml.token_type key");
        return WR_ERR_FORMAT;
    }
    if (n > WRI_GGUF_ARRAY_MAX) {
        wri_log_msg(0, "gguf: tokenizer.ggml.token_type: %" PRIu64
                   " entries exceeds the %u cap", n, WRI_GGUF_ARRAY_MAX);
        return WR_ERR_FORMAT;
    }
    if (n > (rd->size - rd->cur) / esz) {
        wri_log_msg(0, "gguf: tokenizer.ggml.token_type: %" PRIu64
                   " entries exceeds the remaining file size", n);
        return WR_ERR_FORMAT;
    }
    if (n == 0)
        return WR_OK;

    tt = (int32_t *)malloc((size_t)n * sizeof *tt);
    if (!tt)
        return WR_ERR_NOMEM;
    tok->token_type = tt;
    tok->token_type_count = (uint32_t)n;

    for (k = 0; k < n; k++) {
        wri_val v;
        if ((rc = wri_rd_value_scalar(rd, et, &v)) != WR_OK)
            return rc;
        tt[k] = (int32_t)v.i;
    }
    return WR_OK;
}

/* Per-layer SWA/global bool array.  Entries beyond WR_MAX_LAYERS are
 * consumed (cursor honesty) but cannot matter: a model with more layers
 * than WR_MAX_LAYERS is rejected by the loader's hard layer cap. */
static int wri_rd_swa_pattern(wri_gguf_rd *rd, uint32_t vtype,
                              wr_gguf_arch *a)
{
    uint32_t et, esz;
    uint64_t n, k;
    int rc;

    if (vtype != WRI_GGUF_V_ARRAY)
        return wri_skip_value(rd, vtype, 0);
    if ((rc = wri_rd_u32(rd, &et)) != WR_OK)
        return rc;
    if ((rc = wri_rd_u64(rd, &n)) != WR_OK)
        return rc;
    esz = wri_value_scalar_size(et);
    if (esz == 0)
        return wri_skip_array_body(rd, et, n, 0);
    if (n > (rd->size - rd->cur) / esz) {
        wri_log_msg(0, "gguf: sliding_window_pattern count %" PRIu64
                   " exceeds the remaining file size", n);
        return WR_ERR_FORMAT;
    }
    for (k = 0; k < n; k++) {
        wri_val v;
        if ((rc = wri_rd_value_scalar(rd, et, &v)) != WR_OK)
            return rc;
        if (k < WR_MAX_LAYERS)
            a->swa_pattern[k] =
                (uint8_t)((v.is_float ? (v.f != 0.0) : (v.u != 0)) ? 1 : 0);
    }
    a->has_swa_pattern = 1;
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * KV dispatch
 * -------------------------------------------------------------------------- */

/* Architecture-scoped keys, matched by suffix after "<architecture>.".
 * Building the match dynamically from general.architecture means every
 * architecture's rope/attention keys are honored — the origin OS matched
 * a fixed prefix list and silently dropped rope metadata for prefixes
 * not on it. */
static int wri_gguf_handle_arch_kv(wri_gguf_rd *rd, wr_gguf *g,
                                   const char *sfx, uint32_t vtype)
{
    wr_gguf_arch *a = &g->arch;

    if (strcmp(sfx, "context_length") == 0)
        return wri_rd_field_u32(rd, vtype, &a->context_length);
    if (strcmp(sfx, "embedding_length") == 0)
        return wri_rd_field_u32(rd, vtype, &a->embedding_length);
    if (strcmp(sfx, "feed_forward_length") == 0)
        return wri_rd_field_u32(rd, vtype, &a->feed_forward_length);
    if (strcmp(sfx, "block_count") == 0)
        return wri_rd_field_u32(rd, vtype, &a->block_count);
    if (strcmp(sfx, "attention.head_count") == 0)
        return wri_rd_field_u32(rd, vtype, &a->attention_head_count);
    if (strcmp(sfx, "attention.head_count_kv") == 0)
        return wri_rd_field_u32(rd, vtype, &a->attention_head_count_kv);
    if (strcmp(sfx, "attention.key_length") == 0)
        return wri_rd_field_u32(rd, vtype, &a->attention_key_length);
    if (strcmp(sfx, "attention.value_length") == 0)
        return wri_rd_field_u32(rd, vtype, &a->attention_value_length);
    if (strcmp(sfx, "attention.layer_norm_rms_epsilon") == 0)
        return wri_rd_field_f32(rd, vtype,
                                &a->attention_layer_norm_rms_epsilon);
    if (strcmp(sfx, "rope.dimension_count") == 0)
        return wri_rd_field_u32(rd, vtype, &a->rope_dimension_count);
    if (strcmp(sfx, "rope.freq_base") == 0)
        return wri_rd_field_f32(rd, vtype, &a->rope_freq_base);
    if (strcmp(sfx, "rope.freq_base_swa") == 0)
        return wri_rd_field_f32(rd, vtype, &a->rope_freq_base_swa);
    if (strcmp(sfx, "rope.dimension_count_swa") == 0)
        return wri_rd_field_u32(rd, vtype, &a->rope_dimension_count_swa);
    if (strcmp(sfx, "attention.key_length_swa") == 0)
        return wri_rd_field_u32(rd, vtype, &a->attention_key_length_swa);
    if (strcmp(sfx, "attention.value_length_swa") == 0)
        return wri_rd_field_u32(rd, vtype, &a->attention_value_length_swa);
    if (strcmp(sfx, "attention.sliding_window") == 0)
        return wri_rd_field_u32(rd, vtype, &a->attention_sliding_window);
    if (strcmp(sfx, "attention.shared_kv_layers") == 0)
        return wri_rd_field_u32(rd, vtype, &a->attention_shared_kv_layers);
    if (strcmp(sfx, "embedding_length_per_layer_input") == 0)
        return wri_rd_field_u32(rd, vtype,
                                &a->embedding_length_per_layer_input);
    if (strcmp(sfx, "final_logit_softcapping") == 0)
        return wri_rd_field_f32(rd, vtype, &a->final_logit_softcapping);
    if (strcmp(sfx, "attention.sliding_window_pattern") == 0)
        return wri_rd_swa_pattern(rd, vtype, a);

    return wri_skip_value(rd, vtype, 0);
}

static int wri_gguf_handle_kv(wri_gguf_rd *rd, wr_gguf *g, const char *key,
                              uint32_t vtype)
{
    wr_gguf_tokenizer *tok = &g->tokenizer;
    size_t alen;

    if (strcmp(key, "general.alignment") == 0) {
        wri_val v;
        int got = 0;
        uint64_t alg;
        int rc = wri_rd_field_val(rd, vtype, &v, &got);
        if (rc != WR_OK)
            return rc;
        alg = got ? (v.is_float ? wri_f64_to_u64_sat(v.f) : v.u) : 0;
        /* V2: power of two, in the sane [8, 4096] range — a bad value
         * would corrupt the data-offset mask math, so it is an error,
         * never a silent fall-back to the default. */
        if (!got || alg < 8 || alg > 4096 || (alg & (alg - 1)) != 0) {
            wri_log_msg(0, "gguf: general.alignment %" PRIu64
                       " invalid (must be a power of two in [8,4096])", alg);
            return WR_ERR_FORMAT;
        }
        g->alignment = (uint32_t)alg;
        return WR_OK;
    }
    if (strcmp(key, "general.architecture") == 0) {
        if (vtype != WRI_GGUF_V_STRING)
            return wri_skip_value(rd, vtype, 0);
        return wri_rd_string_fixed(rd, g->arch.architecture,
                                   sizeof g->arch.architecture, NULL);
    }
    if (strcmp(key, "tokenizer.ggml.model") == 0) {
        if (vtype != WRI_GGUF_V_STRING)
            return wri_skip_value(rd, vtype, 0);
        return wri_rd_string_fixed(rd, tok->model, sizeof tok->model, NULL);
    }
    if (strcmp(key, "tokenizer.ggml.tokens") == 0)
        return wri_rd_kv_string_array(rd, vtype, &tok->tokens,
                                      &tok->vocab_size, key);
    if (strcmp(key, "tokenizer.ggml.merges") == 0)
        return wri_rd_kv_string_array(rd, vtype, &tok->merges,
                                      &tok->merge_count, key);
    if (strcmp(key, "tokenizer.ggml.token_type") == 0)
        return wri_rd_kv_token_types(rd, vtype, tok);
    if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0)
        return wri_rd_field_i32(rd, vtype, &tok->bos_token_id);
    if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0)
        return wri_rd_field_i32(rd, vtype, &tok->eos_token_id);
    if (strcmp(key, "tokenizer.ggml.padding_token_id") == 0)
        return wri_rd_field_i32(rd, vtype, &tok->pad_token_id);
    if (strcmp(key, "tokenizer.ggml.unknown_token_id") == 0)
        return wri_rd_field_i32(rd, vtype, &tok->unk_token_id);
    if (strcmp(key, "tokenizer.ggml.byte_fallback") == 0)
        return wri_rd_field_bool(rd, vtype, &tok->byte_fallback,
                                 &tok->byte_fallback_present);
    if (strcmp(key, "tokenizer.ggml.add_space_prefix") == 0)
        return wri_rd_field_bool(rd, vtype, &tok->add_space_prefix,
                                 &tok->add_space_prefix_present);

    alen = strlen(g->arch.architecture);
    if (alen > 0 && strncmp(key, g->arch.architecture, alen) == 0 &&
        key[alen] == '.')
        return wri_gguf_handle_arch_kv(rd, g, key + alen + 1, vtype);

    return wri_skip_value(rd, vtype, 0);
}

/* --------------------------------------------------------------------------
 * Open / close
 * -------------------------------------------------------------------------- */

static int wri_cmp_tensor_offset(const void *pa, const void *pb)
{
    const wr_gguf_tensor_info *a = *(const wr_gguf_tensor_info *const *)pa;
    const wr_gguf_tensor_info *b = *(const wr_gguf_tensor_info *const *)pb;
    if (a->offset < b->offset)
        return -1;
    if (a->offset > b->offset)
        return 1;
    return 0;
}

/* V5 overlap check: sort by offset, then any overlap shows up between
 * neighbors.  Runs only after every tensor's range passed the EOF check. */
static int wri_gguf_check_overlaps(wr_gguf *g)
{
    const wr_gguf_tensor_info **order;
    uint64_t i;
    int rc = WR_OK;

    if (g->tensor_count < 2)
        return WR_OK;

    order = (const wr_gguf_tensor_info **)
        malloc((size_t)g->tensor_count * sizeof *order);
    if (!order)
        return WR_ERR_NOMEM;
    for (i = 0; i < g->tensor_count; i++)
        order[i] = &g->tensors[i];
    qsort(order, (size_t)g->tensor_count, sizeof *order,
          wri_cmp_tensor_offset);

    for (i = 1; i < g->tensor_count; i++) {
        const wr_gguf_tensor_info *prev = order[i - 1];
        const wr_gguf_tensor_info *next = order[i];
        if (prev->offset + prev->data_size > next->offset) {
            wri_log_msg(0, "gguf: tensors '%s' and '%s' overlap "
                       "(offsets %" PRIu64 "+%" PRIu64 " and %" PRIu64 ")",
                    prev->name, next->name,
                    prev->offset, prev->data_size, next->offset);
            rc = WR_ERR_FORMAT;
            break;
        }
    }
    free(order);
    return rc;
}

int wri_gguf_open(const char *path, wr_gguf **out)
{
    FILE *f;
    wr_gguf *g;
    wri_gguf_rd rd;
    uint64_t fsize = 0, kv_start, i, aligned;
    uint8_t magic[4];
    int rc;

    if (!out)
        return WR_ERR_INVAL;
    *out = NULL;
    if (!path)
        return WR_ERR_INVAL;

    f = fopen(path, "rb");
    if (!f) {
        wri_log_msg(0, "gguf: cannot open '%s'", path);
        return WR_ERR_IO;
    }
    if (wr_file_size(f, &fsize) != 0) {
        wri_log_msg(0, "gguf: cannot stat '%s'", path);
        fclose(f);
        return WR_ERR_IO;
    }
    if (fsize > (uint64_t)INT64_MAX) {
        wri_log_msg(0, "gguf: '%s': implausible file size", path);
        fclose(f);
        return WR_ERR_FORMAT;
    }

    g = (wr_gguf *)calloc(1, sizeof *g);
    if (!g) {
        fclose(f);
        return WR_ERR_NOMEM;
    }
    g->file = f;
    g->file_size = fsize;
    g->alignment = WRI_GGUF_DEFAULT_ALIGNMENT;
    g->tokenizer.bos_token_id = -1;
    g->tokenizer.eos_token_id = -1;
    g->tokenizer.pad_token_id = -1;
    g->tokenizer.unk_token_id = -1;

    rd.f = f;
    rd.size = fsize;
    rd.cur = 0;

    /* V1: magic + version gate. */
    if ((rc = wri_rd_bytes(&rd, magic, 4)) != WR_OK)
        goto fail;
    if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' ||
        magic[3] != 'F') {
        wri_log_msg(0, "gguf: '%s' is not a GGUF file (bad magic)", path);
        rc = WR_ERR_FORMAT;
        goto fail;
    }
    if ((rc = wri_rd_u32(&rd, &g->version)) != WR_OK)
        goto fail;
    if (g->version != 2 && g->version != 3) {
        if (g->version == 0x02000000u || g->version == 0x03000000u)
            wri_log_msg(0, "gguf: byte-swapped header version — "
                       "big-endian GGUF files are not supported");
        else
            wri_log_msg(0, "gguf: unsupported header version %u "
                       "(v2/v3 only; the v1 layout differs and would "
                       "misparse, so it is rejected)", g->version);
        rc = WR_ERR_FORMAT;
        goto fail;
    }

    if ((rc = wri_rd_u64(&rd, &g->tensor_count)) != WR_OK)
        goto fail;
    if ((rc = wri_rd_u64(&rd, &g->kv_count)) != WR_OK)
        goto fail;

    /* V3: hard limit — an error, never a silent drop of the excess. */
    if (g->tensor_count > WR_GGUF_MAX_TENSORS) {
        wri_log_msg(0, "gguf: %" PRIu64 " tensors exceeds the %d limit",
                g->tensor_count, WR_GGUF_MAX_TENSORS);
        rc = WR_ERR_LIMIT;
        goto fail;
    }
    /* Plausibility: each KV occupies at least 13 bytes on disk (8-byte
     * key length prefix + 4-byte value type + 1-byte value), so a count
     * beyond remaining/13 cannot be honest.  Fails fast instead of
     * grinding a huge loop toward the same EOF error. */
    if (g->kv_count > (rd.size - rd.cur) / 13) {
        wri_log_msg(0, "gguf: kv count %" PRIu64
                   " exceeds what the file could hold", g->kv_count);
        rc = WR_ERR_FORMAT;
        goto fail;
    }

    kv_start = rd.cur;

    /* Pre-scan for general.architecture so "<architecture>.<suffix>"
     * keys match regardless of key order in the file.  Best-effort:
     * any malformation encountered here recurs in the main pass, which
     * owns error reporting. */
    for (i = 0; i < g->kv_count; i++) {
        char key[128];
        uint64_t klen = 0;
        uint32_t vtype = 0;
        if (wri_rd_string_fixed(&rd, key, sizeof key, &klen) != WR_OK)
            break;
        if (wri_rd_u32(&rd, &vtype) != WR_OK)
            break;
        if (klen < sizeof key &&
            strcmp(key, "general.architecture") == 0 &&
            vtype == WRI_GGUF_V_STRING) {
            if (wri_rd_string_fixed(&rd, g->arch.architecture,
                                    sizeof g->arch.architecture,
                                    NULL) != WR_OK)
                g->arch.architecture[0] = '\0';
            break;
        }
        if (wri_skip_value(&rd, vtype, 0) != WR_OK)
            break;
    }
    if (wr_fseek64(f, (int64_t)kv_start, SEEK_SET) != 0) {
        rc = WR_ERR_IO;
        goto fail;
    }
    rd.cur = kv_start;

    /* Main KV pass. */
    for (i = 0; i < g->kv_count; i++) {
        char key[128];
        uint64_t klen = 0;
        uint32_t vtype = 0;
        if ((rc = wri_rd_string_fixed(&rd, key, sizeof key, &klen)) != WR_OK)
            goto fail;
        if ((rc = wri_rd_u32(&rd, &vtype)) != WR_OK)
            goto fail;
        if (klen >= sizeof key) {
            /* An over-long key cannot equal any recognized name; skip
             * its value and move on (its truncation cannot alias a
             * shorter known key under exact comparison). */
            if ((rc = wri_skip_value(&rd, vtype, 0)) != WR_OK)
                goto fail;
            continue;
        }
        if ((rc = wri_gguf_handle_kv(&rd, g, key, vtype)) != WR_OK)
            goto fail;
    }

    /* Tensor headers. */
    if (g->tensor_count > 0) {
        g->tensors = (wr_gguf_tensor_info *)
            calloc((size_t)g->tensor_count, sizeof *g->tensors);
        if (!g->tensors) {
            rc = WR_ERR_NOMEM;
            goto fail;
        }
    }
    for (i = 0; i < g->tensor_count; i++) {
        wr_gguf_tensor_info *t = &g->tensors[i];
        uint64_t nlen = 0, elems;
        uint32_t d;

        if ((rc = wri_rd_string_fixed(&rd, t->name, sizeof t->name,
                                      &nlen)) != WR_OK)
            goto fail;
        if (nlen >= sizeof t->name)
            wri_log_msg(1, "gguf: tensor %" PRIu64 " name of %" PRIu64
                       " bytes truncated to '%s'", i, nlen, t->name);
        if ((rc = wri_rd_u32(&rd, &t->n_dims)) != WR_OK)
            goto fail;
        if (t->n_dims == 0 || t->n_dims > WR_GGUF_MAX_DIMS) {
            wri_log_msg(0, "gguf: tensor '%s': invalid n_dims %u",
                    t->name, t->n_dims);
            rc = WR_ERR_FORMAT;
            goto fail;
        }
        for (d = 0; d < t->n_dims; d++)
            if ((rc = wri_rd_u64(&rd, &t->dims[d])) != WR_OK)
                goto fail;
        if ((rc = wri_rd_u32(&rd, &t->type)) != WR_OK)
            goto fail;
        if ((rc = wri_rd_u64(&rd, &t->offset)) != WR_OK)
            goto fail;

        /* V6: type known and sized, dims nonzero, element count and
         * storage size fit uint64 without overflow. */
        elems = wri_gguf_tensor_elements(t);
        if (elems == 0) {
            wri_log_msg(0, "gguf: tensor '%s': zero or overflowing "
                       "element count", t->name);
            rc = WR_ERR_FORMAT;
            goto fail;
        }
        t->data_size = wri_ggml_type_storage_size(t->type, elems);
        if (t->data_size == 0) {
            wri_log_msg(0, "gguf: tensor '%s': wire type %s (%u) is unknown "
                       "or cannot be sized", t->name,
                    wri_ggml_type_name(t->type), t->type);
            rc = WR_ERR_FORMAT;
            goto fail;
        }
        /* The spec requires tensor data offsets aligned to
         * general.alignment. */
        if (t->offset % g->alignment != 0) {
            wri_log_msg(0, "gguf: tensor '%s': offset %" PRIu64
                       " not a multiple of alignment %u",
                    t->name, t->offset, g->alignment);
            rc = WR_ERR_FORMAT;
            goto fail;
        }
    }

    /* Data section start: cursor aligned up (alignment is a validated
     * power of two, so the mask math is exact). */
    aligned = (rd.cur + (uint64_t)g->alignment - 1) &
              ~((uint64_t)g->alignment - 1);
    g->data_offset = aligned;
    if (g->tensor_count > 0 && g->data_offset > g->file_size) {
        wri_log_msg(0, "gguf: data section offset %" PRIu64
                   " lies past file size %" PRIu64,
                g->data_offset, g->file_size);
        rc = WR_ERR_FORMAT;
        goto fail;
    }

    /* V5: every tensor's byte range inside the real file. */
    for (i = 0; i < g->tensor_count; i++) {
        const wr_gguf_tensor_info *t = &g->tensors[i];
        if (t->offset > g->file_size - g->data_offset ||
            t->data_size > g->file_size - g->data_offset - t->offset) {
            wri_log_msg(0, "gguf: tensor '%s': range [%" PRIu64 ", +%" PRIu64
                       ") extends past file size %" PRIu64,
                    t->name, g->data_offset + t->offset, t->data_size,
                    g->file_size);
            rc = WR_ERR_FORMAT;
            goto fail;
        }
    }
    /* V5: no two tensors overlap. */
    if ((rc = wri_gguf_check_overlaps(g)) != WR_OK)
        goto fail;

    wri_log_msg(2, "gguf: '%s': v%u, %" PRIu64 " tensors, %" PRIu64
               " kv pairs, arch '%s', data section at %" PRIu64,
            path, g->version, g->tensor_count, g->kv_count,
            g->arch.architecture, g->data_offset);

    *out = g;
    return WR_OK;

fail:
    wri_gguf_close(g);
    return rc;
}

void wri_gguf_close(wr_gguf *g)
{
    uint32_t i;

    if (!g)
        return;
    if (g->map_base) {
        wr_file_unmap(g->map_base, g->map_len);
        g->map_base = NULL;
        g->map_len = 0;
    }
    if (g->file) {
        fclose(g->file);
        g->file = NULL;
    }
    if (g->tokenizer.tokens) {
        for (i = 0; i < g->tokenizer.vocab_size; i++)
            free(g->tokenizer.tokens[i]);
        free(g->tokenizer.tokens);
        g->tokenizer.tokens = NULL;
    }
    if (g->tokenizer.merges) {
        for (i = 0; i < g->tokenizer.merge_count; i++)
            free(g->tokenizer.merges[i]);
        free(g->tokenizer.merges);
        g->tokenizer.merges = NULL;
    }
    free(g->tokenizer.token_type);
    g->tokenizer.token_type = NULL;
    free(g->tensors);
    g->tensors = NULL;
    free(g);
}

/* --------------------------------------------------------------------------
 * Tensor access
 * -------------------------------------------------------------------------- */

int wri_gguf_find_tensor(const wr_gguf *g, const char *name)
{
    uint64_t i;

    if (!g || !name)
        return -1;
    for (i = 0; i < g->tensor_count; i++)
        if (strcmp(g->tensors[i].name, name) == 0)
            return (int)i;
    return -1;
}

int wri_gguf_read_tensor_data(wr_gguf *g, uint32_t idx,
                              void *dst, uint64_t dst_bytes)
{
    const wr_gguf_tensor_info *t;
    uint8_t *p;
    uint64_t left;

    if (!g || !g->file || !dst)
        return WR_ERR_INVAL;
    if ((uint64_t)idx >= g->tensor_count)
        return WR_ERR_INVAL;
    t = &g->tensors[idx];
    if (dst_bytes != t->data_size) {
        wri_log_msg(0, "gguf: tensor '%s': read of %" PRIu64
                   " bytes does not match its data size %" PRIu64,
                t->name, dst_bytes, t->data_size);
        return WR_ERR_INVAL;
    }
    /* data_offset + offset + data_size <= file_size was proven at open
     * (V5), so this arithmetic cannot wrap. */
    if (wr_fseek64(g->file, (int64_t)(g->data_offset + t->offset),
                   SEEK_SET) != 0) {
        wri_log_msg(0, "gguf: tensor '%s': seek failure", t->name);
        return WR_ERR_IO;
    }
    p = (uint8_t *)dst;
    left = dst_bytes;
    while (left > 0) {
        size_t chunk = (left > (uint64_t)1 << 30)
                           ? (size_t)((uint64_t)1 << 30)
                           : (size_t)left;
        if (fread(p, 1, chunk, g->file) != chunk) {
            wri_log_msg(0, "gguf: tensor '%s': short read", t->name);
            return WR_ERR_IO;
        }
        p += chunk;
        left -= chunk;
    }
    return WR_OK;
}

int wri_gguf_map_data(wr_gguf *g)
{
    uint64_t len;
    void *base;

    if (!g || !g->file)
        return WR_ERR_INVAL;
    if (g->map_base)
        return WR_OK;   /* idempotent */
    if (g->data_offset >= g->file_size)
        return WR_ERR_IO;   /* nothing to map */
    len = g->file_size - g->data_offset;
    if (len != (uint64_t)(size_t)len)
        return WR_ERR_IO;   /* would not fit a size_t on this host */
    base = wr_file_map_ro(g->file, g->data_offset, (size_t)len);
    if (!base) {
        wri_log_msg(1, "gguf: data-section mapping failed; "
                   "caller falls back to streamed reads");
        return WR_ERR_IO;
    }
    g->map_base = base;
    g->map_len = (size_t)len;
    return WR_OK;
}

const void *wri_gguf_tensor_ptr(const wr_gguf *g, uint32_t idx)
{
    if (!g || !g->map_base || (uint64_t)idx >= g->tensor_count)
        return NULL;
    /* offset + data_size <= map_len was proven at open (V5). */
    return (const uint8_t *)g->map_base + g->tensors[idx].offset;
}
