/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * gguf.h — hardened GGUF container parser.
 *
 * A GGUF file is UNTRUSTED INPUT and this parser is the trust boundary
 * for the whole library.  wri_gguf_open performs a full validation pass
 * BEFORE anything downstream touches tensor data:
 *
 *   V1  magic "GGUF"; header version in {2, 3} (v1 layouts differ and
 *       are rejected, not misparsed) ................. WR_ERR_FORMAT
 *   V2  general.alignment is a power of two .......... WR_ERR_FORMAT
 *   V3  tensor_count <= WR_GGUF_MAX_TENSORS — an error, never a silent
 *       drop of the excess ............................ WR_ERR_LIMIT
 *   V4  every string length is checked against the remaining file size
 *       BEFORE the cursor advances (no u64 cursor wrap via crafted
 *       lengths) ...................................... WR_ERR_FORMAT
 *   V5  every tensor's [data_offset + offset, + storage size) lies
 *       inside the real (fstat) file size, and no two tensors overlap
 *       ............................................... WR_ERR_FORMAT
 *   V6  tensor dims/type are sane (known type id, dims nonzero,
 *       element count and storage size fit uint64 without overflow)
 *       ............................................... WR_ERR_FORMAT
 *
 * Metadata keys are built dynamically as "<architecture>.<suffix>", so
 * every architecture's rope/attention keys are honored (the origin's
 * fixed prefix list silently ignored some).  Float metadata stays float.
 */
#ifndef WR_GGUF_H
#define WR_GGUF_H

#include <stdio.h>

#include "core/internal.h"

/* --------------------------------------------------------------------------
 * GGML wire type ids (GGUF spec numbering — never renumber) and mapping
 * -------------------------------------------------------------------------- */

#define WR_GGML_F32    0u
#define WR_GGML_F16    1u
#define WR_GGML_Q4_0   2u
#define WR_GGML_Q4_1   3u    /* parse-only */
#define WR_GGML_Q5_0   6u    /* parse-only */
#define WR_GGML_Q5_1   7u    /* parse-only */
#define WR_GGML_Q8_0   8u
#define WR_GGML_Q4_K   12u
#define WR_GGML_Q5_K   13u
#define WR_GGML_Q6_K   14u
#define WR_GGML_BF16   30u

/* Map a GGML wire type to the internal dtype:
 *   WR_GGML_F32→WR_DTYPE_F32(0)  F16→F16(1)  BF16→BF16(2)
 *   Q4_0→5  Q8_0→6  Q4_K→7  Q5_K→8  Q6_K→9
 * Returns -1 for parse-only and unknown types; the loader turns that
 * into WR_ERR_UNSUPPORTED naming tensor and type IF the tensor is
 * actually needed. */
int wri_ggml_type_to_dtype(uint32_t ggml_type);

/* Storage bytes for `elements` of a wire type.  Defined for ALL types
 * this parser recognizes including the parse-only ones (needed for the
 * V5 bounds/overlap pass); returns 0 for unknown ids. */
uint64_t wri_ggml_type_storage_size(uint32_t ggml_type, uint64_t elements);

/* Static name for diagnostics ("Q5_1", "BF16", "unknown(27)"...). */
const char *wri_ggml_type_name(uint32_t ggml_type);

/* --------------------------------------------------------------------------
 * Parsed structures (heap-allocated by wri_gguf_open)
 * -------------------------------------------------------------------------- */

#define WR_GGUF_MAX_TENSORS 1024
#define WR_GGUF_NAME_MAX    64
#define WR_GGUF_MAX_DIMS    8

typedef struct wr_gguf_tensor_info {
    char     name[WR_GGUF_NAME_MAX];
    uint32_t type;                    /* WR_GGML_* wire id                */
    uint32_t n_dims;
    uint64_t dims[WR_GGUF_MAX_DIMS];  /* GGUF order: dims[0] fastest      */
    uint64_t offset;                  /* from data section start          */
    uint64_t data_size;               /* validated storage bytes          */
} wr_gguf_tensor_info;

/* Tokenizer metadata.  Strings are heap-allocated, owned by wr_gguf,
 * freed by wri_gguf_close — a tokenizer that outlives the gguf must copy. */
typedef struct wr_gguf_tokenizer {
    char      model[32];         /* tokenizer.ggml.model ("llama","gpt2",..;
                                  * empty when absent) */
    uint32_t  vocab_size;
    char    **tokens;            /* tokens[i] = NUL-terminated vocab string */
    uint32_t  merge_count;
    char    **merges;            /* merges[i] = "left right"                */
    int32_t  *token_type;        /* tokenizer.ggml.token_type[i], GGUF
                                  * numbering (1=normal, 2=unknown,
                                  * 3=control, 4=user_defined, 5=unused,
                                  * 6=byte); NULL when absent               */
    uint32_t  token_type_count;
    int32_t   bos_token_id;      /* -1 when absent */
    int32_t   eos_token_id;
    int32_t   pad_token_id;
    int32_t   unk_token_id;
    uint8_t   byte_fallback;         /* tokenizer.ggml.byte_fallback        */
    uint8_t   byte_fallback_present;
    uint8_t   add_space_prefix;      /* tokenizer.ggml.add_space_prefix —
                                      * READ from metadata (default 1 for
                                      * SPM-family, 0 otherwise), never
                                      * hard-coded per-model              */
    uint8_t   add_space_prefix_present;
} wr_gguf_tokenizer;

/* Architecture metadata.  Keys read as "<architecture>.<suffix>".
 * Float-typed metadata is carried as float (no fixed-point). */
typedef struct wr_gguf_arch {
    char     architecture[32];
    uint32_t context_length;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t block_count;
    uint32_t attention_head_count;
    uint32_t attention_head_count_kv;
    float    attention_layer_norm_rms_epsilon;  /* 0 = absent (caller
                                                 * defaults 1e-5)         */
    uint32_t rope_dimension_count;
    float    rope_freq_base;                    /* 0 = absent             */
    uint32_t attention_key_length;              /* per-head dim; 0=absent */
    uint32_t attention_value_length;
    /* Sliding-window family (Gemma-class). */
    float    rope_freq_base_swa;
    uint32_t rope_dimension_count_swa;
    uint32_t attention_key_length_swa;
    uint32_t attention_value_length_swa;
    uint32_t attention_sliding_window;
    uint32_t attention_shared_kv_layers;
    uint32_t embedding_length_per_layer_input;
    float    final_logit_softcapping;           /* 0 = off                */
    uint8_t  swa_pattern[WR_MAX_LAYERS];        /* 1 = SWA/local          */
    uint8_t  has_swa_pattern;
} wr_gguf_arch;

typedef struct wr_gguf {
    FILE    *file;
    uint64_t file_size;       /* fstat'd once at open — the V5 anchor     */

    uint32_t version;         /* 2 or 3 */
    uint64_t tensor_count;
    uint64_t kv_count;
    uint32_t alignment;       /* validated power of two                   */
    uint64_t data_offset;     /* absolute file offset of the data section */

    wr_gguf_tensor_info *tensors;   /* heap, tensor_count entries         */

    wr_gguf_tokenizer tokenizer;
    wr_gguf_arch      arch;

    /* Optional whole-data-section mapping (wr_model_params.use_mmap).
     * NULL until wri_gguf_map_data succeeds. */
    void    *map_base;        /* points at file offset data_offset        */
    size_t   map_len;
} wr_gguf;

/* --------------------------------------------------------------------------
 * API
 * -------------------------------------------------------------------------- */

/* Open + parse + validate.  On success *out is a heap-allocated handle
 * (release with wri_gguf_close).  On failure *out is NULL and the log
 * says which validation failed. */
int wri_gguf_open(const char *path, wr_gguf **out);

void wri_gguf_close(wr_gguf *g);

/* Index of the tensor named `name`, or -1. */
int wri_gguf_find_tensor(const wr_gguf *g, const char *name);

/* Stream tensor `idx`'s raw bytes into dst.  dst_bytes must equal the
 * tensor's data_size exactly (WR_ERR_INVAL otherwise — no partial
 * weight reads).  Positions the file internally; safe in any order. */
int wri_gguf_read_tensor_data(wr_gguf *g, uint32_t idx,
                              void *dst, uint64_t dst_bytes);

/* Map the whole data section read-only (zero-copy path).  Idempotent.
 * Returns WR_OK and sets map_base/map_len, or WR_ERR_IO — the caller
 * falls back to streaming. */
int wri_gguf_map_data(wr_gguf *g);

/* Pointer to tensor idx's bytes inside the mapping (requires a prior
 * successful wri_gguf_map_data; NULL otherwise). */
const void *wri_gguf_tensor_ptr(const wr_gguf *g, uint32_t idx);

/* Element count of a tensor (product of dims). */
uint64_t wri_gguf_tensor_elements(const wr_gguf_tensor_info *t);

#endif /* WR_GGUF_H */
