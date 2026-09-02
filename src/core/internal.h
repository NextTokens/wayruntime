/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * internal.h — shared internal contracts: dtypes, tensor descriptor,
 * compute ops, limits, engine/model/session structs, counters.
 *
 * Every src/ module includes this header.  It is the numeric anchor of
 * the whole port: the dtype enum values, block-format constants, op ids
 * and flag bits below are pinned to the origin OS's AI stack and to the
 * GGML/GGUF on-disk formats.  DO NOT renumber anything in this file —
 * golden tests, counters ABI and weight mmap-compatibility all depend
 * on the exact values.
 */
#ifndef WR_INTERNAL_H
#define WR_INTERNAL_H

#include <assert.h>
#include <stdint.h>
#include <stddef.h>

#include <wayruntime/wayruntime.h>

#include "platform/platform.h"
#include "core/pool.h"

/* --------------------------------------------------------------------------
 * Toolchain / host gates
 * -------------------------------------------------------------------------- */

#if !defined(__GNUC__)
#  error "wayruntime requires gcc or clang (GCC vector extensions, __atomic builtins)"
#endif

/* Little-endian hosts only: GGUF parsing, F16/BF16 bit-casts and the
 * quantized block layouts all assume LE byte order. */
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#  error "wayruntime supports little-endian hosts only"
#endif

/* Debug-build invariant check.  Used for conditions that indicate a bug
 * INSIDE the library (never for untrusted input — that gets a wr_status). */
#define WRI_ASSERT(cond) assert(cond)

/* --------------------------------------------------------------------------
 * Data types
 *
 * Numeric values are pinned to the origin engine's dtype enum; the
 * self-test fixtures and the quantized-block goldens encode them.
 * Block layouts are byte-identical to llama.cpp's ggml-quants.c, so GGUF
 * tensor bytes load / mmap verbatim.
 * -------------------------------------------------------------------------- */

typedef enum wr_dtype {
    WR_DTYPE_F32   = 0,
    WR_DTYPE_F16   = 1,   /* IEEE binary16.  NOTE: wri_f32_to_f16 TRUNCATES
                           * (no round-to-nearest) — a deliberate parity
                           * decision, see quant.h. */
    WR_DTYPE_BF16  = 2,   /* high 16 bits of an f32 */
    WR_DTYPE_INT8  = 3,
    WR_DTYPE_INT4  = 4,   /* packed nibbles; see nibble-order note below */
    WR_DTYPE_Q4_0  = 5,   /* GGML Q4_0: 32-elem block, fp16 scale + 16 B */
    WR_DTYPE_Q8_0  = 6,   /* GGML Q8_0: 32-elem block, fp16 scale + 32 B */
    WR_DTYPE_Q4_K  = 7,   /* GGML K-quants: 256-elem super-blocks         */
    WR_DTYPE_Q5_K  = 8,
    WR_DTYPE_Q6_K  = 9,
    WR_DTYPE_COUNT = 10
} wr_dtype;

/* NIBBLE-ORDER LANDMINE (verified against the origin compute engine and
 * do-not-unify by design):
 *   INT4:  element at even index i lives in the HIGH nibble of byte i/2,
 *          odd index in the LOW nibble.  Offset-binary nibble: value =
 *          nibble - 8, mapping 0x0..0xF to -8..+7 (NOT two's complement).
 *   Q4_0:  element 2k lives in the LOW nibble of block byte k, element
 *          2k+1 in the HIGH nibble.  Unsigned nibble, value (q - 8)*scale.
 * These are OPPOSITE orders.  Q4_0 follows the GGML file format; INT4 is
 * the engine-native packed format.  Any "cleanup" unifying them corrupts
 * one of the two silently. */

/* Block-format constants (bytes verified against the GGML spec and the
 * origin header's layout comments). */
#define WR_QK4_0            32
#define WR_Q4_0_BLOCK_BYTES 18    /* 2 (fp16 d) + 16 (nibbles)             */
#define WR_QK8_0            32
#define WR_Q8_0_BLOCK_BYTES 34    /* 2 (fp16 d) + 32 (int8)                */
#define WR_QK_K             256   /* K-quant super-block element count     */
#define WR_Q4_K_BLOCK_BYTES 144   /* 2 d + 2 dmin + 12 scales(6b) + 128    */
#define WR_Q5_K_BLOCK_BYTES 176   /* 4 (d+dmin) + 12 scales + 32 hi + 128  */
#define WR_Q6_K_BLOCK_BYTES 210   /* 128 lo4 + 64 hi2 + 16 scales + 2 d    */

/* Parse-only GGML types (recognized and SIZED by the GGUF layer for
 * bounds validation, but with no wr_dtype: loading a model that NEEDS
 * one fails with WR_ERR_UNSUPPORTED naming the tensor and type). */
#define WR_QK4_1            32
#define WR_Q4_1_BLOCK_BYTES 20    /* 2 d + 2 m + 16 nibbles                */
#define WR_QK5_0            32
#define WR_Q5_0_BLOCK_BYTES 22    /* 2 d + 4 qh + 16 nibbles               */
#define WR_QK5_1            32
#define WR_Q5_1_BLOCK_BYTES 24    /* 2 d + 2 m + 4 qh + 16 nibbles         */

/* --------------------------------------------------------------------------
 * Tensor descriptor
 * -------------------------------------------------------------------------- */

#define WR_TENSOR_MAX_DIMS 8

/* Flag bits keep their origin values (self-tests assert on raw flags). */
#define WR_TENSOR_GROWABLE    (1u << 1)  /* KV cache: only valid_rows of
                                          * shape[0] are populated; attention
                                          * reads valid_rows, not shape[0]  */
#define WR_TENSOR_ARENA       (1u << 2)  /* data points into an arena slab:
                                          * wri_tensor_free must NOT free it */
#define WR_TENSOR_GGML_WEIGHT (1u << 3)  /* GGML-layout weight W[out][in],
                                          * `in` contiguous, stored exactly
                                          * as in the GGUF file (mmap-able
                                          * verbatim).  wri_op_matmul reads
                                          * it TRANSPOSED relative to a
                                          * native row-major [K,N] operand. */
/* bit 0 is reserved (origin: cross-process sharing — meaningless here). */

typedef struct wr_tensor {
    void    *data;        /* element/block storage; NULL = empty slot     */
    uint64_t size_bytes;  /* capacity of `data` in bytes (bounds anchor
                           * for every element accessor — never 0 on a
                           * live tensor, including quantized ones)       */
    uint32_t shape[WR_TENSOR_MAX_DIMS]; /* shape[0] is the outermost dim  */
    uint8_t  ndim;
    uint8_t  dtype;       /* wr_dtype */
    uint16_t flags;       /* WR_TENSOR_* */
    uint32_t valid_rows;  /* rows of shape[0] populated; meaningful only
                           * with WR_TENSOR_GROWABLE (KV cursor)          */
} wr_tensor;

/* --------------------------------------------------------------------------
 * Compute op ids
 *
 * Decode is imperative (session.c calls the wri_op_* functions directly);
 * this enum survives for counters, self-tests and trace output, with
 * numeric values pinned to the origin graph-op enum (including the fused
 * 0x80+ block). */
typedef enum wr_op {
    WR_OP_MATMUL        = 0,
    WR_OP_ADD           = 1,
    WR_OP_RELU          = 2,
    WR_OP_SOFTMAX       = 3,
    WR_OP_CONV2D        = 4,
    WR_OP_BATCHNORM     = 5,
    WR_OP_MAXPOOL       = 6,
    WR_OP_AVGPOOL       = 7,
    WR_OP_TRANSPOSE     = 8,
    WR_OP_CONCAT        = 9,
    WR_OP_MUL           = 10,
    WR_OP_GELU          = 11,
    WR_OP_LAYERNORM     = 12,
    WR_OP_ATTENTION     = 13,
    WR_OP_EMBED         = 14,
    WR_OP_RESIDUAL_ADD  = 15,
    WR_OP_RMSNORM       = 16,
    WR_OP_SILU          = 17,
    WR_OP_ROPE          = 18,
    WR_OP_GQA_ATTENTION = 19,
    WR_OP_COUNT         = 20,

    /* Fused ops (high bit marks fused; values pinned). */
    WR_OP_FUSED_MATMUL_BIAS      = 0x80,
    WR_OP_FUSED_MATMUL_RELU      = 0x81,
    WR_OP_FUSED_MATMUL_BIAS_RELU = 0x82,
    WR_OP_FUSED_BATCHNORM_RELU   = 0x83,
    WR_OP_FUSED_SILU_MUL         = 0x84,  /* SwiGLU gate: silu(A) ⊙ B     */
    WR_OP_FUSED_GELU_MUL         = 0x85,  /* GeGLU gate: gelu_tanh(A) ⊙ B */
    WR_OP_RMSNORM_GEMMA          = 0x86   /* rmsnorm, eps 1e-6, weight
                                           * applied direct (the (1+w) is
                                           * pre-folded at load)          */
} wr_op;

/* --------------------------------------------------------------------------
 * Limits
 *
 * Origin limits that were SILENT clamps are now either config-checked
 * hard errors (context, layers, batch) or observable fallbacks
 * (WR_CTR_MATMUL_PERELEM).  Nothing in the library silently truncates.
 * -------------------------------------------------------------------------- */

#define WR_MAX_LAYERS          64     /* weight-slot arrays; loader errors
                                       * (WR_ERR_LIMIT) above this        */
#ifndef WR_ATTN_MAX_SEQ
#define WR_ATTN_MAX_SEQ        4096   /* attention sequence hard cap.
                                       * Session create fails with
                                       * WR_ERR_LIMIT when max_context
                                       * exceeds it — the origin's silent
                                       * eff_sk truncation is BANNED.
                                       * Compile-time overridable.        */
#endif
#define WR_DEFAULT_MAX_CONTEXT 4096   /* default session bound when model
                                       * metadata allows more             */
#define WR_FLASH_BLOCK         64     /* flash-attention K/V tile rows    */
#define WR_FLASH_D_MAX         1024   /* max head_dim for the flash path  */
#define WR_QMM_K_MAX           12288  /* longest decoded weight-row in the
                                       * quantized-matmul fast path; larger
                                       * K falls back to the per-element
                                       * path (counted, never wrong)      */
#define WR_QMM_M_MAX           32     /* max activation rows in the fast
                                       * path (decode M=1; batched LM head
                                       * M<=WR_BATCH_MAX)                 */
#define WR_PARALLEL_THRESHOLD  256    /* min N (or M) before a matmul is
                                       * dispatched to the pool           */
#define WR_MAX_WORKERS         32     /* pool-size clamp.  Origin capped at
                                       * 8 for its own scheduler's sake;
                                       * hosted processes may use more.
                                       * Bit-exactness does NOT depend on
                                       * worker count (see pool.h).       */

/* --------------------------------------------------------------------------
 * Counters
 *
 * Slot order is the public ABI (wr_engine_counters); accumulator slots
 * are relaxed-atomic uint64_t bumped from ops, variant slots are filled
 * at snapshot time from the dispatch state.
 * -------------------------------------------------------------------------- */

extern uint64_t wri_g_counters[WR_COUNTER_COUNT];
/* Zero every counter (engine create: counters are engine-scoped per the
 * public contract, the storage is process-global). */
void wri_counters_reset(void);

/* Upper bound on n_layers * pl_emb_dim (Gemma per-layer-input width):
 * keeps every 32-bit product of the pair exact.  Real models are in the
 * low thousands. */
#define WRI_PLE_WIDTH_MAX (1u << 24)

#define WRI_CTR_ADD(slot, n) \
    __atomic_fetch_add(&wri_g_counters[(slot)], (uint64_t)(n), __ATOMIC_RELAXED)

/* Fill out[] with accumulators + current dispatch variants (slots
 * WR_CTR_MATMUL_VARIANT / _SOFTMAX_VARIANT / _MATMUL_GGML_VARIANT). */
void wri_counters_snapshot(uint64_t out[WR_COUNTER_COUNT]);

/* --------------------------------------------------------------------------
 * SIMD dispatch (kernels.c)
 *
 * Runtime selection scalar/AVX2/AVX-512 on x86-64, NEON on arm64.
 * Selection is atomically published; readers take an acquire snapshot of
 * the function pointer at call time, so re-selection while other threads
 * decode is safe (in-flight ops complete on the old variant).
 * -------------------------------------------------------------------------- */

/* (Re)bind all op dispatch tables.  Called from wr_engine_create and
 * wr_engine_set_simd.  force_scalar wins over prefer_avx2. */
void wri_simd_init(int force_scalar, int prefer_avx2);

int wri_simd_matmul_variant(void);   /* WR_SIMD_* actually bound */
int wri_simd_softmax_variant(void);

/* --------------------------------------------------------------------------
 * Compute ops (kernels.c)
 *
 * All ops:
 *   - validate shapes/dtypes and return WR_OK or WR_ERR_INVAL /
 *     WR_ERR_UNSUPPORTED — the origin's silent early-returns are errors
 *     now;
 *   - read inputs via the bounds-checked accessors (quant.h) or via
 *     dtype-specialized fast paths that honor size_bytes;
 *   - write F32 outputs unless noted;
 *   - honor valid_rows on WR_TENSOR_GROWABLE inputs;
 *   - are safe to call concurrently from different sessions (all state
 *     is in the arguments; the worker pool serializes internally).
 * -------------------------------------------------------------------------- */

/* C[M,N] = A[M,K] × B.  B is either native row-major [K,N], or a GGML
 * weight W[N,K] read transposed when (B->flags & WR_TENSOR_GGML_WEIGHT).
 * A must be F32 (activations); B may be F32/F16/BF16/INT8/INT4 or any
 * quantized dtype (kept-quantized fast paths: Q8_0/Q4_K/Q6_K decode one
 * weight row at a time).  Parallelization contract:
 *   - quantized GGML weights, plus F32 GGML weights when M == 1: N-split
 *     — each output COLUMN's complete k-reduction executes inside one pool
 *     part (bit-exact for any thread count);
 *   - other F32 × F32 shapes: M-split over disjoint row blocks, threshold
 *     WR_PARALLEL_THRESHOLD.
 * Falling off a fast path (K > WR_QMM_K_MAX, odd shapes) bumps
 * WR_CTR_MATMUL_PERELEM and computes per-element — slower, never wrong. */
int wri_op_matmul(const wr_tensor *A, const wr_tensor *B, wr_tensor *C);

/* Elementwise C = A + B (residual add) and C = A ⊙ B.  Shapes equal. */
int wri_op_add(const wr_tensor *A, const wr_tensor *B, wr_tensor *C);
int wri_op_mul(const wr_tensor *A, const wr_tensor *B, wr_tensor *C);

int wri_op_relu(const wr_tensor *A, wr_tensor *C);
int wri_op_gelu(const wr_tensor *A, wr_tensor *C);  /* gelu_pytorch_tanh */
int wri_op_silu(const wr_tensor *A, wr_tensor *C);

/* Softmax along the trailing dimension, one pass per row.  Numerically
 * stabilized (max-shifted); AVX-512 hosts deliberately run the AVX2 body
 * (measured no-win; keep that decision). */
int wri_op_softmax(const wr_tensor *A, wr_tensor *C);

/* Fused GLU gates: C = act(A) ⊙ B, act = silu / gelu_pytorch_tanh. */
int wri_op_fused_silu_mul(const wr_tensor *A, const wr_tensor *B,
                          wr_tensor *C);
int wri_op_fused_gelu_mul(const wr_tensor *A, const wr_tensor *B,
                          wr_tensor *C);

/* RMSNorm: C = X * W / sqrt(mean(X^2) + eps), rows = trailing dim.
 * W == NULL: weightless normalization (Gemma V-norm before KV append).
 * eps comes from model metadata (1e-5 Llama-class, 1e-6 Gemma); the
 * Gemma "direct weight" variant is byte-identical here because the
 * (1+w) transform is pre-folded into the loaded norm weights. */
int wri_op_rmsnorm(const wr_tensor *X, const wr_tensor *W, wr_tensor *C,
                   float eps);

/* LayerNorm (kept for op self-test parity; no production model path). */
int wri_op_layernorm(const wr_tensor *X, const wr_tensor *scale,
                     const wr_tensor *bias, wr_tensor *C, float eps);

/* RoPE: rotate per-head pairs of X (layout [seq, heads, head_dim] or a
 * 2-D collapse of it) at absolute position `pos` for the first row, with
 * theta base `theta_base` (10000 Llama, 1e6 Qwen3, per-layer for Gemma).
 * n_rot_pairs: rotate only the first n pairs, pass the rest through
 * unrotated (Gemma partial rotary); 0 = rotate all pairs.  C may alias X
 * (in-place rotation is well-defined pair-by-pair).
 * The origin op carried pos/theta/n_rot in a scratch param tensor; they
 * are honest scalars here. */
int wri_op_rope(const wr_tensor *X, wr_tensor *C, uint32_t pos,
                float theta_base, uint32_t n_rot_pairs);

/* Grouped-query attention with flash tiling (WR_FLASH_BLOCK online
 * softmax; no O(seq_k) score materialization).
 *   Q [seq_q, n_q_heads, head_dim], K/V [seq_k, n_kv_heads, head_dim],
 *   C [seq_q, n_q_heads, head_dim]; n_q_heads % n_kv_heads == 0.
 * K/V honor valid_rows when GROWABLE (the KV-cache read path).
 * scale_override <= 0: use 1/sqrt(head_dim); > 0: use it verbatim
 * (Gemma passes an explicit attention scale).
 * window == 0: attend to everything; > 0: sliding window — a query
 * attends only to the last `window` keys (real masking; the origin
 * deferred this and passed 0).
 * seq_k above WR_ATTN_MAX_SEQ is WR_ERR_LIMIT — never truncated. */
int wri_op_gqa_attention(const wr_tensor *Q, const wr_tensor *K,
                         const wr_tensor *V, wr_tensor *C,
                         float scale_override, uint32_t window);

/* Embedding row fetch: out_row[hidden] = F32(table[token_id, :]).
 * table may be any supported dtype (kept-quantized embeddings decode on
 * the fly).  token_id >= table->shape[0] is WR_ERR_INVAL. */
int wri_op_embed(const wr_tensor *table, uint32_t token_id,
                 wr_tensor *out_row);

/* Logit soft-capping: x[i] = cap * tanh(x[i] / cap).  cap > 0. */
void wri_softcap(float *x, uint64_t n, float cap);

/* --------------------------------------------------------------------------
 * Engine
 * -------------------------------------------------------------------------- */

struct wr_engine {
    wr_engine_config cfg;        /* resolved copy (defaults filled in)    */
    wr_pool         *pool;       /* persistent worker pool                */
    uint32_t         models_live;/* refcount guarding engine destroy      */
};

/* v1 is single-engine-per-process; this is the anchor ops and logging
 * reach it through.  NULL when no engine exists. */
extern wr_engine *wri_g_engine;

/* Logging through the engine's sink (stderr fallback).  printf-style. */
void wri_log_msg(int level, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

/* --------------------------------------------------------------------------
 * Model
 *
 * Immutable after wri_model_commit.  Weight slots are embedded
 * descriptors whose data points into the model's arena (or the mmap'd
 * file); an absent optional tensor has data == NULL.  Hyperparameters
 * that were q16 fixed-point in the origin (embed scale, attention scale,
 * softcap, rope bases) are real floats here.
 * -------------------------------------------------------------------------- */

struct wri_arena;   /* tensor.h */
struct wr_gguf;     /* gguf.h   */
struct wr_bpe;      /* bpe.h    */

struct wr_model {
    wr_engine *engine;
    char       arch[32];          /* GGUF architecture string             */

    /* Hyperparameters. */
    uint32_t n_layers;
    uint32_t n_q_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;            /* global-attention head dim            */
    uint32_t head_dim_swa;        /* SWA-layer head dim; 0 = head_dim     */
    uint32_t hidden_dim;
    uint32_t ffn_dim;             /* scalar fallback FFN dim              */
    uint32_t ffn_dim_per_layer[WR_MAX_LAYERS]; /* 0 = use ffn_dim (per-layer
                                   * FFN dims: MatFormer models)          */
    uint32_t vocab_size;
    uint32_t train_context;       /* from metadata                        */
    uint32_t max_context;         /* effective session bound              */
    float    rms_eps;             /* attention.layer_norm_rms_epsilon     */
    float    rope_freq_base;      /* global layers                        */
    float    rope_freq_base_swa;  /* SWA layers (Gemma); loader fills 10000
                                   * when the file has no value            */
    uint32_t n_rot_pairs;         /* partial-rotary pair count on global
                                   * layers; 0 = full rotary              */
    float    embed_scale;         /* embedding multiplied by this; 0 = off */
    float    attn_scale;          /* explicit attention scale; 0 = default
                                   * 1/sqrt(head_dim)                     */
    float    logit_softcap;       /* 0 = off                              */
    uint32_t sliding_window;      /* SWA window size (keys); 0 = none     */
    uint8_t  swa_pattern[WR_MAX_LAYERS]; /* 1 = SWA/local, 0 = global     */
    uint8_t  has_swa;
    uint8_t  has_qk_norm;         /* per-head Q/K RMSNorm pre-RoPE        */
    uint8_t  is_gemma;            /* Gemma-class sandwich-norm forward    */
    uint8_t  tied_lm_head;        /* lm_head aliases embed_table          */
    uint8_t  swa_mode;            /* WR_SWA_* from wr_model_params        */
    uint32_t kv_shared_layers;    /* Gemma KV sharing: the last N layers
                                   * REUSE the post-RoPE K/V of the last
                                   * non-shared layer of the matching
                                   * SWA/global type (their own wk/wv are
                                   * present in the file but DEAD).
                                   * 0 = every layer fresh.               */
    uint32_t pl_emb_dim;          /* per-layer-input embedding dim        */

    /* Weight slots.  GGML-layout matmul weights carry
     * WR_TENSOR_GGML_WEIGHT; norm vectors are plain F32. */
    wr_tensor embed_table;                       /* [vocab, hidden]       */
    wr_tensor final_norm;                        /* [hidden]              */
    wr_tensor lm_head;                           /* [vocab, hidden] GGML
                                   * layout (may alias embed_table when
                                   * tied).  Sessions build a per-call
                                   * TRANSPOSED VIEW [hidden, vocab] with
                                   * wri_tensor_view_2d — the shared
                                   * descriptor is never mutated (the
                                   * origin's in-place reshape raced
                                   * across sessions).                    */
    wr_tensor attn_norm[WR_MAX_LAYERS];          /* [hidden]              */
    wr_tensor wq[WR_MAX_LAYERS];                 /* [hidden -> n_q*hd]    */
    wr_tensor wk[WR_MAX_LAYERS];                 /* [hidden -> n_kv*hd]   */
    wr_tensor wv[WR_MAX_LAYERS];                 /* [hidden -> n_kv*hd]   */
    wr_tensor wo[WR_MAX_LAYERS];                 /* [n_q*hd -> hidden]    */
    wr_tensor ffn_norm[WR_MAX_LAYERS];           /* [hidden]              */
    wr_tensor w_gate[WR_MAX_LAYERS];             /* [hidden -> ffn_l]     */
    wr_tensor w_up[WR_MAX_LAYERS];               /* [hidden -> ffn_l]     */
    wr_tensor w_down[WR_MAX_LAYERS];             /* [ffn_l -> hidden]     */
    wr_tensor q_norm[WR_MAX_LAYERS];             /* [head_dim] (qk-norm)  */
    wr_tensor k_norm[WR_MAX_LAYERS];             /* [head_dim]            */

    /* Gemma-class extras (absent slots have data == NULL). */
    wr_tensor post_attn_norm[WR_MAX_LAYERS];     /* [hidden]              */
    wr_tensor post_ffw_norm[WR_MAX_LAYERS];      /* [hidden]              */
    wr_tensor post_norm[WR_MAX_LAYERS];          /* [hidden]              */
    wr_tensor per_layer_token_embd;              /* [vocab, n_layers*pl_emb]
                                   * ONE tensor — the origin's lo/hi split
                                   * existed only for a 1 GiB kernel
                                   * allocation cap                       */
    wr_tensor per_layer_model_proj;              /* [hidden, n_layers*pl_emb] */
    wr_tensor per_layer_proj_norm;               /* [pl_emb]              */
    wr_tensor inp_gate[WR_MAX_LAYERS];           /* [hidden -> pl_emb]    */
    wr_tensor pl_proj[WR_MAX_LAYERS];            /* [pl_emb -> hidden]    */
    wr_tensor layer_output_scale[WR_MAX_LAYERS]; /* [1] — applied DIRECT,
                                   * no (1+w) transform                   */

    /* Backing storage + bookkeeping. */
    struct wri_arena *arena;      /* weight slab (64 B aligned bump)      */
    struct wr_gguf   *gguf;       /* retained iff use_mmap (weight bytes
                                   * reference the mapping); else NULL    */
    char             *gguf_path;  /* owned copy of the load path; lets a
                                   * tokenizer re-read vocab metadata when
                                   * the mapping was not retained         */
    wr_tokenizer     *cached_tok; /* lazily built for wr_generate         */
    wr_mutex         *lock;       /* guards refcounts + cached_tok        */
    uint32_t          session_refs;
    uint32_t          tokenizer_refs;
    uint8_t           weights_committed; /* wri_model_commit ran          */
};

/* --------------------------------------------------------------------------
 * Session
 *
 * One decode stream: per-layer growable F16 KV cache preallocated to
 * max_context at create, plus the reusable per-step scratch set.  A
 * session is protected by ONE mutex taken for the whole of every decode
 * call (the origin's spinlock + in_flight counter collapse into it).
 * -------------------------------------------------------------------------- */

struct wr_session {
    wr_model  *model;
    wr_mutex  *lock;
    uint32_t   max_context;
    uint32_t   pos;              /* tokens in the KV cache == next position */
    wr_status  last_status;      /* wr_session_status()                    */

    /* KV cache: F16, WR_TENSOR_GROWABLE, shape [max_context, n_kv_heads,
     * head_dim_L] (head_dim_L is per-layer on Gemma).  valid_rows is the
     * cursor; appends convert F32 rows to F16.  On KV-shared layers these
     * slots ALIAS the source layer's tensors (no separate storage). */
    wr_tensor  kv_k[WR_MAX_LAYERS];
    wr_tensor  kv_v[WR_MAX_LAYERS];

    /* Per-step scratch, allocated once from the session slab.  The origin
     * schema minus its pos/params tensors (now scalar op arguments):
     *   residual, normed              [hidden]
     *   q                             [1, n_q_heads, head_dim_max]
     *   k_row, v_row                  [1, n_kv_heads, head_dim_max]
     *   attn                          [1, n_q_heads, head_dim_max]
     *   ffn_gate, ffn_up              [ffn_max]
     *   ffn_proj                      [hidden]
     *   logits                        [vocab]  (F32 — wr_step's view)
     * Gemma per-layer-input extras (empty slots otherwise):
     *   ple_proj, ple_table           [n_layers * pl_emb]
     *   ple_g, ple_pli                [pl_emb]
     *   ple_delta                     [hidden]                            */
    wr_tensor  residual, normed, q, k_row, v_row, attn;
    wr_tensor  ffn_gate, ffn_up, ffn_proj, logits;
    wr_tensor  ple_proj, ple_table, ple_g, ple_pli, ple_delta;

    void      *slab;             /* one wr_map_alloc backing KV + scratch */
    size_t     slab_bytes;
};

/* Decode entry points (session.c).  Callers hold no locks; these take
 * s->lock themselves. */

/* One token forward pass; on WR_OK the logits are in s->logits (F32,
 * vocab_size) and s->pos advanced by one.  compute_logits == 0 skips the
 * LM-head projection (prefill).  WR_ERR_CTX_FULL consumed nothing. */
wr_status wri_session_step(wr_session *s, uint32_t token, int compute_logits);

/* Prefill loop over ids[0..n): steps without LM head.  Returns tokens
 * consumed or negative status; all-or-nothing on context checks. */
int wri_session_prefill(wr_session *s, const uint32_t *ids, uint32_t n);

/* Batched decode (batch.c): step n sessions' layer stacks, then coalesce
 * the LM-head projections of all n [1, hidden] rows into one GEMM against
 * the shared lm_head view and demux into each session's logits scratch.
 * Bit-exact versus n serial wri_session_step calls (the GEMM's per-column
 * reductions are unchanged).  All sessions share one model; n <=
 * WR_BATCH_MAX. */
int wri_batch_decode_step(wr_session *const *sessions, const uint32_t *tokens,
                          uint32_t n);

/* Golden self-tests, ported from the origin (unit_tests.c drives them).
 * Each returns 0 on pass. */
int wri_self_test_quant_dequant(void);
int wri_self_test_matmul_simd(void);
int wri_self_test_softmax(void);
int wri_self_test_flash_attn(void);
int wri_self_test_fused(void);
int wri_self_test_ops(void);
int wri_self_test_qmm_bit_equality(void);  /* 200-rep re-dispatch equality */
int wri_self_test_f32_ggml_nsplit(void);   /* F32 head N-split bit-equal   */
int wri_self_test_llm_step(void);          /* tiny-model decode fixture    */
int wri_self_test_batch_step(void);        /* batched == serial bit-exact  */

#endif /* WR_INTERNAL_H */
