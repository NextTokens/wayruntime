/* SPDX-License-Identifier: Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * wayruntime — public C API.
 *
 * A freestanding, CPU-only LLM inference library: GGUF model loading,
 * BPE tokenization, autoregressive decode with a per-session KV cache,
 * batched decode across sessions, and sampling.
 *
 * This header is self-contained: it depends only on <stdint.h> and
 * <stddef.h> and compiles under -std=c11 -pedantic.  Everything else
 * in the library is an implementation detail behind the opaque types
 * below.
 *
 * ---------------------------------------------------------------------------
 * Global contracts
 * ---------------------------------------------------------------------------
 *  Engine     One wr_engine per process (v1).  A second wr_engine_create
 *             before the first engine is destroyed returns WR_ERR_STATE.
 *             All library state (thread pool, SIMD dispatch, counters,
 *             weight arena) is scoped to the engine.
 *
 *  Lifetime   Destruction order is the reverse of creation order:
 *             sampler/session before model, model before engine.
 *             Freeing a parent while a child is alive is a contract
 *             violation: the call fails safely (logged, object leaked),
 *             it never frees memory a child still references.
 *
 *  Threads    Objects are externally synchronized per object:
 *               - one wr_session may be driven by one thread at a time;
 *               - different sessions of the SAME model may run decode
 *                 concurrently from different threads (model weights are
 *                 immutable after load);
 *               - wr_engine / wr_model query calls are safe from any
 *                 thread; create/destroy calls are not.
 *             The library owns an internal worker pool; callers never
 *             see its threads.
 *
 *  Errors     wr_status WR_OK == 0; every error is negative.  There is
 *             no silent truncation anywhere in the API: exceeding the
 *             session context is WR_ERR_CTX_FULL, an unsupported GGUF
 *             tensor type is WR_ERR_UNSUPPORTED, malformed input is
 *             WR_ERR_FORMAT / WR_ERR_INVAL.
 *
 *  Memory     Buffers returned as `char *` / owned pointers (for example
 *             wr_generate_result.text) are released with wr_free().
 *             Pointers returned as `const float *` (logits) are borrowed
 *             views into session-owned scratch — never freed by the
 *             caller and only valid until the next decode call on that
 *             session.
 * ---------------------------------------------------------------------------
 */
#ifndef WAYRUNTIME_H
#define WAYRUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Version
 * -------------------------------------------------------------------------- */

#define WR_VERSION_MAJOR 0
#define WR_VERSION_MINOR 1
#define WR_VERSION_PATCH 0

/* Returns the compiled library version, e.g. "0.1.0".  Static storage;
 * never freed. */
const char *wr_version(void);

/* --------------------------------------------------------------------------
 * Status codes
 * -------------------------------------------------------------------------- */

typedef enum wr_status {
    WR_OK              = 0,

    WR_ERR_IO          = -1,   /* file open/read/stat failure (host errno-level) */
    WR_ERR_FORMAT      = -2,   /* malformed GGUF / tokenizer data: bad magic,
                                * unsupported header version, offsets outside the
                                * file, overlapping tensors, non-pow2 alignment,
                                * string length past EOF, ... */
    WR_ERR_UNSUPPORTED = -3,   /* well-formed but not implemented: unknown
                                * architecture, tensor dtype the compute path
                                * cannot execute (e.g. Q4_1/Q5_0/Q5_1/IQ*),
                                * unsupported host (big-endian) */
    WR_ERR_INVAL       = -4,   /* bad argument: NULL where forbidden, zero
                                * dimension, shape mismatch, out-of-range config */
    WR_ERR_NOMEM       = -5,   /* allocation failure (heap or weight arena) */
    WR_ERR_CTX_FULL    = -6,   /* KV cache is at max_context; the step/prefill
                                * consumed nothing.  Never silently truncated. */
    WR_ERR_STATE       = -7,   /* call not legal in the current object state
                                * (second engine, freed parent, session busy) */
    WR_ERR_LIMIT       = -8,   /* a compiled hard limit would be exceeded
                                * (layers > WR_MAX_LAYERS, batch > WR_BATCH_MAX,
                                * max_context > attention sequence cap) */
    WR_ERR_BUSY        = -9,   /* object still referenced (model with live
                                * sessions, engine with live models) */
    WR_ERR_INTERNAL    = -10   /* invariant violation inside the library */
} wr_status;

/* Static human-readable name for a status code ("WR_ERR_CTX_FULL"). */
const char *wr_status_str(wr_status s);

/* --------------------------------------------------------------------------
 * Opaque types
 * -------------------------------------------------------------------------- */

typedef struct wr_engine    wr_engine;     /* process-wide runtime          */
typedef struct wr_model     wr_model;      /* immutable loaded weight set   */
typedef struct wr_tokenizer wr_tokenizer;  /* BPE tokenizer                 */
typedef struct wr_session   wr_session;    /* decode state: KV cache + pos  */
typedef struct wr_sampler   wr_sampler;    /* sampling state: RNG + history */

/* --------------------------------------------------------------------------
 * Callbacks
 * -------------------------------------------------------------------------- */

/* Log sink.  level: 0 = error, 1 = warn, 2 = info, 3 = debug.  `msg` is a
 * NUL-terminated line without trailing newline, valid only for the duration
 * of the call.  May be invoked from library worker threads; must be
 * thread-safe and must not call back into wayruntime. */
typedef void (*wr_log_fn)(int level, const char *msg, void *user);

/* Streaming token callback (wr_generate_params.on_token).  Called once per
 * generated token, after sampling, before the next decode step.  `piece` is
 * the detokenized UTF-8 byte string for this token (byte-level alphabets
 * already normalized to real bytes), valid only during the call.  Return 0
 * to continue, nonzero to stop generation (stop_reason = WR_STOP_CALLBACK).
 * Invoked on the caller's thread.  Must not call wayruntime functions on
 * the generating session. */
typedef int (*wr_token_cb)(uint32_t token_id, const char *piece, void *user);

/* Token-mask (constrained decoding) callback, installed on a sampler with
 * wr_sampler_set_mask.  Consulted during wr_sample for candidate tokens;
 * return nonzero to ALLOW the token, 0 to forbid it.  `piece` follows the
 * wr_token_cb normalization contract (real bytes, not the byte-level
 * alphabet's Ġ/Ċ/ĉ/č surrogates).  If the mask forbids every candidate,
 * wr_sample returns the best forbidden token rather than failing (the
 * grammar consumer is expected to stop instead).  Called on the sampling
 * thread; must be fast and must not call back into wayruntime. */
typedef int (*wr_token_mask_fn)(uint32_t token_id, const char *piece,
                                void *user);

/* --------------------------------------------------------------------------
 * Engine
 * -------------------------------------------------------------------------- */

/* SIMD variants reported by wr_engine_simd_variant.  Values are stable ABI
 * (they also appear in the counters snapshot). */
enum {
    WR_SIMD_SCALAR = 0,
    WR_SIMD_AVX2   = 1,
    WR_SIMD_AVX512 = 2,
    WR_SIMD_NEON   = 3
};

/* Engine configuration.  Zero-initialize (`wr_engine_config cfg = {0};`)
 * then override fields; every zero value means "default". */
typedef struct wr_engine_config {
    uint32_t  n_threads;    /* worker threads for parallel ops.
                             * 0 = online CPU count (clamped to the compiled
                             * worker cap). 1 = fully serial execution. */
    int32_t   force_scalar; /* nonzero: disable all SIMD kernels (parity /
                             * debugging knob) */
    int32_t   prefer_avx2;  /* nonzero: on AVX-512 hosts, bind the AVX2
                             * kernels instead (downclock avoidance) */
    uint32_t  arena_mb;     /* weight arena reservation in MiB.
                             * 0 = size from the model at load time. */
    wr_log_fn log_fn;       /* NULL = stderr */
    void     *log_user;     /* opaque pointer passed to log_fn */
} wr_engine_config;

/* Create the process engine.  `cfg` may be NULL for all-defaults.  Probes
 * CPU features and binds SIMD kernel variants, creates the worker pool.
 * Fails with WR_ERR_STATE if an engine already exists in this process (v1
 * is single-engine), WR_ERR_UNSUPPORTED on a big-endian host. */
wr_status wr_engine_create(const wr_engine_config *cfg, wr_engine **out);

/* Destroy the engine.  All models must have been freed first, else the
 * call is refused (logged; engine leaks).  `e` may be NULL (no-op). */
void wr_engine_destroy(wr_engine *e);

/* Currently bound matmul SIMD variant (WR_SIMD_*). */
int wr_engine_simd_variant(const wr_engine *e);

/* Re-select SIMD kernels at runtime (atomic republish; safe while sessions
 * are decoding on other threads — in-flight ops finish on the old variant).
 * Returns the variant that actually bound (WR_SIMD_*), or a negative
 * wr_status. */
int wr_engine_set_simd(wr_engine *e, int force_scalar, int prefer_avx2);

/* Operation counters snapshot.  Slot order is stable ABI: */
enum {
    WR_CTR_MATMUL_CALLS      = 0,  /* total matmul invocations              */
    WR_CTR_MATMUL_ELEMS      = 1,  /* total output elements across matmuls  */
    WR_CTR_SOFTMAX_CALLS     = 2,
    WR_CTR_ATTENTION_CALLS   = 3,  /* attention + GQA invocations           */
    WR_CTR_FUSED_CALLS       = 4,  /* fused-op invocations                  */
    WR_CTR_MATMUL_VARIANT    = 5,  /* WR_SIMD_* bound for F32 matmul        */
    WR_CTR_SOFTMAX_VARIANT   = 6,  /* WR_SIMD_* bound for softmax           */
    WR_CTR_MATMUL_GGML_SIMD  = 7,  /* SIMD fast-path hits on GGML weights   */
    WR_CTR_MATMUL_GGML_VARIANT = 8,/* WR_SIMD_* bound for GGML-weight matmul*/
    WR_CTR_MATMUL_GGML_QUANT = 9,  /* kept-quantized weight fast-path hits  */
    WR_CTR_MATMUL_PERELEM    = 10, /* slow per-element fallback hits — a
                                    * nonzero delta here during decode means
                                    * a fast path was missed, not an error  */
    WR_CTR_MATMUL_PAR_N      = 11, /* N-split parallel matmul dispatches    */
    WR_COUNTER_COUNT         = 12
};

/* Fill out[0..WR_COUNTER_COUNT-1] with a relaxed-atomic snapshot. */
wr_status wr_engine_counters(const wr_engine *e,
                             uint64_t out[WR_COUNTER_COUNT]);

/* --------------------------------------------------------------------------
 * Model
 * -------------------------------------------------------------------------- */

/* Kept-quantized weight policy. */
enum {
    WR_QUANT_AUTO   = 0,  /* dequantize to F32 except very large tensors
                           * (embedding/LM head above the F32-blowup
                           * threshold stay in their quantized dtype) */
    WR_QUANT_KEEP   = 1,  /* keep every weight in its file dtype when the
                           * compute path supports it */
    WR_QUANT_F32    = 2   /* dequantize everything to F32 (large!) */
};

/* Sliding-window attention policy (Gemma-class models only). */
enum {
    WR_SWA_ENABLED  = 0,  /* real sliding-window masking on SWA layers
                           * (matches the reference implementation) */
    WR_SWA_DISABLED = 1   /* SWA layers attend to the full context
                           * (origin-parity mode used by the golden tests) */
};

/* Model load parameters.  Zero-initialize; zero fields mean "default". */
typedef struct wr_model_params {
    uint32_t max_context;   /* upper bound for sessions on this model.
                             * 0 = min(model's trained context, 4096).
                             * Values above the compiled attention sequence
                             * cap fail with WR_ERR_LIMIT — never clamped. */
    int32_t  keep_quantized;/* WR_QUANT_* (0 = WR_QUANT_AUTO) */
    int32_t  swa_mode;      /* WR_SWA_* (0 = WR_SWA_ENABLED) */
    int32_t  use_mmap;      /* nonzero: map the GGUF file and reference
                             * kept-quantized weight bytes in place instead
                             * of streaming them into the arena */
} wr_model_params;

/* Load a GGUF model.  Validates the file (version 2/3 only, tensor bounds
 * against the real file size, power-of-2 alignment), detects the
 * architecture, streams/maps weights, and commits the weight set.
 * `params` may be NULL for defaults.
 * Errors: WR_ERR_IO, WR_ERR_FORMAT (malformed file), WR_ERR_UNSUPPORTED
 * (unknown arch, or a REQUIRED tensor in a dtype the compute path lacks —
 * the log names the tensor and type), WR_ERR_LIMIT, WR_ERR_NOMEM. */
wr_status wr_model_load(wr_engine *e, const char *gguf_path,
                        const wr_model_params *params, wr_model **out);

/* Free a model.  All sessions and tokenizers created from it must already
 * be destroyed, else the call is refused (logged; model leaks).  `m` may
 * be NULL (no-op). */
void wr_model_free(wr_model *m);

/* Model description (filled by wr_model_get_info). */
typedef struct wr_model_info {
    char     arch[32];        /* GGUF architecture string ("llama", "qwen3",
                               * "gemma4", ...) */
    char     quant[16];       /* dominant weight dtype name ("Q4_K", "F16") */
    uint32_t n_layers;
    uint32_t n_q_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;        /* global-attention head dim */
    uint32_t head_dim_swa;    /* SWA-layer head dim (0 = same as head_dim) */
    uint32_t hidden_dim;
    uint32_t ffn_dim;         /* scalar FFN dim (per-layer dims may differ) */
    uint32_t vocab_size;
    uint32_t train_context;   /* context length from model metadata */
    uint32_t max_context;     /* effective session bound after params */
    float    rope_freq_base;  /* RoPE theta (10000 Llama, 1e6 Qwen3, ...) */
    float    rms_eps;         /* RMSNorm epsilon from metadata */
    float    logit_softcap;   /* 0 = disabled */
    uint32_t sliding_window;  /* SWA window size (0 = no SWA layers) */
    uint8_t  is_gemma;        /* Gemma-class sandwich-norm forward */
    uint8_t  has_qk_norm;     /* per-head Q/K RMSNorm (Qwen3, Gemma) */
    uint8_t  tied_lm_head;    /* LM head shares the embedding table */
    uint8_t  has_swa;         /* per-layer SWA/global pattern present */
} wr_model_info;

wr_status wr_model_get_info(const wr_model *m, wr_model_info *out);

/* --------------------------------------------------------------------------
 * Tokenizer
 * -------------------------------------------------------------------------- */

/* Build a tokenizer from the model's GGUF vocab/merges metadata.  The
 * tokenizer keeps its own copy of the vocab: it stays valid until
 * wr_tokenizer_free, independent of session activity, but must be freed
 * before its model. */
wr_status wr_tokenizer_from_model(const wr_model *m, wr_tokenizer **out);

void wr_tokenizer_free(wr_tokenizer *t);

/* wr_tokenize_ex flags. */
enum {
    WR_TOK_PARSE_SPECIAL = 1 << 0  /* recognize special-token strings in
                                    * `text` and emit their ids.  OFF by
                                    * default: user text can then never
                                    * inject control tokens. */
};

/* Encode UTF-8 text to token ids.  Returns the number of ids produced, or
 * a negative wr_status.  If `ids` is NULL or `max_ids` is 0, returns the
 * required count without writing.  If the result does not fit, returns
 * WR_ERR_LIMIT (nothing partial is written). */
int wr_tokenize(const wr_tokenizer *t, const char *text,
                uint32_t *ids, int max_ids);
int wr_tokenize_ex(const wr_tokenizer *t, const char *text,
                   uint32_t *ids, int max_ids, uint32_t flags);

/* Decode token ids to UTF-8 bytes in one pass.  Writes at most
 * `max_bytes - 1` bytes plus a NUL.  Returns bytes written (excluding
 * NUL), or a negative wr_status.  A too-small buffer truncates at the
 * last whole token boundary and returns the bytes written (graceful —
 * this is the one deliberate truncation in the API and it is signalled
 * by the return value reaching neither the full decode length nor an
 * error). */
int wr_detokenize(const wr_tokenizer *t, const uint32_t *ids, int n,
                  char *buf, int max_bytes);

/* Decode a single token id to its UTF-8 piece (normalized real bytes).
 * Returns bytes written (excluding NUL) or negative wr_status. */
int wr_token_piece(const wr_tokenizer *t, uint32_t id,
                   char *buf, int max_bytes);

/* Exact-string vocab lookup: the id of `token_text`, or -1 if absent.
 * This is the chat-template probe (e.g. "<|im_end|>"). */
int wr_token_id(const wr_tokenizer *t, const char *token_text);

int32_t wr_token_eos(const wr_tokenizer *t);   /* -1 if not declared */
int32_t wr_token_bos(const wr_tokenizer *t);   /* -1 if not declared */

/* --------------------------------------------------------------------------
 * Session — decode state
 * -------------------------------------------------------------------------- */

/* Session parameters.  Zero-initialize. */
typedef struct wr_session_params {
    uint32_t max_context;  /* override the model's max_context downward for
                            * this session (smaller KV allocation).
                            * 0 = model's max_context.  Values above the
                            * model bound fail with WR_ERR_INVAL. */
} wr_session_params;

/* Create a decode session: allocates the full per-layer F16 KV cache up
 * front (max_context rows — no reallocation during decode, no surprise
 * OOM mid-generation) plus the per-step scratch set.  `params` may be
 * NULL.  Multiple sessions per model are allowed; they share the
 * immutable weights. */
wr_status wr_session_create(wr_model *m, const wr_session_params *params,
                            wr_session **out);

/* Destroy a session.  Must not be called while another thread is inside a
 * decode call on this session. */
void wr_session_destroy(wr_session *s);

/* Bulk prefill WITHOUT computing LM-head logits (the projection is skipped
 * entirely — that is most of the per-token cost at prompt time).
 *
 *   Discipline: pass the first n_prompt-1 prompt tokens here, then feed
 *   the LAST prompt token to wr_step() to obtain the first logits.
 *   Feeding all n tokens here and the last one again via wr_step would
 *   append it to the KV cache twice and corrupt the context.
 *
 * Returns the number of tokens consumed (== n on success) or a negative
 * wr_status.  If n would exceed the context, consumes nothing and
 * returns WR_ERR_CTX_FULL. */
int wr_prefill(wr_session *s, const uint32_t *ids, uint32_t n);

/* One decode step: run the forward pass for `token` and return a pointer
 * to the F32 logits vector (length = vocab_size).  No copy: the pointer
 * refers to session-owned scratch and is valid until the next
 * wr_step / wr_prefill / wr_batch_step / wr_session_destroy on this
 * session.  Returns NULL on error; wr_session_status() then reports the
 * cause (WR_ERR_CTX_FULL when the KV cache is at max_context — the step
 * consumed nothing). */
const float *wr_step(wr_session *s, uint32_t token);

/* Maximum sessions per batched step. */
#define WR_BATCH_MAX 16

/* Batched decode across up to WR_BATCH_MAX sessions of the SAME model:
 * runs each session's layer stack, then coalesces the N LM-head
 * projections into a single GEMM and demuxes the results (bit-exact
 * versus N serial wr_step calls).  On success, logits_out[i] receives
 * session i's logits pointer under the same borrowed-view lifetime rules
 * as wr_step.  All sessions must be distinct and idle.
 * Returns the number of sessions stepped (== n), or a negative wr_status;
 * on error no session's KV cache has advanced. */
int wr_batch_step(wr_session *const *sessions, const uint32_t *tokens,
                  uint32_t n, const float **logits_out);

/* Tokens currently held in the KV cache (== positions consumed). */
uint32_t wr_session_pos(const wr_session *s);

/* The session's context bound. */
uint32_t wr_session_max_context(const wr_session *s);

/* Status of the most recent decode call on this session (WR_OK after a
 * successful call; the failure cause after wr_step returned NULL). */
wr_status wr_session_status(const wr_session *s);

/* --------------------------------------------------------------------------
 * Sampler
 * -------------------------------------------------------------------------- */

/* Sampling parameters — real floats.
 * Chain order: repetition penalty → temperature → top-k → top-p → draw.
 * temperature == 0 selects greedy argmax regardless of other fields (the
 * parity-proven default). */
typedef struct wr_sample_params {
    float    temperature;    /* 0 = greedy argmax; typical 0.2 .. 1.5 */
    int32_t  top_k;          /* 0 = disabled */
    float    top_p;          /* nucleus mass; <= 0 or >= 1 = disabled */
    float    repeat_penalty; /* 1.0 = disabled; > 1 penalizes recent tokens
                              * (positive logits divided by, negative
                              * multiplied by the penalty) */
    int32_t  repeat_last_n;  /* how many recent tokens the penalty sees;
                              * 0 disables the penalty */
    uint64_t seed;           /* PRNG seed; 0 = fixed default seed (runs are
                              * reproducible unless the caller varies it) */
} wr_sample_params;

/* All-defaults parameter set: greedy (temperature 0), everything else off. */
wr_sample_params wr_sample_params_default(void);

/* Create a sampler.  `p` may be NULL (defaults).  A sampler is
 * self-contained (RNG state + penalty history) and externally
 * synchronized: use one sampler per concurrent generation. */
wr_status wr_sampler_create(const wr_sample_params *p, wr_sampler **out);

void wr_sampler_free(wr_sampler *s);

/* Reset RNG (to `seed`) and clear the penalty history.  Call between
 * independent generations that reuse one sampler. */
void wr_sampler_reset(wr_sampler *s, uint64_t seed);

/* Sample one token id from `logits[0..vocab-1]`.  Applies the configured
 * chain, consults the mask callback if installed, records the choice in
 * the penalty history, and returns the token id (>= 0), or a negative
 * wr_status on invalid arguments.  Does not modify `logits`. */
int32_t wr_sample(wr_sampler *s, const float *logits, uint32_t vocab);

/* Install (or clear, with allow == NULL) the token-mask callback.
 * `tok` supplies the piece strings passed to the callback; it must
 * outlive the mask installation.  Pass tok == NULL only together with
 * allow == NULL. */
void wr_sampler_set_mask(wr_sampler *s, const wr_tokenizer *tok,
                         wr_token_mask_fn allow, void *user);

/* --------------------------------------------------------------------------
 * Generate — one-call facade
 * -------------------------------------------------------------------------- */

/* wr_generate_params.flags */
enum {
    WR_GEN_CHAT_TEMPLATE = 1 << 0, /* frame `prompt` with the model's
                                    * auto-detected chat template (ChatML
                                    * variants, Gemma turn markers) and stop
                                    * on the template's end marker (dropped
                                    * from the output text) */
    WR_GEN_RAW           = 1 << 1  /* verbatim prompt, no BOS injection */
};

/* Stop reasons. */
enum {
    WR_STOP_EOS        = 0,  /* model emitted its EOS token */
    WR_STOP_TEMPLATE   = 1,  /* chat-template stop marker matched */
    WR_STOP_MAX_TOKENS = 2,  /* max_tokens generated */
    WR_STOP_CTX_FULL   = 3,  /* context filled before another token fit */
    WR_STOP_CALLBACK   = 4   /* on_token returned nonzero */
};

typedef struct wr_generate_params {
    const char  *prompt;      /* UTF-8; required */
    uint32_t     max_tokens;  /* 0 = until EOS/stop/context */
    wr_sampler  *sampler;     /* NULL = greedy argmax */
    uint32_t     flags;       /* WR_GEN_* */
    int32_t      stop_token;  /* extra stop token id; <= 0 = none, so a
                               * zero-initialized struct is safe.  (Token
                               * id 0 therefore cannot serve as an extra
                               * stop token; EOS handling is unaffected.) */
    wr_token_cb  on_token;    /* optional streaming callback */
    void        *token_user;  /* opaque pointer for on_token */
} wr_generate_params;

typedef struct wr_generate_result {
    char    *text;        /* generated text (stop markers dropped), UTF-8,
                           * NUL-terminated.  Owned by the caller: release
                           * with wr_free(). */
    uint32_t tokens_in;   /* prompt tokens consumed (after templating) */
    uint32_t tokens_out;  /* tokens generated */
    int32_t  stop_reason; /* WR_STOP_* */
} wr_generate_result;

/* Run tokenize → (template) → prefill(n-1) → step/sample loop → one-pass
 * detokenize on `s`.  The session's tokenizer is created internally on
 * first use and cached on the model.  On success fills `*out`; on error
 * `out->text` is NULL and nothing needs freeing.  The session retains the
 * conversation in its KV cache, so consecutive wr_generate calls on one
 * session continue the same context. */
wr_status wr_generate(wr_session *s, const wr_generate_params *p,
                      wr_generate_result *out);

/* Release a buffer the library handed to the caller (wr_generate_result
 * .text).  NULL is a no-op. */
void wr_free(void *ptr);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* WAYRUNTIME_H */
