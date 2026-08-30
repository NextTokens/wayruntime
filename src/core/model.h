/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * model.h — model construction: loader (GGUF → weight slots) and the
 * validate/commit gate.
 *
 * Split of responsibilities:
 *   loader.c  walks the GGUF by canonical tensor names, decides
 *             kept-quantized vs dequant-to-F32, fills the wr_model
 *             weight slots and hyperparameters;
 *   model.c   owns wri_model_create/validate/commit/free — the single
 *             gate deciding whether a weight set is complete and
 *             consistent before any session may run.
 *
 * Everything here is called from wr_model_load; sessions only ever see a
 * committed model.
 */
#ifndef WR_MODEL_H
#define WR_MODEL_H

#include "core/internal.h"
#include "core/gguf.h"

/* --------------------------------------------------------------------------
 * Architecture identification
 * -------------------------------------------------------------------------- */

/* Forward-pass families. */
typedef enum wr_arch_class {
    WR_ARCH_UNKNOWN     = 0,
    WR_ARCH_LLAMA_CLASS = 1,  /* llama / qwen / qwen2 / mistral / gemma /
                               * gemma2 arch strings: RMSNorm + RoPE + GQA
                               * + SwiGLU */
    WR_ARCH_QWEN3       = 2,  /* llama-class + per-head Q/K RMSNorm before
                               * RoPE + decoupled head_dim.  Detected by
                               * arch string "qwen3" OR the presence of
                               * blk.0.attn_q_norm.weight */
    WR_ARCH_GEMMA4      = 3   /* sandwich norms (eps 1e-6, (1+w) pre-folded
                               * at load), GeGLU, embed scale sqrt(hidden),
                               * logit softcap, per-layer SWA/global with
                               * distinct head_dim + rope base, partial
                               * rotary, KV-sharing span, per-layer-input
                               * (PLE) MatFormer */
} wr_arch_class;

/* Classify from arch string + tensor presence probes. */
wr_arch_class wri_arch_classify(const wr_gguf *g);

/* --------------------------------------------------------------------------
 * Model lifecycle (model.c)
 * -------------------------------------------------------------------------- */

/* Allocate an empty, zero-initialized wr_model bound to the engine.
 * Zero-init matters: every slot's data == NULL until the loader fills it
 * (garbage in an unset gating field once produced a wrong forward path —
 * fail-closed, always). */
wr_model *wri_model_create(wr_engine *e);

/* Validate the filled weight set against the hyperparameters:
 *   - required slots present per architecture class (llama-class core
 *     set; + q_norm/k_norm for qwen3; + sandwich norms, PLE set,
 *     layer_output_scale for gemma4 — PLE is REQUIRED when the arch is
 *     gemma4-with-PLE-dims: a forward without it decodes garbage);
 *   - every slot's shape agrees with n_layers/hidden/heads/ffn dims
 *     (per-layer head_dim and ffn_dim honored);
 *   - matmul weight slots carry WR_TENSOR_GGML_WEIGHT; norm vectors are
 *     F32 and do NOT;
 *   - n_layers <= WR_MAX_LAYERS, max_context <= WR_ATTN_MAX_SEQ.
 * Returns WR_OK or WR_ERR_FORMAT/WR_ERR_INVAL/WR_ERR_LIMIT with a log
 * line naming the failing slot. */
int wri_model_validate(const wr_model *m);

/* Flip the model to committed (immutable).  Requires wri_model_validate
 * == WR_OK; sessions refuse uncommitted models (WR_ERR_STATE). */
int wri_model_commit(wr_model *m);

/* Tear down: refuses (WR_ERR_BUSY) while session_refs/tokenizer_refs are
 * nonzero; otherwise frees non-arena tensors, the arena, the cached
 * tokenizer, and the retained gguf mapping. */
int wri_model_destroy(wr_model *m);

/* --------------------------------------------------------------------------
 * Loader (loader.c)
 * -------------------------------------------------------------------------- */

/* Full load path backing wr_model_load: open+validate GGUF, classify
 * arch, size the arena, fill hyperparameters (floats from float
 * metadata; eps defaults 1e-5 llama-class / 1e-6 gemma), stream or map
 * weights into slots, then validate + commit.
 *
 * Weight-dtype policy (params->keep_quantized):
 *   norm vectors           always dequantized to F32
 *   matmul weights         WR_QUANT_AUTO: kept in file dtype when a
 *                          kept-quantized fast path exists (Q8_0/Q4_K/
 *                          Q6_K) or when dequantizing would exceed the
 *                          F32-blowup threshold (embedding/LM head above
 *                          WRI_EMBED_F32_LIMIT stay quantized); else F32.
 *   BF16 tensors           converted to F32 at load (v1 policy)
 *   parse-only types       WR_ERR_UNSUPPORTED naming tensor + type
 *
 * Tied LM head: when output.weight is absent, lm_head aliases
 * embed_table ([vocab, hidden] GGML layout) and m->tied_lm_head is set —
 * decode builds per-call transposed views (tensor.h). */
int wri_model_load_gguf(wr_engine *e, const char *path,
                        const wr_model_params *params, wr_model **out);

/* Embedding/LM-head tensors whose F32 expansion exceeds this stay
 * kept-quantized under WR_QUANT_AUTO. */
#define WRI_EMBED_F32_LIMIT (768ull * 1024 * 1024)

/* Canonical GGUF tensor names (loader.c resolves per layer with
 * snprintf("blk.%u.<stem>.weight")).  Global tensors:
 *   token_embd.weight            embedding [vocab, hidden]
 *   output_norm.weight           final norm [hidden]
 *   output.weight                untied LM head (optional)
 *   per_layer_token_embd.weight  gemma4 PLE table
 *   per_layer_model_proj.weight  gemma4 PLE projection
 *   per_layer_proj_norm.weight   gemma4 PLE norm
 * Per-layer stems:
 *   attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm,
 *   attn_output, post_attention_norm, post_norm, ffn_norm, ffn_gate,
 *   ffn_up, ffn_down, post_ffw_norm, inp_gate, proj, layer_output_scale
 * Per-layer FFN dim is read from each layer's ffn_gate dims (MatFormer);
 * per-layer head_dim from the SWA pattern (gemma4). */

#endif /* WR_MODEL_H */
