/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * loader.c — GGUF → wr_model weight loader.
 *
 * Orchestrates the whole of wr_model_load: open + validate the container
 * (gguf.c is the trust boundary — by the time this file runs, tensor
 * offsets/sizes are proven in-bounds), classify the architecture, resolve
 * hyperparameters, size the weight arena in one up-front pass, stream or
 * map every weight into its slot, then hand the filled model to the
 * validate/commit gate in model.c.
 *
 * Invariants owned here:
 *   - GGML weights load byte-verbatim in W[out][in] layout — never
 *     transposed; matmul reads them transposed via the flag model.c sets.
 *   - Dequantizing at load never changes element order, so dequantized
 *     F32 projections keep the same transposed layout (and the flag).
 *   - No silent fallbacks: a missing required tensor or a tensor in a
 *     dtype without a compute path is an error naming the tensor, never
 *     a zero-fill (the origin OS loader zero-filled on stream failure —
 *     that path is deliberately gone).
 *
 * Two-pass structure: pass 1 (plan) resolves every slot against the file,
 * decides kept-quantized vs dequant-to-F32 per the policy below, and
 * accumulates the exact arena requirement; pass 2 (fill) bump-allocates
 * and materializes.  Allocation can therefore never fail piecemeal
 * mid-load (tensor.h arena contract).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/internal.h"
#include "core/model.h"
#include "core/tensor.h"
#include "core/quant.h"

/* NOTE: core/mathx.h is deliberately NOT included here — its `wri_log`
 * (natural logarithm) collides with internal.h's `wri_log` logger, so the
 * two headers cannot appear in one translation unit.  The one square root
 * this file needs (the embedding scale) is computed with the same Newton
 * iteration the origin OS loader used, which also keeps libm out. */

/* sqrt(v) by Newton iteration on floats; v is a small model dimension
 * (hidden size), for which 24 iterations converge fully. */
static float loader_sqrtf(float v)
{
    if (v <= 0.0f)
        return 0.0f;
    float g = v;
    for (int i = 0; i < 24; i++)
        g = 0.5f * (g + v / g);
    return g;
}

/* --------------------------------------------------------------------------
 * Architecture classification
 * -------------------------------------------------------------------------- */

static int llama_class_match(const char *s)
{
    return strcmp(s, "llama") == 0 || strcmp(s, "qwen2") == 0 ||
           strcmp(s, "qwen") == 0  || strcmp(s, "qwen3") == 0 ||
           strcmp(s, "mistral") == 0 || strcmp(s, "gemma") == 0 ||
           strcmp(s, "gemma2") == 0;
}

wr_arch_class wri_arch_classify(const wr_gguf *g)
{
    if (!g)
        return WR_ARCH_UNKNOWN;
    const char *a = g->arch.architecture;
    if (!a[0])
        return WR_ARCH_UNKNOWN;
    if (strcmp(a, "gemma4") == 0)
        return WR_ARCH_GEMMA4;
    if (!llama_class_match(a))
        return WR_ARCH_UNKNOWN;
    /* Per-head Q/K RMSNorm marks the qwen3 forward variant even when the
     * arch string is generic — probe the first layer's q-norm tensor. */
    if (strcmp(a, "qwen3") == 0 ||
        wri_gguf_find_tensor(g, "blk.0.attn_q_norm.weight") >= 0)
        return WR_ARCH_QWEN3;
    return WR_ARCH_LLAMA_CLASS;
}

/* --------------------------------------------------------------------------
 * Slot plan
 * -------------------------------------------------------------------------- */

/* Policy class of a slot — decides both the kept-quantized policy and the
 * expected file-dim orientation:
 *   NORM    1-D vector, always dequantized to F32.
 *   MATMUL  projection weight; descriptor holds logical [K, N], which for
 *           a GGML file equals (dims[0], dims[1]) directly.
 *   EMBED   row-indexed table; descriptor [rows, width] = (dims[1], dims[0]).
 *           Kept quantized only above the F32-blowup threshold.
 *   LMHEAD  output projection; table orientation, but the matmul keep
 *           policy PLUS the blowup threshold apply.  */
enum { POL_NORM = 0, POL_MATMUL = 1, POL_EMBED = 2, POL_LMHEAD = 3 };

typedef struct slot_plan {
    char       name[WR_GGUF_NAME_MAX];
    wr_tensor *dst;
    int32_t    gi;        /* gguf tensor index */
    uint8_t    pol;
    uint8_t    keep;      /* copy/reference file bytes verbatim */
    uint8_t    ndim;
    wr_dtype   dtype;     /* in-memory dtype after load */
    uint32_t   shape[2];  /* descriptor shape */
    uint64_t   elems;
} slot_plan;

typedef struct load_ctx {
    wr_gguf        *g;
    wr_model       *m;
    wr_model_params p;
    int             use_map;    /* mapping succeeded; zero-copy where kept */
    slot_plan      *plan;
    uint32_t        n_plan;
    uint32_t        cap_plan;
    uint64_t        arena_need; /* bytes the arena must hold (64-aligned) */
    uint64_t        max_raw;    /* largest staged raw tensor (stream mode) */
} load_ctx;

/* --------------------------------------------------------------------------
 * Weight-dtype policy (see model.h for the contract)
 * -------------------------------------------------------------------------- */

static int resolve_dtype(const load_ctx *c, uint8_t pol, const char *name,
                         uint32_t wire, uint64_t elems,
                         wr_dtype *out_dt, uint8_t *out_keep)
{
    int d = wri_ggml_type_to_dtype(wire);
    if (d < 0) {
        /* Parse-only or unknown wire type on a tensor the model NEEDS:
         * honest refusal naming tensor and type — never a zero-fill. */
        wri_log_msg(0, "loader: tensor %s is stored as %s — no compute path "
                "for that type (WR_ERR_UNSUPPORTED)",
                name, wri_ggml_type_name(wire));
        return WR_ERR_UNSUPPORTED;
    }
    wr_dtype dt = (wr_dtype)d;

    /* Norm vectors are always F32, whatever the policy asks. */
    if (pol == POL_NORM || c->p.keep_quantized == WR_QUANT_F32) {
        *out_dt   = WR_DTYPE_F32;
        *out_keep = (dt == WR_DTYPE_F32);
        return WR_OK;
    }

    if (c->p.keep_quantized == WR_QUANT_KEEP) {
        if (dt == WR_DTYPE_BF16) {
            /* v1 policy: BF16 always converts to F32 at load. */
            *out_dt = WR_DTYPE_F32;
            *out_keep = 0;
        } else {
            *out_dt = dt;
            *out_keep = 1;
        }
        return WR_OK;
    }

    /* WR_QUANT_AUTO */
    if (dt == WR_DTYPE_F32) {
        *out_dt = dt;
        *out_keep = 1;
        return WR_OK;
    }
    /* F32-blowup guard: a quantized embedding/LM-head table whose F32
     * expansion exceeds the limit stays in its file dtype (the embed op
     * and the matmul both consume quantized rows on the fly). */
    if (dt >= WR_DTYPE_Q4_0 && (pol == POL_EMBED || pol == POL_LMHEAD) &&
        elems * 4u > WRI_EMBED_F32_LIMIT) {
        *out_dt = dt;
        *out_keep = 1;
        return WR_OK;
    }
    /* Kept-quantized fast path exists for these matmul weight dtypes. */
    if ((pol == POL_MATMUL || pol == POL_LMHEAD) &&
        ((dt == WR_DTYPE_Q8_0 && (elems % WR_QK8_0) == 0) ||
         ((dt == WR_DTYPE_Q4_K || dt == WR_DTYPE_Q6_K) &&
          (elems % WR_QK_K) == 0))) {
        *out_dt = dt;
        *out_keep = 1;
        return WR_OK;
    }
    *out_dt = WR_DTYPE_F32;
    *out_keep = 0;
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Plan pass
 * -------------------------------------------------------------------------- */

/* Resolve one slot against the file and append it to the plan.
 * Returns WR_OK (0) planned, 1 skipped (optional tensor absent), or a
 * negative status. */
static int add_slot(load_ctx *c, wr_tensor *dst, const char *name,
                    uint8_t pol, int required,
                    uint32_t ndim, uint32_t s0, uint32_t s1)
{
    int gi = wri_gguf_find_tensor(c->g, name);
    if (gi < 0) {
        if (!required)
            return 1;
        wri_log_msg(0, "loader: required tensor %s is missing", name);
        return WR_ERR_FORMAT;
    }
    const wr_gguf_tensor_info *ti = &c->g->tensors[gi];
    uint64_t elems  = wri_gguf_tensor_elements(ti);
    uint64_t expect = (uint64_t)s0 * (ndim == 2 ? s1 : 1u);
    if (elems != expect || expect == 0) {
        wri_log_msg(0, "loader: tensor %s has %llu elements, expected %llu",
                name, (unsigned long long)elems,
                (unsigned long long)expect);
        return WR_ERR_FORMAT;
    }
    if (ndim == 2) {
        /* The innermost (contiguous) file dim pins the orientation: `in`
         * for projection weights, `width` for row-indexed tables.  A
         * transposed file would pass the element-count check and then
         * silently corrupt the forward — reject it here. */
        uint32_t want_d0 = (pol == POL_EMBED || pol == POL_LMHEAD) ? s1 : s0;
        if (ti->n_dims < 2 || ti->dims[0] != want_d0) {
            wri_log_msg(0, "loader: tensor %s dims [%llu,%llu] do not match the "
                    "expected layout (innermost %u)", name,
                    (unsigned long long)(ti->n_dims >= 1 ? ti->dims[0] : 0),
                    (unsigned long long)(ti->n_dims >= 2 ? ti->dims[1] : 0),
                    want_d0);
            return WR_ERR_FORMAT;
        }
    }

    wr_dtype dt;
    uint8_t  keep = 0;
    int rc = resolve_dtype(c, pol, name, ti->type, elems, &dt, &keep);
    if (rc != WR_OK)
        return rc;

    if (keep) {
        /* A kept tensor's descriptor sizing must equal the validated file
         * storage exactly, or the bounds anchor (size_bytes) would lie. */
        uint64_t want = wri_dtype_bytes_for_count(dt, elems);
        if (want != ti->data_size) {
            wri_log_msg(0, "loader: tensor %s storage is %llu bytes, expected "
                    "%llu for %s", name,
                    (unsigned long long)ti->data_size,
                    (unsigned long long)want, wri_dtype_name(dt));
            return WR_ERR_FORMAT;
        }
    }

    if (c->n_plan >= c->cap_plan) {
        wri_log_msg(0, "loader: internal plan overflow at %s", name);
        return WR_ERR_INTERNAL;
    }
    slot_plan *sp = &c->plan[c->n_plan++];
    size_t nl = strlen(name);
    if (nl >= sizeof sp->name)
        nl = sizeof sp->name - 1;
    memcpy(sp->name, name, nl);
    sp->name[nl] = '\0';
    sp->dst      = dst;
    sp->gi       = gi;
    sp->pol      = pol;
    sp->keep     = keep;
    sp->ndim     = (uint8_t)ndim;
    sp->dtype    = dt;
    sp->shape[0] = s0;
    sp->shape[1] = s1;
    sp->elems    = elems;

    uint64_t bytes = keep ? ti->data_size : elems * 4u;
    if (!(keep && c->use_map))
        c->arena_need += (bytes + 63u) & ~63ull;
    if (!keep && !c->use_map && ti->data_size > c->max_raw)
        c->max_raw = ti->data_size;
    return WR_OK;
}

static int plan_weights(load_ctx *c)
{
    wr_model *m = c->m;
    char nb[96];
    int rc;

    rc = add_slot(c, &m->embed_table, "token_embd.weight", POL_EMBED, 1,
                  2, m->vocab_size, m->hidden_dim);
    if (rc < 0) return rc;
    rc = add_slot(c, &m->final_norm, "output_norm.weight", POL_NORM, 1,
                  1, m->hidden_dim, 0);
    if (rc < 0) return rc;
    /* Untied LM head when the file carries one; otherwise the head ties to
     * the embedding table after the fill pass. */
    rc = add_slot(c, &m->lm_head, "output.weight", POL_LMHEAD, 0,
                  2, m->vocab_size, m->hidden_dim);
    if (rc < 0) return rc;
    if (rc == 1)
        m->tied_lm_head = 1;

    int ple = (m->is_gemma && m->pl_emb_dim != 0);
    if (ple) {
        /* One per-layer-input table.  The origin OS split it into two
         * vocab halves purely to duck a kernel allocation cap; hosted
         * mappings have no such cap, so the split is gone.  PLE is
         * required fail-closed: the forward without it decodes garbage. */
        uint32_t plw = m->n_layers * m->pl_emb_dim;
        rc = add_slot(c, &m->per_layer_token_embd,
                      "per_layer_token_embd.weight", POL_EMBED, 1,
                      2, m->vocab_size, plw);
        if (rc < 0) return rc;
        rc = add_slot(c, &m->per_layer_model_proj,
                      "per_layer_model_proj.weight", POL_MATMUL, 1,
                      2, m->hidden_dim, plw);
        if (rc < 0) return rc;
        rc = add_slot(c, &m->per_layer_proj_norm,
                      "per_layer_proj_norm.weight", POL_NORM, 1,
                      1, m->pl_emb_dim, 0);
        if (rc < 0) return rc;
    }

    for (uint32_t l = 0; l < m->n_layers; l++) {
        /* Per-layer head dim: SWA layers may use a distinct head_dim. */
        uint32_t hd = (m->swa_pattern[l] && m->head_dim_swa)
                      ? m->head_dim_swa : m->head_dim;

        /* Per-layer FFN dim comes from THIS layer's ffn_gate output dim —
         * variable-depth FFN models vary it by layer, and the tensor is
         * the ground truth even when the metadata is scalar. */
        snprintf(nb, sizeof nb, "blk.%u.ffn_gate.weight", l);
        uint32_t ffn_l = m->ffn_dim;
        {
            int gi = wri_gguf_find_tensor(c->g, nb);
            if (gi >= 0 && c->g->tensors[gi].n_dims >= 2)
                ffn_l = (uint32_t)c->g->tensors[gi].dims[1];
        }
        if (ffn_l == 0) {
            wri_log_msg(0, "loader: layer %u has no FFN dim (no metadata and no "
                    "ffn_gate tensor)", l);
            return WR_ERR_FORMAT;
        }
        m->ffn_dim_per_layer[l] = ffn_l;
        if (m->ffn_dim == 0)
            m->ffn_dim = ffn_l;

        snprintf(nb, sizeof nb, "blk.%u.attn_norm.weight", l);
        rc = add_slot(c, &m->attn_norm[l], nb, POL_NORM, 1,
                      1, m->hidden_dim, 0);
        if (rc < 0) return rc;
        snprintf(nb, sizeof nb, "blk.%u.attn_q.weight", l);
        rc = add_slot(c, &m->wq[l], nb, POL_MATMUL, 1,
                      2, m->hidden_dim, m->n_q_heads * hd);
        if (rc < 0) return rc;
        snprintf(nb, sizeof nb, "blk.%u.attn_k.weight", l);
        rc = add_slot(c, &m->wk[l], nb, POL_MATMUL, 1,
                      2, m->hidden_dim, m->n_kv_heads * hd);
        if (rc < 0) return rc;
        snprintf(nb, sizeof nb, "blk.%u.attn_v.weight", l);
        rc = add_slot(c, &m->wv[l], nb, POL_MATMUL, 1,
                      2, m->hidden_dim, m->n_kv_heads * hd);
        if (rc < 0) return rc;

        if (m->has_qk_norm) {
            snprintf(nb, sizeof nb, "blk.%u.attn_q_norm.weight", l);
            rc = add_slot(c, &m->q_norm[l], nb, POL_NORM, 1, 1, hd, 0);
            if (rc < 0) return rc;
            snprintf(nb, sizeof nb, "blk.%u.attn_k_norm.weight", l);
            rc = add_slot(c, &m->k_norm[l], nb, POL_NORM, 1, 1, hd, 0);
            if (rc < 0) return rc;
        }

        snprintf(nb, sizeof nb, "blk.%u.attn_output.weight", l);
        rc = add_slot(c, &m->wo[l], nb, POL_MATMUL, 1,
                      2, m->n_q_heads * hd, m->hidden_dim);
        if (rc < 0) return rc;

        if (m->is_gemma) {
            snprintf(nb, sizeof nb, "blk.%u.post_attention_norm.weight", l);
            rc = add_slot(c, &m->post_attn_norm[l], nb, POL_NORM, 1,
                          1, m->hidden_dim, 0);
            if (rc < 0) return rc;
        }

        snprintf(nb, sizeof nb, "blk.%u.ffn_norm.weight", l);
        rc = add_slot(c, &m->ffn_norm[l], nb, POL_NORM, 1,
                      1, m->hidden_dim, 0);
        if (rc < 0) return rc;
        snprintf(nb, sizeof nb, "blk.%u.ffn_gate.weight", l);
        rc = add_slot(c, &m->w_gate[l], nb, POL_MATMUL, 1,
                      2, m->hidden_dim, ffn_l);
        if (rc < 0) return rc;
        snprintf(nb, sizeof nb, "blk.%u.ffn_up.weight", l);
        rc = add_slot(c, &m->w_up[l], nb, POL_MATMUL, 1,
                      2, m->hidden_dim, ffn_l);
        if (rc < 0) return rc;
        snprintf(nb, sizeof nb, "blk.%u.ffn_down.weight", l);
        rc = add_slot(c, &m->w_down[l], nb, POL_MATMUL, 1,
                      2, ffn_l, m->hidden_dim);
        if (rc < 0) return rc;

        if (m->is_gemma) {
            snprintf(nb, sizeof nb, "blk.%u.post_ffw_norm.weight", l);
            rc = add_slot(c, &m->post_ffw_norm[l], nb, POL_NORM, 1,
                          1, m->hidden_dim, 0);
            if (rc < 0) return rc;
        }
        if (ple) {
            snprintf(nb, sizeof nb, "blk.%u.post_norm.weight", l);
            rc = add_slot(c, &m->post_norm[l], nb, POL_NORM, 1,
                          1, m->hidden_dim, 0);
            if (rc < 0) return rc;
            snprintf(nb, sizeof nb, "blk.%u.inp_gate.weight", l);
            rc = add_slot(c, &m->inp_gate[l], nb, POL_MATMUL, 1,
                          2, m->hidden_dim, m->pl_emb_dim);
            if (rc < 0) return rc;
            snprintf(nb, sizeof nb, "blk.%u.proj.weight", l);
            rc = add_slot(c, &m->pl_proj[l], nb, POL_MATMUL, 1,
                          2, m->pl_emb_dim, m->hidden_dim);
            if (rc < 0) return rc;
            snprintf(nb, sizeof nb, "blk.%u.layer_output_scale.weight", l);
            rc = add_slot(c, &m->layer_output_scale[l], nb, POL_NORM, 1,
                          1, 1, 0);
            if (rc < 0) return rc;
        }
    }
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Fill pass
 * -------------------------------------------------------------------------- */

static int fill_slot(load_ctx *c, const slot_plan *sp, uint8_t *staging)
{
    wr_gguf *g = c->g;
    const wr_gguf_tensor_info *ti = &g->tensors[sp->gi];
    int rc;

    if (sp->keep && c->use_map) {
        /* Zero-copy: the descriptor references the read-only mapping.
         * Model weights are immutable after commit, so a read-only view
         * is exactly right; WR_TENSOR_ARENA keeps teardown off it. */
        const void *src = wri_gguf_tensor_ptr(g, (uint32_t)sp->gi);
        if (!src)
            return WR_ERR_INTERNAL;
        return wri_tensor_init(sp->dst, sp->dtype, sp->ndim, sp->shape, 0,
                               (void *)(uintptr_t)src);
    }

    if (sp->keep) {
        /* Verbatim byte stream into the arena — block formats included:
         * the on-disk layout IS the in-memory layout (mmap-parity). */
        void *dstb = wri_arena_alloc(c->m->arena, (size_t)ti->data_size);
        if (!dstb)
            return WR_ERR_INTERNAL;   /* sizing pass bug, not input data */
        rc = wri_gguf_read_tensor_data(g, (uint32_t)sp->gi, dstb,
                                       ti->data_size);
        if (rc != WR_OK)
            return rc;
        return wri_tensor_init(sp->dst, sp->dtype, sp->ndim, sp->shape, 0,
                               dstb);
    }

    /* Dequant/convert to F32.  Element order is preserved, so a converted
     * projection stays in the transposed GGML layout and keeps behaving
     * as a flagged weight. */
    float *dstf = wri_arena_alloc(c->m->arena,
                                  (size_t)(sp->elems * sizeof(float)));
    if (!dstf)
        return WR_ERR_INTERNAL;

    const void *src;
    if (c->use_map) {
        src = wri_gguf_tensor_ptr(g, (uint32_t)sp->gi);
        if (!src)
            return WR_ERR_INTERNAL;
    } else {
        rc = wri_gguf_read_tensor_data(g, (uint32_t)sp->gi, staging,
                                       ti->data_size);
        if (rc != WR_OK)
            return rc;
        src = staging;
    }

    switch (ti->type) {
    case WR_GGML_F32:
        memcpy(dstf, src, (size_t)(sp->elems * sizeof(float)));
        break;
    case WR_GGML_F16: {
        const uint16_t *h = (const uint16_t *)src;
        for (uint64_t i = 0; i < sp->elems; i++)
            dstf[i] = wri_f16_to_f32(h[i]);
        break;
    }
    case WR_GGML_BF16: {
        /* BF16 is the high half of an F32; widening is exact. */
        const uint16_t *h = (const uint16_t *)src;
        for (uint64_t i = 0; i < sp->elems; i++)
            dstf[i] = wri_bf16_to_f32(h[i]);
        break;
    }
    default: {
        int d = wri_ggml_type_to_dtype(ti->type);
        if (d < 0)
            return WR_ERR_UNSUPPORTED;   /* resolve_dtype already gated */
        rc = wri_dequant_row((wr_dtype)d, src, dstf, sp->elems);
        if (rc != WR_OK)
            return rc;
        break;
    }
    }
    return wri_tensor_init(sp->dst, WR_DTYPE_F32, sp->ndim, sp->shape, 0,
                           dstf);
}

/* --------------------------------------------------------------------------
 * Hyperparameters
 * -------------------------------------------------------------------------- */

static int fill_hparams(load_ctx *c, wr_arch_class cls)
{
    const wr_gguf_arch *a = &c->g->arch;
    wr_model *m = c->m;
    size_t i;

    for (i = 0; i + 1 < sizeof m->arch && a->architecture[i]; i++)
        m->arch[i] = a->architecture[i];
    m->arch[i] = '\0';

    m->n_layers = a->block_count;
    if (m->n_layers == 0) {
        wri_log_msg(0, "loader: %s.block_count missing or 0", m->arch);
        return WR_ERR_FORMAT;
    }
    if (m->n_layers > WR_MAX_LAYERS) {
        /* Documented compile-time cap — an error, never a silent clamp
         * (the origin OS loader quietly truncated deeper models). */
        wri_log_msg(0, "loader: model has %u layers; the compiled cap is %u "
                "(WR_ERR_LIMIT)", m->n_layers, (unsigned)WR_MAX_LAYERS);
        return WR_ERR_LIMIT;
    }

    m->hidden_dim = a->embedding_length;
    m->vocab_size = c->g->tokenizer.vocab_size;
    if (m->vocab_size == 0) {
        /* No tokenizer vocab in the metadata: the embedding table's outer
         * dim is the authoritative fallback. */
        int gi = wri_gguf_find_tensor(c->g, "token_embd.weight");
        if (gi >= 0 && c->g->tensors[gi].n_dims >= 2)
            m->vocab_size = (uint32_t)c->g->tensors[gi].dims[1];
    }
    if (m->hidden_dim == 0 || m->vocab_size == 0) {
        wri_log_msg(0, "loader: missing core metadata (hidden=%u vocab=%u)",
                m->hidden_dim, m->vocab_size);
        return WR_ERR_FORMAT;
    }

    m->n_q_heads  = a->attention_head_count ? a->attention_head_count : 1;
    m->n_kv_heads = a->attention_head_count_kv ? a->attention_head_count_kv
                                               : m->n_q_heads;
    if (m->n_q_heads % m->n_kv_heads != 0) {
        wri_log_msg(0, "loader: head_count %u not a multiple of head_count_kv "
                "%u", m->n_q_heads, m->n_kv_heads);
        return WR_ERR_FORMAT;
    }

    m->train_context  = a->context_length;
    m->ffn_dim        = a->feed_forward_length;  /* may be 0; per-layer
                                                  * dims fill in below   */
    m->rope_freq_base = (a->rope_freq_base > 0.0f) ? a->rope_freq_base
                                                   : 10000.0f;
    m->swa_mode       = (uint8_t)c->p.swa_mode;

    if (cls == WR_ARCH_GEMMA4) {
        m->is_gemma    = 1;
        m->has_qk_norm = 1;
        m->rms_eps = (a->attention_layer_norm_rms_epsilon > 0.0f)
                     ? a->attention_layer_norm_rms_epsilon : 1e-6f;
        m->head_dim = a->attention_key_length;
        if (m->head_dim == 0) {
            wri_log_msg(0, "loader: gemma4 requires %s.attention.key_length",
                    m->arch);
            return WR_ERR_FORMAT;
        }
        m->head_dim_swa       = a->attention_key_length_swa; /* 0 = same  */
        /* SWA layers rotate at their own base; when the file carries no
         * value the reference implementations (and the origin engine's
         * RoPE default) use 10000, NOT the global base. */
        m->rope_freq_base_swa = (a->rope_freq_base_swa >= 1.0f)
                                    ? a->rope_freq_base_swa : 10000.0f;
        m->sliding_window     = a->attention_sliding_window;
        m->kv_shared_layers   = a->attention_shared_kv_layers;
        if (m->kv_shared_layers >= m->n_layers) {
            wri_log_msg(0, "loader: shared_kv_layers %u >= block_count %u",
                    m->kv_shared_layers, m->n_layers);
            return WR_ERR_FORMAT;
        }
        m->pl_emb_dim = a->embedding_length_per_layer_input;
        /* The per-layer-input table is [vocab, n_layers * pl_emb_dim];
         * bound the folded width in 64-bit so every later 32-bit product
         * of the two is exact (the session slices it per layer). */
        if ((uint64_t)m->pl_emb_dim * (uint64_t)m->n_layers >
            (uint64_t)WRI_PLE_WIDTH_MAX) {
            wri_log_msg(0, "loader: per-layer input width %llu exceeds the "
                    "supported maximum %u (WR_ERR_LIMIT)",
                    (unsigned long long)m->pl_emb_dim *
                        (unsigned long long)m->n_layers,
                    (unsigned)WRI_PLE_WIDTH_MAX);
            return WR_ERR_LIMIT;
        }
        m->has_swa    = a->has_swa_pattern ? 1 : 0;
        for (uint32_t l = 0; l < m->n_layers; l++)
            m->swa_pattern[l] = (m->has_swa && a->swa_pattern[l]) ? 1 : 0;
        /* Embedding rows are scaled by sqrt(hidden). */
        m->embed_scale = loader_sqrtf((float)m->hidden_dim);
        /* Attention scale is exactly 1.0 for this family: the per-head
         * Q/K RMSNorm already controls the QK magnitude, so the score
         * takes NO extra scale.  1/sqrt(head_dim) here once shipped a
         * 16x-too-small score → flat softmax → confident garbage; keep
         * this explicit and never "simplify" it back to the default. */
        m->attn_scale    = 1.0f;
        m->logit_softcap = a->final_logit_softcapping;
        /* Partial rotary on GLOBAL layers: rotary factor 0.25 of head_dim
         * → head_dim/4 dims → head_dim/8 pairs.  SWA layers rotate fully
         * (the decode passes 0 for them). */
        m->n_rot_pairs = m->head_dim / 8;
    } else {
        m->is_gemma    = 0;
        m->has_qk_norm = (cls == WR_ARCH_QWEN3) ? 1 : 0;
        m->rms_eps = (a->attention_layer_norm_rms_epsilon > 0.0f)
                     ? a->attention_layer_norm_rms_epsilon : 1e-5f;
        /* Prefer the explicit per-head dim — qwen3 decouples head_dim
         * from hidden/n_heads; fall back to the classic convention. */
        m->head_dim = a->attention_key_length;
        if (m->head_dim == 0)
            m->head_dim = m->hidden_dim / m->n_q_heads;
        if (m->head_dim == 0)
            m->head_dim = 64;   /* origin-OS last-resort default */
        /* Honor a declared partial rotary dimension (full otherwise). */
        if (a->rope_dimension_count &&
            a->rope_dimension_count < m->head_dim)
            m->n_rot_pairs = a->rope_dimension_count / 2;
    }

    /* Effective session context bound.  An explicit request above the
     * compiled attention cap is an error — never clamped; the default is
     * bounded by the cap and the library default. */
    if (c->p.max_context) {
        if (c->p.max_context > WR_ATTN_MAX_SEQ) {
            wri_log_msg(0, "loader: max_context %u exceeds the attention "
                    "sequence cap %u (WR_ERR_LIMIT; never clamped)",
                    c->p.max_context, (unsigned)WR_ATTN_MAX_SEQ);
            return WR_ERR_LIMIT;
        }
        m->max_context = c->p.max_context;
    } else {
        uint32_t eff = m->train_context ? m->train_context
                                        : WR_DEFAULT_MAX_CONTEXT;
        if (eff > WR_DEFAULT_MAX_CONTEXT)
            eff = WR_DEFAULT_MAX_CONTEXT;
        if (eff > WR_ATTN_MAX_SEQ)
            eff = WR_ATTN_MAX_SEQ;
        m->max_context = eff;
    }
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Load entry points
 * -------------------------------------------------------------------------- */

int wri_model_load_gguf(wr_engine *e, const char *path,
                        const wr_model_params *params, wr_model **out)
{
    if (out)
        *out = NULL;
    if (!e || !path || !out)
        return WR_ERR_INVAL;

    load_ctx c;
    memset(&c, 0, sizeof c);
    if (params)
        c.p = *params;
    if (c.p.keep_quantized < WR_QUANT_AUTO ||
        c.p.keep_quantized > WR_QUANT_F32)
        return WR_ERR_INVAL;
    if (c.p.swa_mode != WR_SWA_ENABLED && c.p.swa_mode != WR_SWA_DISABLED)
        return WR_ERR_INVAL;
    if (c.p.use_mmap < WR_MMAP_DISABLED)
        return WR_ERR_INVAL;

    int rc = wri_gguf_open(path, &c.g);
    if (rc != WR_OK)
        return rc;

    wr_arch_class cls = wri_arch_classify(c.g);
    if (cls == WR_ARCH_UNKNOWN) {
        const char *as = c.g->arch.architecture;
        wri_log_msg(0, "loader: unsupported architecture \"%s\" — supported: "
                "llama / qwen / qwen2 / qwen3 / mistral / gemma / gemma2 / "
                "gemma4 (WR_ERR_UNSUPPORTED)", as[0] ? as : "(none)");
        wri_gguf_close(c.g);
        return WR_ERR_UNSUPPORTED;
    }

    c.m = wri_model_create(e);
    if (!c.m) {
        wri_gguf_close(c.g);
        return WR_ERR_NOMEM;
    }

    uint8_t *staging = NULL;

    /* Owned copy of the path: wr_tokenizer_from_model re-reads the vocab
     * metadata from here when the mapping is not retained. */
    size_t path_len = strlen(path) + 1;
    c.m->gguf_path = malloc(path_len);
    if (!c.m->gguf_path) {
        wri_gguf_close(c.g);
        c.g = NULL;
        rc = WR_ERR_NOMEM;
        goto fail;
    }
    memcpy(c.m->gguf_path, path, path_len);

    rc = fill_hparams(&c, cls);
    if (rc != WR_OK)
        goto fail;

    /* Zero-initialized parameters deliberately prefer mmap.  Positive
     * values retain the pre-tristate "nonzero enables mmap" behavior;
     * WR_MMAP_DISABLED is the sole explicit streaming opt-out. */
    if (c.p.use_mmap != WR_MMAP_DISABLED) {
        int mrc = wri_gguf_map_data(c.g);
        if (mrc == WR_OK)
            c.use_map = 1;
        else
            wri_log_msg(1, "loader: mapping %s failed (%s) — streaming instead",
                    path, wr_status_str((wr_status)mrc));
    }

    c.cap_plan = 6u + c.m->n_layers * 17u;
    c.plan = calloc(c.cap_plan, sizeof *c.plan);
    if (!c.plan) {
        rc = WR_ERR_NOMEM;
        goto fail;
    }

    rc = plan_weights(&c);
    if (rc != WR_OK)
        goto fail;

    /* One arena for everything that is not referenced in the mapping.
     * The plan pass computed the exact need; a configured reservation
     * must cover it or the load refuses honestly. */
    {
        uint64_t need = c.arena_need + 64u;
        if (e->cfg.arena_mb) {
            uint64_t cap = (uint64_t)e->cfg.arena_mb << 20;
            if (need > cap) {
                wri_log_msg(0, "loader: weights need %llu MiB but the engine "
                        "arena is configured at %u MiB (WR_ERR_NOMEM)",
                        (unsigned long long)((need >> 20) + 1),
                        e->cfg.arena_mb);
                rc = WR_ERR_NOMEM;
                goto fail;
            }
            need = cap;
        }
        c.m->arena = wri_arena_create((size_t)need);
        if (!c.m->arena) {
            rc = WR_ERR_NOMEM;
            goto fail;
        }
    }

    if (c.max_raw) {
        staging = malloc((size_t)c.max_raw);
        if (!staging) {
            rc = WR_ERR_NOMEM;
            goto fail;
        }
    }

    for (uint32_t s = 0; s < c.n_plan; s++) {
        rc = fill_slot(&c, &c.plan[s], staging);
        if (rc != WR_OK) {
            wri_log_msg(0, "loader: loading tensor %s failed (%s)",
                    c.plan[s].name, wr_status_str((wr_status)rc));
            goto fail;
        }
    }
    free(staging);
    staging = NULL;
    free(c.plan);
    c.plan = NULL;

    if (c.m->tied_lm_head) {
        /* The LM head aliases the embedding bytes: a descriptor copy that
         * keeps WR_TENSOR_ARENA, so teardown never frees the shared
         * storage twice.  Decode builds per-call [hidden, vocab] views of
         * this descriptor; the shared table itself is never reshaped. */
        c.m->lm_head = c.m->embed_table;
    }

    rc = wri_model_commit(c.m);
    if (rc != WR_OK)
        goto fail;

    if (c.use_map) {
        c.m->gguf = c.g;   /* weight bytes reference the mapping */
    } else {
        wri_gguf_close(c.g);
        c.g = NULL;
    }

    wri_log_msg(2, "loader: %s: arch=%s layers=%u hidden=%u heads=%u/%u "
            "head_dim=%u vocab=%u ffn=%u ctx=%u%s%s",
            path, c.m->arch, c.m->n_layers, c.m->hidden_dim,
            c.m->n_q_heads, c.m->n_kv_heads, c.m->head_dim,
            c.m->vocab_size, c.m->ffn_dim, c.m->max_context,
            c.m->tied_lm_head ? " tied-lm-head" : "",
            c.use_map ? " mmap" : "");

    *out = c.m;
    return WR_OK;

fail:
    free(staging);
    free(c.plan);
    /* Destroy the model BEFORE closing the container: mapped weight
     * tensors reference the mapping until their descriptors are freed. */
    if (c.m) {
        c.m->gguf = NULL;
        (void)wri_model_destroy(c.m);
    }
    if (c.g)
        wri_gguf_close(c.g);
    return rc;
}

wr_status wr_model_load(wr_engine *e, const char *gguf_path,
                        const wr_model_params *params, wr_model **out)
{
    if (out)
        *out = NULL;
    if (!e || !gguf_path || !out)
        return WR_ERR_INVAL;
    return (wr_status)wri_model_load_gguf(e, gguf_path, params, out);
}
