/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * session.c — the decode engine.
 *
 * A session is one autoregressive decode stream over a committed model:
 * a per-layer F16 KV cache plus a reusable per-step scratch set, all
 * carved from ONE wr_map_alloc slab at create time (no allocation during
 * decode, no surprise OOM mid-generation).  The forward pass supports
 * two families:
 *
 *   Llama/Qwen3    RMSNorm → Q/K/V → (per-head Q/K RMSNorm, Qwen3) →
 *                  RoPE → KV append → GQA → O-proj → residual →
 *                  RMSNorm → SwiGLU → residual
 *
 *   Gemma          sandwich norms (post-attention and post-feedforward
 *                  norms on the sublayer OUTPUT, pre-residual), GeGLU,
 *                  scaled embedding, per-layer head_dim / RoPE base by
 *                  SWA-vs-global layer type, partial rotary on global
 *                  layers, weightless V-RMSNorm before the KV append,
 *                  KV sharing (the trailing layers reuse the post-RoPE
 *                  K/V of the last non-shared layer of matching type),
 *                  per-layer-input (PLE) fold, per-layer output scale
 *                  applied DIRECT, and the final logit softcap.
 *
 * Concurrency: one mutex per session, held for the whole of every decode
 * call.  In the origin OS this was a spinlock + an in-flight counter +
 * a cache-reclaim shield; a plain mutex is the honest hosted equivalent.
 * Model weights are immutable after commit and are never mutated here —
 * in particular the tied LM head is used through a per-call transposed
 * VIEW (wri_tensor_view_2d), never by reshaping the shared descriptor
 * (the origin OS reshaped it in place, which races when several sessions
 * share one model).
 *
 * Context is a hard boundary: a step or prefill that would exceed
 * max_context returns WR_ERR_CTX_FULL and consumes NOTHING.  A failed
 * step rolls its partial KV appends back, so the cache never holds a
 * half-written position.
 */

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "core/internal.h"
#include "core/tensor.h"
#include "core/quant.h"

/* --------------------------------------------------------------------------
 * Small helpers
 * -------------------------------------------------------------------------- */

static uint64_t align64_u64(uint64_t x)
{
    return (x + 63u) & ~(uint64_t)63u;
}

/* Stack-local reshaped view of a session-owned scratch tensor.  The
 * canonical descriptors keep max-capacity shapes; ops see the per-layer
 * logical shape through one of these.  A view never owns bytes and is
 * never freed.  (Model tensors are NOT viewed through this helper — the
 * LM head goes through wri_tensor_view_2d.) */
static wr_tensor scratch_view(const wr_tensor *t, uint32_t ndim,
                              uint32_t d0, uint32_t d1, uint32_t d2)
{
    wr_tensor v = *t;
    v.ndim = (uint8_t)ndim;
    for (uint32_t i = 0; i < WR_TENSOR_MAX_DIMS; i++) v.shape[i] = 0;
    v.shape[0] = d0;
    if (ndim >= 2) v.shape[1] = d1;
    if (ndim >= 3) v.shape[2] = d2;
    v.flags |= WR_TENSOR_ARENA;                 /* views never own bytes */
    v.flags &= (uint16_t)~WR_TENSOR_GROWABLE;   /* scratch has no cursor */
    v.valid_rows = 0;
    return v;
}

static wr_tensor view1(const wr_tensor *t, uint32_t d0)
{
    return scratch_view(t, 1, d0, 0, 0);
}

static wr_tensor view2(const wr_tensor *t, uint32_t d0, uint32_t d1)
{
    return scratch_view(t, 2, d0, d1, 0);
}

static wr_tensor view3(const wr_tensor *t, uint32_t d0, uint32_t d1,
                       uint32_t d2)
{
    return scratch_view(t, 3, d0, d1, d2);
}

/* Newton-iteration square root for the PLE table scale (runs once per
 * token, cold path).  Ported verbatim from the origin engine so the
 * per-layer-input arithmetic stays bit-identical; mathx approximations
 * are NOT interchangeable here without regenerating goldens. */
static float newton_sqrtf(float x)
{
    if (x <= 0.0f) return 0.0f;
    float g = x > 1.0f ? x : 1.0f;
    for (int i = 0; i < 24; i++) g = 0.5f * (g + x / g);
    return g;
}

/* Per-layer head dim: Gemma SWA layers may run a narrower head than the
 * global layers.  MUST agree between KV sizing (create) and decode. */
static uint32_t layer_head_dim(const wr_model *m, uint32_t L)
{
    if (m->is_gemma && m->has_swa && m->swa_pattern[L] && m->head_dim_swa)
        return m->head_dim_swa;
    return m->head_dim;
}

/* Gemma KV sharing: the source layer whose post-RoPE cached K/V layer L
 * reads.  Non-shared layers (and non-gemma models) read their own cache
 * (returns L).  A shared layer reuses the LAST non-shared layer of its
 * own SWA/global type; its own wk/wv exist in the file but are dead.
 * If no layer of the matching type precedes the shared span (a config no
 * real model produces), the layer falls back to a fresh cache of its own
 * rather than attending over an empty one. */
static uint32_t kv_source_layer(const wr_model *m, uint32_t L, int local)
{
    if (!m->is_gemma || m->kv_shared_layers == 0 ||
        m->kv_shared_layers >= m->n_layers)
        return L;
    uint32_t first_shared = m->n_layers - m->kv_shared_layers;
    if (L < first_shared) return L;
    for (int cand = (int)first_shared - 1; cand >= 0; cand--) {
        int cand_local = (m->has_swa && m->swa_pattern[cand]) ? 1 : 0;
        if (cand_local == local) return (uint32_t)cand;
    }
    return L;
}

/* Append one [1, kv_heads, hd] F32 row into a growable F16 KV tensor at
 * its cursor.  Conversion is the TRUNCATING wri_f32_to_f16 (a pinned
 * parity decision — see quant.h). */
static wr_status kv_append_row(wr_tensor *kv, const wr_tensor *row,
                               uint32_t kv_heads, uint32_t hd)
{
    if (!kv->data || !row->data) return WR_ERR_INTERNAL;
    uint32_t rowpos = kv->valid_rows;
    if (rowpos >= kv->shape[0]) return WR_ERR_CTX_FULL;
    uint64_t row_elems = (uint64_t)kv_heads * hd;
    if (((uint64_t)rowpos + 1u) * row_elems * 2u > kv->size_bytes)
        return WR_ERR_INTERNAL;    /* sizing bug, not reachable from input */
    const float *src = (const float *)row->data;
    uint16_t    *dst = (uint16_t *)kv->data + (uint64_t)rowpos * row_elems;
    for (uint64_t i = 0; i < row_elems; i++)
        dst[i] = wri_f32_to_f16(src[i]);
    return wri_tensor_set_valid_rows(kv, rowpos + 1) == WR_OK
               ? WR_OK : WR_ERR_INTERNAL;
}

/* Roll every KV cursor back to the session's committed position.  Called
 * after a mid-step failure so the cache never keeps a partial position.
 * Alias slots (KV-shared layers) carry their own cursor copy which is
 * never read — resetting it too is harmless. */
void wri_session_kv_rollback(wr_session *s)
{
    for (uint32_t L = 0; L < s->model->n_layers; L++) {
        if (s->kv_k[L].data)
            (void)wri_tensor_set_valid_rows(&s->kv_k[L], s->pos);
        if (s->kv_v[L].data)
            (void)wri_tensor_set_valid_rows(&s->kv_v[L], s->pos);
    }
}

static wr_status op_fail(const char *what, uint32_t layer, int rc)
{
    wri_log_msg(0, "session: %s failed at layer %u (rc=%d)", what, layer, rc);
    return (rc < 0) ? (wr_status)rc : WR_ERR_INTERNAL;
}

/* --------------------------------------------------------------------------
 * The forward pass minus the LM head.
 *
 * Caller holds s->lock and has verified s->pos < s->max_context.  On
 * success the post-final-norm hidden state is in s->normed [hidden] and
 * each non-shared layer's KV cursor has advanced by one; s->pos is NOT
 * advanced (the caller does that after the LM head / batch coalesce).
 * On failure the caller must run wri_session_kv_rollback.
 *
 * Non-static without a header declaration by design: batch.c drives the
 * same body for the batched decode path (internal.h is frozen; this is
 * the decode-body/step seam the batched path needs, mirroring the origin
 * split).
 * -------------------------------------------------------------------------- */
wr_status wri_session_decode_body(wr_session *s, uint32_t token)
{
    const wr_model *m = s->model;
    const uint32_t hidden = m->hidden_dim;
    const uint32_t qheads = m->n_q_heads;
    const uint32_t kheads = m->n_kv_heads;
    const uint32_t pos    = s->pos;
    const float    eps    = m->rms_eps;
    const float    theta_default =
        (m->rope_freq_base > 0.0f) ? m->rope_freq_base : 10000.0f;
    int rc;

    /* Embed: residual ← embed_table[token].  Out-of-range token ids are
     * an error (WR_ERR_INVAL from the op), never clamped. */
    rc = wri_op_embed(&m->embed_table, token, &s->residual);
    if (rc != WR_OK) return op_fail("embed", 0, rc);

    /* Gemma scales the token embedding by ~sqrt(hidden) (a real float on
     * the model, resolved by the loader). */
    if (m->is_gemma && m->embed_scale != 0.0f) {
        float *r = (float *)s->residual.data;
        for (uint32_t j = 0; j < hidden; j++) r[j] *= m->embed_scale;
    }

    /* Gemma per-layer-input (PLE) precompute — once per token, on the
     * scaled embedding:
     *   proj  = per_layer_model_proj @ residual      [n_layers, pl_emb]
     *   proj  = RMSNorm(proj, per_layer_proj_norm)   (per row)
     *   table = per_layer_token_embd[token] * sqrt(pl_emb)
     *   ple_table ← (proj + table) * (1/sqrt(2))
     * The 1/sqrt(hidden) projection scale of the reference formulation
     * is deliberately omitted: the following RMSNorm is scale-invariant
     * and washes it out. */
    if (m->is_gemma && m->per_layer_token_embd.data &&
        m->per_layer_model_proj.data && m->per_layer_proj_norm.data &&
        s->ple_table.data && m->pl_emb_dim) {
        const uint32_t ple = m->pl_emb_dim;
        const uint32_t pl  = m->n_layers * ple;
        wr_tensor rv = view2(&s->residual, 1, hidden);
        wr_tensor pv = view2(&s->ple_proj, 1, pl);
        rc = wri_op_matmul(&rv, &m->per_layer_model_proj, &pv);
        if (rc != WR_OK) return op_fail("ple proj matmul", 0, rc);
        wr_tensor pr = view2(&s->ple_proj, m->n_layers, ple);
        rc = wri_op_rmsnorm(&pr, &m->per_layer_proj_norm, &pr, eps);
        if (rc != WR_OK) return op_fail("ple proj norm", 0, rc);
        rc = wri_op_embed(&m->per_layer_token_embd, token, &s->ple_table);
        if (rc != WR_OK) return op_fail("ple table embed", 0, rc);
        {
            const float tscale = newton_sqrtf((float)ple);  /* sqrt(256)=16 */
            const float comb   = 0.70710678f;               /* 1/sqrt(2)    */
            const float *pj = (const float *)s->ple_proj.data;
            float       *tb = (float *)s->ple_table.data;
            for (uint32_t j = 0; j < pl; j++)
                tb[j] = (pj[j] + tb[j] * tscale) * comb;
        }
    }

    for (uint32_t L = 0; L < m->n_layers; L++) {
        const uint32_t ffn_L =
            m->ffn_dim_per_layer[L] ? m->ffn_dim_per_layer[L] : m->ffn_dim;

        if (m->is_gemma) {
            /* ---- Gemma sandwich-norm block ---- */
            const int      local = (m->has_swa && m->swa_pattern[L]) ? 1 : 0;
            const uint32_t hd_L  = layer_head_dim(m, L);
            const float    theta_L =
                (local && m->rope_freq_base_swa > 0.0f)
                    ? m->rope_freq_base_swa : theta_default;
            /* Partial rotary only on GLOBAL layers; SWA layers rotate
             * every pair (0 = full). */
            const uint32_t n_rot = local ? 0u : m->n_rot_pairs;
            const uint32_t ksrc  = kv_source_layer(m, L, local);
            const int      shared = (ksrc != L);
            /* Real sliding-window masking, default ON; WR_SWA_DISABLED
             * reproduces the origin OS (which always passed 0) for the
             * ported goldens. */
            const uint32_t window =
                (local && m->swa_mode != WR_SWA_DISABLED)
                    ? m->sliding_window : 0u;

            /* A1: input norm */
            rc = wri_op_rmsnorm(&s->residual, &m->attn_norm[L], &s->normed,
                                eps);
            if (rc != WR_OK) return op_fail("attn norm", L, rc);

            /* A2: Q projection always; K/V only when this layer owns its
             * cache (shared layers reuse layer ksrc's cached K/V). */
            {
                wr_tensor nv = view2(&s->normed, 1, hidden);
                wr_tensor qv = view2(&s->q, 1, qheads * hd_L);
                rc = wri_op_matmul(&nv, &m->wq[L], &qv);
                if (rc != WR_OK) return op_fail("wq matmul", L, rc);
                if (!shared) {
                    wr_tensor kv2 = view2(&s->k_row, 1, kheads * hd_L);
                    rc = wri_op_matmul(&nv, &m->wk[L], &kv2);
                    if (rc != WR_OK) return op_fail("wk matmul", L, rc);
                    wr_tensor vv2 = view2(&s->v_row, 1, kheads * hd_L);
                    rc = wri_op_matmul(&nv, &m->wv[L], &vv2);
                    if (rc != WR_OK) return op_fail("wv matmul", L, rc);
                }
            }

            /* A3: per-head Q/K RMSNorm — Q always; K only when freshly
             * computed.  The (1+w) transform is pre-folded at load, so
             * the plain weighted RMSNorm is byte-identical. */
            if (m->has_qk_norm && m->q_norm[L].data && m->k_norm[L].data) {
                wr_tensor qh = view2(&s->q, qheads, hd_L);
                rc = wri_op_rmsnorm(&qh, &m->q_norm[L], &qh, eps);
                if (rc != WR_OK) return op_fail("q norm", L, rc);
                if (!shared) {
                    wr_tensor kh = view2(&s->k_row, kheads, hd_L);
                    rc = wri_op_rmsnorm(&kh, &m->k_norm[L], &kh, eps);
                    if (rc != WR_OK) return op_fail("k norm", L, rc);
                }
            }

            /* A4: RoPE — Q always; K only when freshly computed.  Shared
             * layers reuse the source layer's ALREADY-rotated cached K:
             * re-rotating it would double the angle. */
            {
                wr_tensor q3 = view3(&s->q, 1, qheads, hd_L);
                rc = wri_op_rope(&q3, &q3, pos, theta_L, n_rot);
                if (rc != WR_OK) return op_fail("q rope", L, rc);
            }
            if (!shared) {
                wr_tensor k3 = view3(&s->k_row, 1, kheads, hd_L);
                rc = wri_op_rope(&k3, &k3, pos, theta_L, n_rot);
                if (rc != WR_OK) return op_fail("k rope", L, rc);

                /* Weightless pure-RMS normalization of V before the
                 * append — without it the un-normed V blows up the
                 * attention output and the residual explodes. */
                wr_tensor v2 = view2(&s->v_row, kheads, hd_L);
                rc = wri_op_rmsnorm(&v2, NULL, &v2, eps);
                if (rc != WR_OK) return op_fail("v norm", L, rc);

                /* A5: append into THIS layer's cache (per-layer head_dim). */
                wr_status ast = kv_append_row(&s->kv_k[L], &s->k_row,
                                              kheads, hd_L);
                if (ast != WR_OK) return ast;
                ast = kv_append_row(&s->kv_v[L], &s->v_row, kheads, hd_L);
                if (ast != WR_OK) return ast;
            }

            /* A6: GQA over this layer's cache or the shared source's.
             * scale_override comes from the model (Gemma's explicit
             * attention scale — 1.0 for this family, NOT 1/sqrt(hd):
             * the per-head Q/K RMSNorm already controls the magnitude). */
            {
                wr_tensor q3 = view3(&s->q, 1, qheads, hd_L);
                wr_tensor a3 = view3(&s->attn, 1, qheads, hd_L);
                rc = wri_op_gqa_attention(&q3, &s->kv_k[ksrc],
                                          &s->kv_v[ksrc], &a3,
                                          m->attn_scale, window);
                if (rc != WR_OK) return op_fail("gqa", L, rc);
            }

            /* A7: output projection */
            {
                wr_tensor a2 = view2(&s->attn, 1, qheads * hd_L);
                wr_tensor p2 = view2(&s->ffn_proj, 1, hidden);
                rc = wri_op_matmul(&a2, &m->wo[L], &p2);
                if (rc != WR_OK) return op_fail("wo matmul", L, rc);
            }
            /* A8: post-attention norm on the sublayer OUTPUT, pre-residual */
            rc = wri_op_rmsnorm(&s->ffn_proj, &m->post_attn_norm[L],
                                &s->ffn_proj, eps);
            if (rc != WR_OK) return op_fail("post-attn norm", L, rc);
            /* A9: first residual */
            rc = wri_op_add(&s->residual, &s->ffn_proj, &s->residual);
            if (rc != WR_OK) return op_fail("attn residual", L, rc);

            /* F1: pre-feedforward norm */
            rc = wri_op_rmsnorm(&s->residual, &m->ffn_norm[L], &s->normed,
                                eps);
            if (rc != WR_OK) return op_fail("ffn norm", L, rc);
            /* F2: gate/up projections (per-layer FFN width) */
            {
                wr_tensor nv = view2(&s->normed, 1, hidden);
                wr_tensor g2 = view2(&s->ffn_gate, 1, ffn_L);
                rc = wri_op_matmul(&nv, &m->w_gate[L], &g2);
                if (rc != WR_OK) return op_fail("gate matmul", L, rc);
                wr_tensor u2 = view2(&s->ffn_up, 1, ffn_L);
                rc = wri_op_matmul(&nv, &m->w_up[L], &u2);
                if (rc != WR_OK) return op_fail("up matmul", L, rc);
            }
            /* F3: GeGLU — gelu_tanh(gate) ⊙ up */
            {
                wr_tensor g1 = view1(&s->ffn_gate, ffn_L);
                wr_tensor u1 = view1(&s->ffn_up, ffn_L);
                rc = wri_op_fused_gelu_mul(&g1, &u1, &g1);
                if (rc != WR_OK) return op_fail("geglu", L, rc);
            }
            /* F4: down projection */
            {
                wr_tensor g2 = view2(&s->ffn_gate, 1, ffn_L);
                wr_tensor p2 = view2(&s->ffn_proj, 1, hidden);
                rc = wri_op_matmul(&g2, &m->w_down[L], &p2);
                if (rc != WR_OK) return op_fail("down matmul", L, rc);
            }
            /* F5: post-feedforward norm on the OUTPUT, pre-residual */
            rc = wri_op_rmsnorm(&s->ffn_proj, &m->post_ffw_norm[L],
                                &s->ffn_proj, eps);
            if (rc != WR_OK) return op_fail("post-ffw norm", L, rc);
            /* F6: second residual */
            rc = wri_op_add(&s->residual, &s->ffn_proj, &s->residual);
            if (rc != WR_OK) return op_fail("ffn residual", L, rc);

            /* PLE fold: gate the raw post-residual hidden, gelu, multiply
             * by this layer's slice of the precomputed per-layer input,
             * project back, post-norm, add as a third residual. */
            if (m->per_layer_token_embd.data && m->inp_gate[L].data &&
                m->pl_proj[L].data && m->post_norm[L].data &&
                s->ple_table.data) {
                const uint32_t ple = m->pl_emb_dim;
                {
                    wr_tensor rv = view2(&s->residual, 1, hidden);
                    wr_tensor gv = view2(&s->ple_g, 1, ple);
                    rc = wri_op_matmul(&rv, &m->inp_gate[L], &gv);
                    if (rc != WR_OK) return op_fail("ple gate matmul", L, rc);
                }
                memcpy(s->ple_pli.data,
                       (const float *)s->ple_table.data + (uint64_t)L * ple,
                       (size_t)ple * sizeof(float));
                rc = wri_op_fused_gelu_mul(&s->ple_g, &s->ple_pli, &s->ple_g);
                if (rc != WR_OK) return op_fail("ple geglu", L, rc);
                {
                    wr_tensor gv = view2(&s->ple_g, 1, ple);
                    wr_tensor dv = view2(&s->ple_delta, 1, hidden);
                    rc = wri_op_matmul(&gv, &m->pl_proj[L], &dv);
                    if (rc != WR_OK) return op_fail("ple proj matmul", L, rc);
                }
                rc = wri_op_rmsnorm(&s->ple_delta, &m->post_norm[L],
                                    &s->ple_delta, eps);
                if (rc != WR_OK) return op_fail("ple post norm", L, rc);
                rc = wri_op_add(&s->residual, &s->ple_delta, &s->residual);
                if (rc != WR_OK) return op_fail("ple residual", L, rc);
            }

            /* Per-layer output scale, applied DIRECT — no (1+w) transform. */
            if (m->layer_output_scale[L].data) {
                float sc = wri_read_elem_f32(&m->layer_output_scale[L], 0);
                if (sc != 1.0f) {
                    float *r = (float *)s->residual.data;
                    for (uint32_t j = 0; j < hidden; j++) r[j] *= sc;
                }
            }
            continue;
        }

        /* ---- Llama/Qwen3 block ---- */
        const uint32_t hd = m->head_dim;

        rc = wri_op_rmsnorm(&s->residual, &m->attn_norm[L], &s->normed, eps);
        if (rc != WR_OK) return op_fail("attn norm", L, rc);

        {
            wr_tensor nv = view2(&s->normed, 1, hidden);
            wr_tensor qv = view2(&s->q, 1, qheads * hd);
            rc = wri_op_matmul(&nv, &m->wq[L], &qv);
            if (rc != WR_OK) return op_fail("wq matmul", L, rc);
            wr_tensor kv2 = view2(&s->k_row, 1, kheads * hd);
            rc = wri_op_matmul(&nv, &m->wk[L], &kv2);
            if (rc != WR_OK) return op_fail("wk matmul", L, rc);
            wr_tensor vv2 = view2(&s->v_row, 1, kheads * hd);
            rc = wri_op_matmul(&nv, &m->wv[L], &vv2);
            if (rc != WR_OK) return op_fail("wv matmul", L, rc);
        }

        /* Qwen3: per-head Q/K RMSNorm BEFORE RoPE.  q_norm/k_norm are
         * [head_dim] scales; normalizing [heads, hd] rows applies the
         * same scale to every head independently. */
        if (m->has_qk_norm && m->q_norm[L].data && m->k_norm[L].data) {
            wr_tensor qh = view2(&s->q, qheads, hd);
            rc = wri_op_rmsnorm(&qh, &m->q_norm[L], &qh, eps);
            if (rc != WR_OK) return op_fail("q norm", L, rc);
            wr_tensor kh = view2(&s->k_row, kheads, hd);
            rc = wri_op_rmsnorm(&kh, &m->k_norm[L], &kh, eps);
            if (rc != WR_OK) return op_fail("k norm", L, rc);
        }

        /* RoPE on Q and K at the absolute position. */
        {
            wr_tensor q3 = view3(&s->q, 1, qheads, hd);
            rc = wri_op_rope(&q3, &q3, pos, theta_default, m->n_rot_pairs);
            if (rc != WR_OK) return op_fail("q rope", L, rc);
            wr_tensor k3 = view3(&s->k_row, 1, kheads, hd);
            rc = wri_op_rope(&k3, &k3, pos, theta_default, m->n_rot_pairs);
            if (rc != WR_OK) return op_fail("k rope", L, rc);
        }

        /* Append K/V (F32→F16, truncating) into this layer's cache. */
        {
            wr_status ast = kv_append_row(&s->kv_k[L], &s->k_row, kheads, hd);
            if (ast != WR_OK) return ast;
            ast = kv_append_row(&s->kv_v[L], &s->v_row, kheads, hd);
            if (ast != WR_OK) return ast;
        }

        /* GQA attention over the cache (reads valid_rows, not shape[0]). */
        {
            wr_tensor q3 = view3(&s->q, 1, qheads, hd);
            wr_tensor a3 = view3(&s->attn, 1, qheads, hd);
            rc = wri_op_gqa_attention(&q3, &s->kv_k[L], &s->kv_v[L], &a3,
                                      0.0f, 0);
            if (rc != WR_OK) return op_fail("gqa", L, rc);
        }

        /* Output projection + residual. */
        {
            wr_tensor a2 = view2(&s->attn, 1, qheads * hd);
            wr_tensor p2 = view2(&s->ffn_proj, 1, hidden);
            rc = wri_op_matmul(&a2, &m->wo[L], &p2);
            if (rc != WR_OK) return op_fail("wo matmul", L, rc);
        }
        rc = wri_op_add(&s->residual, &s->ffn_proj, &s->residual);
        if (rc != WR_OK) return op_fail("attn residual", L, rc);

        /* FFN: RMSNorm → SwiGLU → down projection → residual. */
        rc = wri_op_rmsnorm(&s->residual, &m->ffn_norm[L], &s->normed, eps);
        if (rc != WR_OK) return op_fail("ffn norm", L, rc);
        {
            wr_tensor nv = view2(&s->normed, 1, hidden);
            wr_tensor g2 = view2(&s->ffn_gate, 1, ffn_L);
            rc = wri_op_matmul(&nv, &m->w_gate[L], &g2);
            if (rc != WR_OK) return op_fail("gate matmul", L, rc);
            wr_tensor u2 = view2(&s->ffn_up, 1, ffn_L);
            rc = wri_op_matmul(&nv, &m->w_up[L], &u2);
            if (rc != WR_OK) return op_fail("up matmul", L, rc);
        }
        {
            wr_tensor g1 = view1(&s->ffn_gate, ffn_L);
            wr_tensor u1 = view1(&s->ffn_up, ffn_L);
            rc = wri_op_fused_silu_mul(&g1, &u1, &g1);
            if (rc != WR_OK) return op_fail("swiglu", L, rc);
        }
        {
            wr_tensor g2 = view2(&s->ffn_gate, 1, ffn_L);
            wr_tensor p2 = view2(&s->ffn_proj, 1, hidden);
            rc = wri_op_matmul(&g2, &m->w_down[L], &p2);
            if (rc != WR_OK) return op_fail("down matmul", L, rc);
        }
        rc = wri_op_add(&s->residual, &s->ffn_proj, &s->residual);
        if (rc != WR_OK) return op_fail("ffn residual", L, rc);
    }

    /* Final RMSNorm — post-norm hidden lands in s->normed; the LM head is
     * the caller's job (single step: below; batched: coalesced). */
    rc = wri_op_rmsnorm(&s->residual, &m->final_norm, &s->normed, eps);
    if (rc != WR_OK) return op_fail("final norm", m->n_layers, rc);

    return WR_OK;
}

/* --------------------------------------------------------------------------
 * LM head: logits ← normed × lm_head, through a per-call transposed view.
 *
 * The lm_head slot is stored [vocab, hidden] (GGML layout, file-verbatim;
 * when tied it aliases the embedding).  The matmul needs a [hidden,
 * vocab]-shaped operand to make N = vocab — with the GGML flag the same
 * bytes are then read token-major, which is exactly the stored layout.
 * The shared descriptor is NEVER reshaped in place (that raced across
 * sessions in the origin OS); each call builds a private view.
 * -------------------------------------------------------------------------- */
static wr_status lm_head_project(wr_session *s)
{
    const wr_model *m = s->model;
    wr_tensor lmv;
    int rc = wri_tensor_view_2d(&m->lm_head, m->hidden_dim, m->vocab_size,
                                &lmv);
    if (rc != WR_OK) return op_fail("lm-head view", m->n_layers, rc);
    {
        wr_tensor nv = view2(&s->normed, 1, m->hidden_dim);
        wr_tensor lv = view2(&s->logits, 1, m->vocab_size);
        rc = wri_op_matmul(&nv, &lmv, &lv);
        if (rc != WR_OK) return op_fail("lm-head matmul", m->n_layers, rc);
    }
    /* Gemma final-logit soft-cap: x = cap * tanh(x / cap).  Applied in
     * the batched path too (batch.c). */
    if (m->is_gemma && m->logit_softcap > 0.0f)
        wri_softcap((float *)s->logits.data, m->vocab_size, m->logit_softcap);
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Decode entry points (internal.h contract)
 * -------------------------------------------------------------------------- */

wr_status wri_session_step(wr_session *s, uint32_t token, int compute_logits)
{
    if (!s) return WR_ERR_INVAL;
    wr_mutex_lock(s->lock);
    wr_status st;
    if (s->pos >= s->max_context) {
        st = WR_ERR_CTX_FULL;               /* consumed nothing */
    } else {
        st = wri_session_decode_body(s, token);
        if (st == WR_OK && compute_logits)
            st = lm_head_project(s);
        if (st == WR_OK)
            s->pos++;
        else
            wri_session_kv_rollback(s);     /* no partial positions */
    }
    s->last_status = st;
    wr_mutex_unlock(s->lock);
    return st;
}

int wri_session_prefill(wr_session *s, const uint32_t *ids, uint32_t n)
{
    if (!s) return WR_ERR_INVAL;
    if (n == 0) return 0;
    if (!ids) return WR_ERR_INVAL;

    wr_mutex_lock(s->lock);
    wr_status st = WR_OK;
    uint32_t consumed = 0;
    /* All-or-nothing context check: if the whole prompt cannot fit, the
     * call consumes NOTHING (never a silent partial prefill). */
    if ((uint64_t)s->pos + n > s->max_context) {
        st = WR_ERR_CTX_FULL;
    } else {
        for (uint32_t i = 0; i < n; i++) {
            st = wri_session_decode_body(s, ids[i]);
            if (st != WR_OK) { wri_session_kv_rollback(s); break; }
            s->pos++;
            consumed++;
        }
    }
    s->last_status = st;
    wr_mutex_unlock(s->lock);
    return (st == WR_OK) ? (int)consumed : (int)st;
}

/* --------------------------------------------------------------------------
 * Session lifecycle
 * -------------------------------------------------------------------------- */

/* One pass over the session's memory plan.  With base == NULL only the
 * total is accumulated; with a slab base the tensors are carved and
 * initialized.  The two passes MUST agree — this is the single source of
 * truth for both. */
static wr_status session_build(wr_session *s, const wr_model *m,
                               uint32_t max_ctx, uint8_t *base,
                               uint64_t *total_out)
{
    uint64_t cur = 0;
    int rc;

    /* KV cache: F16 [max_ctx, n_kv_heads, hd_L] per non-shared layer.
     * Shared layers alias their source's descriptors — no storage. */
    for (uint32_t L = 0; L < m->n_layers; L++) {
        const int local = (m->is_gemma && m->has_swa && m->swa_pattern[L])
                              ? 1 : 0;
        const uint32_t ksrc = kv_source_layer(m, L, local);
        if (ksrc != L) {
            if (base) {
                s->kv_k[L] = s->kv_k[ksrc];   /* ksrc < L: already carved */
                s->kv_v[L] = s->kv_v[ksrc];
            }
            continue;
        }
        const uint32_t hd_L = layer_head_dim(m, L);
        const uint64_t elems = (uint64_t)max_ctx * m->n_kv_heads * hd_L;
        const uint64_t bytes = wri_dtype_bytes_for_count(WR_DTYPE_F16, elems);
        uint32_t shape3[3] = { max_ctx, m->n_kv_heads, hd_L };
        for (int side = 0; side < 2; side++) {
            wr_tensor *t = side ? &s->kv_v[L] : &s->kv_k[L];
            void *p = base ? (void *)(base + cur) : NULL;
            cur += align64_u64(bytes);
            if (!base) continue;
            rc = wri_tensor_init(t, WR_DTYPE_F16, 3, shape3,
                                 WR_TENSOR_GROWABLE, p);
            if (rc != WR_OK) return (wr_status)rc;
            t->valid_rows = 0;
        }
    }

    /* Scratch set (all F32).  Head-dim'd scratches are sized for the
     * widest layer; per-layer views narrow them. */
    const uint32_t hd_max = (m->head_dim_swa > m->head_dim)
                                ? m->head_dim_swa : m->head_dim;
    uint32_t ffn_max = m->ffn_dim;
    for (uint32_t L = 0; L < m->n_layers; L++)
        if (m->ffn_dim_per_layer[L] > ffn_max)
            ffn_max = m->ffn_dim_per_layer[L];

    const int want_ple = (m->is_gemma && m->per_layer_token_embd.data &&
                          m->pl_emb_dim) ? 1 : 0;
    const uint32_t pl_total = want_ple ? m->n_layers * m->pl_emb_dim : 0;

    struct scratch_plan {
        wr_tensor *t;
        uint32_t   ndim, d0, d1, d2;
        int        enabled;
    } plan[15];
    uint32_t np = 0;
#define WRI_PLAN(tp, nd, a, b, c, en) \
    do { plan[np].t = (tp); plan[np].ndim = (nd); plan[np].d0 = (a);        \
         plan[np].d1 = (b); plan[np].d2 = (c); plan[np].enabled = (en);     \
         np++; } while (0)
    WRI_PLAN(&s->residual, 1, m->hidden_dim, 0, 0, 1);
    WRI_PLAN(&s->normed,   1, m->hidden_dim, 0, 0, 1);
    WRI_PLAN(&s->q,        3, 1, m->n_q_heads,  hd_max, 1);
    WRI_PLAN(&s->k_row,    3, 1, m->n_kv_heads, hd_max, 1);
    WRI_PLAN(&s->v_row,    3, 1, m->n_kv_heads, hd_max, 1);
    WRI_PLAN(&s->attn,     3, 1, m->n_q_heads,  hd_max, 1);
    WRI_PLAN(&s->ffn_gate, 1, ffn_max, 0, 0, 1);
    WRI_PLAN(&s->ffn_up,   1, ffn_max, 0, 0, 1);
    WRI_PLAN(&s->ffn_proj, 1, m->hidden_dim, 0, 0, 1);
    WRI_PLAN(&s->logits,   1, m->vocab_size, 0, 0, 1);
    WRI_PLAN(&s->ple_proj,  1, pl_total,      0, 0, want_ple);
    WRI_PLAN(&s->ple_table, 1, pl_total,      0, 0, want_ple);
    WRI_PLAN(&s->ple_g,     1, m->pl_emb_dim, 0, 0, want_ple);
    WRI_PLAN(&s->ple_pli,   1, m->pl_emb_dim, 0, 0, want_ple);
    WRI_PLAN(&s->ple_delta, 1, m->hidden_dim, 0, 0, want_ple);
#undef WRI_PLAN

    for (uint32_t i = 0; i < np; i++) {
        if (!plan[i].enabled) continue;
        uint64_t elems = (uint64_t)plan[i].d0;
        if (plan[i].ndim >= 2) elems *= plan[i].d1;
        if (plan[i].ndim >= 3) elems *= plan[i].d2;
        const uint64_t bytes = elems * 4u;
        void *p = base ? (void *)(base + cur) : NULL;
        cur += align64_u64(bytes);
        if (!base) continue;
        uint32_t shape[3] = { plan[i].d0, plan[i].d1, plan[i].d2 };
        rc = wri_tensor_init(plan[i].t, WR_DTYPE_F32, plan[i].ndim, shape,
                             0, p);
        if (rc != WR_OK) return (wr_status)rc;
    }

    *total_out = cur;
    return WR_OK;
}

wr_status wr_session_create(wr_model *m, const wr_session_params *params,
                            wr_session **out)
{
    if (!out) return WR_ERR_INVAL;
    *out = NULL;
    if (!m) return WR_ERR_INVAL;
    if (!m->weights_committed || !m->lock) return WR_ERR_STATE;
    if (m->n_layers == 0) return WR_ERR_INVAL;
    if (m->n_layers > WR_MAX_LAYERS) return WR_ERR_LIMIT;
    if (m->n_kv_heads == 0 || m->n_q_heads == 0 || m->head_dim == 0 ||
        m->hidden_dim == 0 || m->vocab_size == 0 || m->max_context == 0)
        return WR_ERR_INVAL;
    if (m->kv_shared_layers >= m->n_layers && m->kv_shared_layers != 0)
        return WR_ERR_INVAL;

    uint32_t max_ctx = m->max_context;
    if (params && params->max_context) {
        if (params->max_context > m->max_context) return WR_ERR_INVAL;
        max_ctx = params->max_context;
    }
    if (max_ctx > WR_ATTN_MAX_SEQ) return WR_ERR_LIMIT;

    wr_session *s = calloc(1, sizeof(*s));
    if (!s) return WR_ERR_NOMEM;
    s->model       = m;
    s->max_context = max_ctx;
    s->pos         = 0;
    s->last_status = WR_OK;

    s->lock = wr_mutex_create();
    if (!s->lock) { free(s); return WR_ERR_NOMEM; }

    /* Pass 1: size the slab.  Pass 2: carve it. */
    uint64_t total = 0;
    wr_status st = session_build(s, m, max_ctx, NULL, &total);
    if (st != WR_OK || total == 0 || total > (uint64_t)SIZE_MAX) {
        wr_mutex_destroy(s->lock);
        free(s);
        return (st != WR_OK) ? st : WR_ERR_INVAL;
    }
    s->slab = wr_map_alloc((size_t)total);
    if (!s->slab) {
        wri_log_msg(0, "session: slab alloc failed (%" PRIu64 " bytes)", total);
        wr_mutex_destroy(s->lock);
        free(s);
        return WR_ERR_NOMEM;
    }
    s->slab_bytes = (size_t)total;

    uint64_t carved = 0;
    st = session_build(s, m, max_ctx, (uint8_t *)s->slab, &carved);
    if (st != WR_OK || carved != total) {
        wr_map_free(s->slab, s->slab_bytes);
        wr_mutex_destroy(s->lock);
        free(s);
        return (st != WR_OK) ? st : WR_ERR_INTERNAL;
    }

    wr_mutex_lock(m->lock);
    m->session_refs++;
    wr_mutex_unlock(m->lock);

    *out = s;
    return WR_OK;
}

void wr_session_destroy(wr_session *s)
{
    if (!s) return;
    wr_model *m = s->model;
    if (m && m->lock) {
        wr_mutex_lock(m->lock);
        if (m->session_refs > 0) m->session_refs--;
        wr_mutex_unlock(m->lock);
    }
    /* Every tensor points into the slab (WR_TENSOR_ARENA) — one release. */
    if (s->slab) wr_map_free(s->slab, s->slab_bytes);
    if (s->lock) wr_mutex_destroy(s->lock);
    free(s);
}

/* --------------------------------------------------------------------------
 * Public decode surface
 * -------------------------------------------------------------------------- */

int wr_prefill(wr_session *s, const uint32_t *ids, uint32_t n)
{
    return wri_session_prefill(s, ids, n);
}

const float *wr_step(wr_session *s, uint32_t token)
{
    if (!s) return NULL;
    if (wri_session_step(s, token, 1) != WR_OK) return NULL;
    return (const float *)s->logits.data;
}

uint32_t wr_session_pos(const wr_session *s)
{
    return s ? s->pos : 0;
}

uint32_t wr_session_max_context(const wr_session *s)
{
    return s ? s->max_context : 0;
}

wr_status wr_session_status(const wr_session *s)
{
    return s ? s->last_status : WR_ERR_INVAL;
}

/* --------------------------------------------------------------------------
 * Golden self-test: the tiny-model decode fixture.
 *
 * Ported from the origin engine's step self-test: a 2-layer synthetic
 * model (4 q-heads, 2 kv-heads, head_dim 8, hidden 32, vocab 64, ffn 64,
 * context 16) with all-zero weights — the point is that every op in the
 * per-layer walk dispatches, the KV cursor advances, and the logits
 * vector is written.  The port adds the context-full contract check the
 * origin could not express (it silently failed the append instead).
 * -------------------------------------------------------------------------- */

static int fx_tensor(wr_tensor *t, wr_dtype dt, uint32_t ndim,
                     uint32_t d0, uint32_t d1, uint16_t flags)
{
    uint32_t shape[2] = { d0, d1 };
    return wri_tensor_init(t, dt, ndim, shape, flags, NULL);
}

static void fx_model_teardown(wr_model *m)
{
    wri_tensor_free(&m->embed_table);
    wri_tensor_free(&m->final_norm);
    wri_tensor_free(&m->lm_head);
    for (uint32_t L = 0; L < WR_MAX_LAYERS; L++) {
        wri_tensor_free(&m->attn_norm[L]);
        wri_tensor_free(&m->ffn_norm[L]);
        wri_tensor_free(&m->wq[L]);
        wri_tensor_free(&m->wk[L]);
        wri_tensor_free(&m->wv[L]);
        wri_tensor_free(&m->wo[L]);
        wri_tensor_free(&m->w_gate[L]);
        wri_tensor_free(&m->w_up[L]);
        wri_tensor_free(&m->w_down[L]);
    }
    if (m->lock) { wr_mutex_destroy(m->lock); m->lock = NULL; }
}

int wri_self_test_llm_step(void)
{
    enum { HID = 32, VOC = 64, FFN = 64, HD = 8, QH = 4, KVH = 2, CTX = 16 };
    wr_model fm;
    memset(&fm, 0, sizeof(fm));
    fm.engine       = wri_g_engine;
    fm.n_layers     = 2;
    fm.n_q_heads    = QH;
    fm.n_kv_heads   = KVH;
    fm.head_dim     = HD;
    fm.hidden_dim   = HID;
    fm.ffn_dim      = FFN;
    fm.vocab_size   = VOC;
    fm.train_context = CTX;
    fm.max_context  = CTX;
    fm.rms_eps      = 1e-5f;
    fm.rope_freq_base = 10000.0f;

    int ok = 0;
    wr_session *s = NULL;

    fm.lock = wr_mutex_create();
    if (!fm.lock) return -1;

    /* Weight set: all zeros.  GGML matmul weights use the logical [K, N]
     * shape; the LM head is stored file-verbatim [vocab, hidden] and the
     * decode path builds its transposed view. */
    if (fx_tensor(&fm.embed_table, WR_DTYPE_F32, 2, VOC, HID, 0) != WR_OK ||
        fx_tensor(&fm.final_norm, WR_DTYPE_F32, 1, HID, 0, 0) != WR_OK ||
        fx_tensor(&fm.lm_head, WR_DTYPE_F32, 2, VOC, HID,
                  WR_TENSOR_GGML_WEIGHT) != WR_OK)
        goto out;
    for (uint32_t L = 0; L < 2; L++) {
        if (fx_tensor(&fm.attn_norm[L], WR_DTYPE_F32, 1, HID, 0, 0) != WR_OK ||
            fx_tensor(&fm.ffn_norm[L],  WR_DTYPE_F32, 1, HID, 0, 0) != WR_OK ||
            fx_tensor(&fm.wq[L], WR_DTYPE_F32, 2, HID, QH * HD,
                      WR_TENSOR_GGML_WEIGHT) != WR_OK ||
            fx_tensor(&fm.wk[L], WR_DTYPE_F32, 2, HID, KVH * HD,
                      WR_TENSOR_GGML_WEIGHT) != WR_OK ||
            fx_tensor(&fm.wv[L], WR_DTYPE_F32, 2, HID, KVH * HD,
                      WR_TENSOR_GGML_WEIGHT) != WR_OK ||
            fx_tensor(&fm.wo[L], WR_DTYPE_F32, 2, QH * HD, HID,
                      WR_TENSOR_GGML_WEIGHT) != WR_OK ||
            fx_tensor(&fm.w_gate[L], WR_DTYPE_F32, 2, HID, FFN,
                      WR_TENSOR_GGML_WEIGHT) != WR_OK ||
            fx_tensor(&fm.w_up[L], WR_DTYPE_F32, 2, HID, FFN,
                      WR_TENSOR_GGML_WEIGHT) != WR_OK ||
            fx_tensor(&fm.w_down[L], WR_DTYPE_F32, 2, FFN, HID,
                      WR_TENSOR_GGML_WEIGHT) != WR_OK)
            goto out;
    }
    fm.weights_committed = 1;

    if (wr_session_create(&fm, NULL, &s) != WR_OK || !s) {
        wri_log_msg(0, "llm-step self-test: session create failed");
        goto out;
    }

    /* One decode step: every op dispatches, KV cursor 0→1, logits land. */
    if (wri_session_step(s, 1, 1) != WR_OK) {
        wri_log_msg(0, "llm-step self-test: step failed (%d)",
                (int)wr_session_status(s));
        goto out;
    }
    if (s->kv_k[0].valid_rows != 1 || s->kv_v[1].valid_rows != 1 ||
        s->pos != 1 || !s->logits.data) {
        wri_log_msg(0, "llm-step self-test: KV cursor/pos wrong after step");
        goto out;
    }

    /* Out-of-vocab token: refused (WR_ERR_INVAL), nothing consumed. */
    if (wri_session_step(s, VOC + 7, 1) != WR_ERR_INVAL || s->pos != 1 ||
        s->kv_k[0].valid_rows != 1) {
        wri_log_msg(0, "llm-step self-test: out-of-vocab token not refused");
        goto out;
    }

    /* Prefill discipline: 14 more without the LM head, then one step. */
    {
        uint32_t ids[14];
        for (uint32_t i = 0; i < 14; i++) ids[i] = (i % VOC);
        if (wri_session_prefill(s, ids, 14) != 14 || s->pos != 15) {
            wri_log_msg(0, "llm-step self-test: prefill failed");
            goto out;
        }
    }
    if (wri_session_step(s, 2, 1) != WR_OK || s->pos != CTX) {
        wri_log_msg(0, "llm-step self-test: step to full context failed");
        goto out;
    }

    /* Context exhaustion is a hard error that consumes nothing. */
    if (wri_session_step(s, 3, 1) != WR_ERR_CTX_FULL || s->pos != CTX ||
        s->kv_k[0].valid_rows != CTX ||
        wr_session_status(s) != WR_ERR_CTX_FULL) {
        wri_log_msg(0, "llm-step self-test: context-full contract violated");
        goto out;
    }

    ok = 1;
    wr_session_destroy(s);
    s = NULL;
    if (fm.session_refs != 0) {
        wri_log_msg(0, "llm-step self-test: session_refs leak");
        ok = 0;
    }

out:
    if (s) wr_session_destroy(s);
    fx_model_teardown(&fm);
    return ok ? 0 : -1;
}
