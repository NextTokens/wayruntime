/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * model.c — model lifecycle: create / validate / commit / destroy.
 *
 * This file is the single gate deciding whether a weight set is complete
 * and consistent before any session may run.  The loader (loader.c)
 * fills the slots; nothing downstream ever sees an uncommitted model.
 *
 * Ported from the origin OS's model registry, minus everything that was
 * OS machinery there: the pid-ownership registry and its lifecycle FSM,
 * quota/profiler hooks, the persist-to-kernel ceremony, the weights
 * attestation (out of scope for v1 by decision), and a disk-driver
 * bit-clamp workaround the origin needed for large file offsets (hosted
 * file I/O does not have that bug).  What is kept — verbatim in intent —
 * is the weight-set validation and the canonical GGML-weight flagging
 * rules, both of which are load-bearing for GGUF compatibility.
 */
#include <stdlib.h>
#include <string.h>

#include "core/internal.h"
#include "core/model.h"
#include "core/tensor.h"
#include "core/quant.h"

/* --------------------------------------------------------------------------
 * Create
 * -------------------------------------------------------------------------- */

wr_model *wri_model_create(wr_engine *e)
{
    if (!e)
        return NULL;

    /* Zero-init matters: every slot's data == NULL until the loader fills
     * it, and every gating scalar (is_gemma, has_swa, head_dim_swa, ...)
     * starts at its documented "off" value.  The origin loader once
     * pattern-filled this struct and a garbage truthy gating byte sent
     * every non-matching model down the wrong forward path — fail-closed,
     * always. */
    wr_model *m = calloc(1, sizeof *m);
    if (!m)
        return NULL;

    m->lock = wr_mutex_create();
    if (!m->lock) {
        free(m);
        return NULL;
    }
    m->engine = e;
    e->models_live++;   /* create/destroy are externally synchronized */
    return m;
}

/* --------------------------------------------------------------------------
 * Validation helpers
 *
 * Every failure logs a line naming the failing slot (GGUF tensor naming,
 * so the message points straight back into the file).
 * -------------------------------------------------------------------------- */

static int v_missing(const char *stem, int layer)
{
    if (layer >= 0)
        wri_log_msg(0, "model: validate: blk.%d.%s.weight is missing", layer, stem);
    else
        wri_log_msg(0, "model: validate: %s.weight is missing", stem);
    return WR_ERR_FORMAT;
}

/* Norm-vector slot: present, F32, unflagged, exactly n elements. */
static int check_norm(const wr_tensor *t, const char *stem, int layer,
                      uint64_t n)
{
    if (!t->data || !t->size_bytes)
        return v_missing(stem, layer);
    if (t->dtype != WR_DTYPE_F32) {
        wri_log_msg(0, "model: validate: %s (layer %d) is %s — norm vectors must "
                "be F32", stem, layer, wri_dtype_name((wr_dtype)t->dtype));
        return WR_ERR_FORMAT;
    }
    if (t->flags & WR_TENSOR_GGML_WEIGHT) {
        wri_log_msg(0, "model: validate: %s (layer %d) carries the GGML-weight "
                "flag — norm vectors must not", stem, layer);
        return WR_ERR_INVAL;
    }
    if (wri_tensor_total_elements(t) != n) {
        wri_log_msg(0, "model: validate: %s (layer %d) has %llu elements, "
                "expected %llu", stem, layer,
                (unsigned long long)wri_tensor_total_elements(t),
                (unsigned long long)n);
        return WR_ERR_FORMAT;
    }
    return WR_OK;
}

/* Matmul-weight slot: present, GGML-flagged, descriptor [k, n].
 * The descriptor holds the LOGICAL matmul dims (k = reduce, n = out);
 * the bytes stay in file order W[out][in] and the flag tells matmul to
 * read them transposed — see internal.h invariant notes. */
static int check_weight(const wr_tensor *t, const char *stem, int layer,
                        uint32_t k, uint32_t n)
{
    if (!t->data || !t->size_bytes)
        return v_missing(stem, layer);
    if (!(t->flags & WR_TENSOR_GGML_WEIGHT)) {
        wri_log_msg(0, "model: validate: %s (layer %d) lacks the GGML-weight "
                "flag — matmul would read it untransposed", stem, layer);
        return WR_ERR_INVAL;
    }
    if (t->ndim != 2 || t->shape[0] != k || t->shape[1] != n) {
        wri_log_msg(0, "model: validate: %s (layer %d) shape [%u,%u] ndim=%u, "
                "expected [%u,%u]", stem, layer,
                t->ndim >= 1 ? t->shape[0] : 0,
                t->ndim >= 2 ? t->shape[1] : 0,
                (unsigned)t->ndim, k, n);
        return WR_ERR_FORMAT;
    }
    return WR_OK;
}

/* Row-indexed table slot (embedding-style): present, NOT GGML-flagged
 * (it is fetched row-by-row, never matmul'd), descriptor [rows, width]. */
static int check_table(const wr_tensor *t, const char *stem,
                       uint32_t rows, uint32_t width)
{
    if (!t->data || !t->size_bytes)
        return v_missing(stem, -1);
    if (t->flags & WR_TENSOR_GGML_WEIGHT) {
        wri_log_msg(0, "model: validate: %s carries the GGML-weight flag — "
                "indexed tables must not", stem);
        return WR_ERR_INVAL;
    }
    if (t->ndim != 2 || t->shape[0] != rows || t->shape[1] != width) {
        wri_log_msg(0, "model: validate: %s shape [%u,%u] ndim=%u, expected "
                "[%u,%u]", stem,
                t->ndim >= 1 ? t->shape[0] : 0,
                t->ndim >= 2 ? t->shape[1] : 0,
                (unsigned)t->ndim, rows, width);
        return WR_ERR_FORMAT;
    }
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Validate
 * -------------------------------------------------------------------------- */

int wri_model_validate(const wr_model *m)
{
    int rc;

    if (!m || !m->engine)
        return WR_ERR_INVAL;

    if (m->n_layers == 0) {
        wri_log_msg(0, "model: validate: n_layers is 0");
        return WR_ERR_INVAL;
    }
    if (m->n_layers > WR_MAX_LAYERS) {
        /* Documented hard cap: the weight-slot arrays are compiled at
         * WR_MAX_LAYERS.  The origin silently truncated deeper models;
         * here it is an error, never a clamp. */
        wri_log_msg(0, "model: validate: %u layers exceeds the compiled cap of "
                "%u (WR_ERR_LIMIT)", m->n_layers, (unsigned)WR_MAX_LAYERS);
        return WR_ERR_LIMIT;
    }
    if (!m->hidden_dim || !m->vocab_size || !m->n_q_heads ||
        !m->n_kv_heads || !m->head_dim) {
        wri_log_msg(0, "model: validate: incomplete dims (hidden=%u vocab=%u "
                "heads=%u/%u head_dim=%u)", m->hidden_dim, m->vocab_size,
                m->n_q_heads, m->n_kv_heads, m->head_dim);
        return WR_ERR_INVAL;
    }
    if (m->n_q_heads % m->n_kv_heads != 0) {
        wri_log_msg(0, "model: validate: n_q_heads %u is not a multiple of "
                "n_kv_heads %u", m->n_q_heads, m->n_kv_heads);
        return WR_ERR_INVAL;
    }
    if (m->max_context == 0) {
        wri_log_msg(0, "model: validate: max_context is 0");
        return WR_ERR_INVAL;
    }
    if (m->max_context > WR_ATTN_MAX_SEQ) {
        wri_log_msg(0, "model: validate: max_context %u exceeds the attention "
                "sequence cap %u (WR_ERR_LIMIT; never clamped)",
                m->max_context, (unsigned)WR_ATTN_MAX_SEQ);
        return WR_ERR_LIMIT;
    }
    if (m->is_gemma && m->kv_shared_layers >= m->n_layers) {
        wri_log_msg(0, "model: validate: kv_shared_layers %u >= n_layers %u — "
                "no source layer would remain", m->kv_shared_layers,
                m->n_layers);
        return WR_ERR_FORMAT;
    }

    rc = check_table(&m->embed_table, "token_embd",
                     m->vocab_size, m->hidden_dim);
    if (rc != WR_OK) return rc;
    rc = check_norm(&m->final_norm, "output_norm", -1, m->hidden_dim);
    if (rc != WR_OK) return rc;

    /* LM head keeps its file-order descriptor [vocab, hidden] and the
     * GGML-weight flag; decode builds a per-call [hidden, vocab] view
     * (tensor.h) — the shared descriptor is never mutated. */
    {
        const wr_tensor *t = &m->lm_head;
        if (!t->data || !t->size_bytes)
            return v_missing("output", -1);
        if (!(t->flags & WR_TENSOR_GGML_WEIGHT)) {
            wri_log_msg(0, "model: validate: output (lm_head) lacks the "
                    "GGML-weight flag");
            return WR_ERR_INVAL;
        }
        if (t->ndim != 2 || t->shape[0] != m->vocab_size ||
            t->shape[1] != m->hidden_dim) {
            wri_log_msg(0, "model: validate: output (lm_head) shape [%u,%u], "
                    "expected [%u,%u]",
                    t->ndim >= 1 ? t->shape[0] : 0,
                    t->ndim >= 2 ? t->shape[1] : 0,
                    m->vocab_size, m->hidden_dim);
            return WR_ERR_FORMAT;
        }
    }

    /* Per-layer-input (PLE) set — REQUIRED when the model declares PLE
     * dims: the forward without it decodes garbage, so this fails closed
     * rather than degrading. */
    int ple = (m->is_gemma && m->pl_emb_dim != 0);
    if (ple) {
        uint32_t plw = m->n_layers * m->pl_emb_dim;
        rc = check_table(&m->per_layer_token_embd, "per_layer_token_embd",
                         m->vocab_size, plw);
        if (rc != WR_OK) return rc;
        rc = check_weight(&m->per_layer_model_proj, "per_layer_model_proj",
                          -1, m->hidden_dim, plw);
        if (rc != WR_OK) return rc;
        rc = check_norm(&m->per_layer_proj_norm, "per_layer_proj_norm", -1,
                        m->pl_emb_dim);
        if (rc != WR_OK) return rc;
    }

    for (uint32_t l = 0; l < m->n_layers; l++) {
        int li = (int)l;
        /* Per-layer head dim: SWA layers may use a distinct (smaller)
         * head_dim; 0 means "same as global". */
        uint32_t hd = (m->has_swa && m->swa_pattern[l] && m->head_dim_swa)
                      ? m->head_dim_swa : m->head_dim;
        uint32_t ffn_l = m->ffn_dim_per_layer[l] ? m->ffn_dim_per_layer[l]
                                                 : m->ffn_dim;
        if (ffn_l == 0) {
            wri_log_msg(0, "model: validate: layer %u has no FFN dim", l);
            return WR_ERR_INVAL;
        }

        rc = check_norm(&m->attn_norm[l], "attn_norm", li, m->hidden_dim);
        if (rc != WR_OK) return rc;
        rc = check_weight(&m->wq[l], "attn_q", li,
                          m->hidden_dim, m->n_q_heads * hd);
        if (rc != WR_OK) return rc;
        rc = check_weight(&m->wk[l], "attn_k", li,
                          m->hidden_dim, m->n_kv_heads * hd);
        if (rc != WR_OK) return rc;
        rc = check_weight(&m->wv[l], "attn_v", li,
                          m->hidden_dim, m->n_kv_heads * hd);
        if (rc != WR_OK) return rc;
        rc = check_weight(&m->wo[l], "attn_output", li,
                          m->n_q_heads * hd, m->hidden_dim);
        if (rc != WR_OK) return rc;

        if (m->has_qk_norm) {
            rc = check_norm(&m->q_norm[l], "attn_q_norm", li, hd);
            if (rc != WR_OK) return rc;
            rc = check_norm(&m->k_norm[l], "attn_k_norm", li, hd);
            if (rc != WR_OK) return rc;
        }

        rc = check_norm(&m->ffn_norm[l], "ffn_norm", li, m->hidden_dim);
        if (rc != WR_OK) return rc;
        rc = check_weight(&m->w_gate[l], "ffn_gate", li,
                          m->hidden_dim, ffn_l);
        if (rc != WR_OK) return rc;
        rc = check_weight(&m->w_up[l], "ffn_up", li, m->hidden_dim, ffn_l);
        if (rc != WR_OK) return rc;
        rc = check_weight(&m->w_down[l], "ffn_down", li,
                          ffn_l, m->hidden_dim);
        if (rc != WR_OK) return rc;

        if (m->is_gemma) {
            rc = check_norm(&m->post_attn_norm[l], "post_attention_norm",
                            li, m->hidden_dim);
            if (rc != WR_OK) return rc;
            rc = check_norm(&m->post_ffw_norm[l], "post_ffw_norm", li,
                            m->hidden_dim);
            if (rc != WR_OK) return rc;
        }
        if (ple) {
            rc = check_norm(&m->post_norm[l], "post_norm", li,
                            m->hidden_dim);
            if (rc != WR_OK) return rc;
            rc = check_weight(&m->inp_gate[l], "inp_gate", li,
                              m->hidden_dim, m->pl_emb_dim);
            if (rc != WR_OK) return rc;
            rc = check_weight(&m->pl_proj[l], "proj", li,
                              m->pl_emb_dim, m->hidden_dim);
            if (rc != WR_OK) return rc;
            rc = check_norm(&m->layer_output_scale[l], "layer_output_scale",
                            li, 1);
            if (rc != WR_OK) return rc;
        }
    }

    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Commit
 * -------------------------------------------------------------------------- */

/* The canonical GGML-weight flag set (see the list in model.h).  These are
 * exactly the tensors matmul consumes: they stay in file layout W[out][in]
 * (mmap-verbatim, never transposed at load) and the flag makes matmul read
 * them transposed relative to a native row-major operand.  Norm vectors go
 * through rmsnorm and the embedding tables are row-indexed, so they stay
 * unflagged.  Getting one of these wrong silently transposes a projection
 * — this list is load-bearing, do not extend or trim it casually. */
static void apply_ggml_weight_flags(wr_model *m)
{
    if (m->lm_head.data)
        m->lm_head.flags |= WR_TENSOR_GGML_WEIGHT;

    for (uint32_t l = 0; l < m->n_layers && l < WR_MAX_LAYERS; l++) {
        wr_tensor *w[7] = {
            &m->wq[l], &m->wk[l], &m->wv[l], &m->wo[l],
            &m->w_gate[l], &m->w_up[l], &m->w_down[l]
        };
        for (int i = 0; i < 7; i++)
            if (w[i]->data)
                w[i]->flags |= WR_TENSOR_GGML_WEIGHT;
    }

    if (m->is_gemma) {
        if (m->per_layer_model_proj.data)
            m->per_layer_model_proj.flags |= WR_TENSOR_GGML_WEIGHT;
        for (uint32_t l = 0; l < m->n_layers && l < WR_MAX_LAYERS; l++) {
            if (m->inp_gate[l].data)
                m->inp_gate[l].flags |= WR_TENSOR_GGML_WEIGHT;
            if (m->pl_proj[l].data)
                m->pl_proj[l].flags |= WR_TENSOR_GGML_WEIGHT;
        }
    }
}

int wri_model_commit(wr_model *m)
{
    if (!m)
        return WR_ERR_INVAL;
    if (m->weights_committed)
        return WR_OK;

    /* Flag first, then validate: validation asserts the flags are in
     * place, so commit stays the one gate that both applies the canonical
     * rules and proves the weight set consistent. */
    apply_ggml_weight_flags(m);

    int rc = wri_model_validate(m);
    if (rc != WR_OK)
        return rc;

    m->weights_committed = 1;
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Destroy
 * -------------------------------------------------------------------------- */

static void free_all_slots(wr_model *m)
{
    /* Loader-built tensors are all arena- or mapping-backed, so
     * wri_tensor_free only zeroes their descriptors; individually owned
     * tensors (hand-built models in tests) are actually released.  A tied
     * LM head is a descriptor alias of the embedding — zero it instead of
     * freeing so shared bytes are never released twice. */
    if (m->tied_lm_head)
        memset(&m->lm_head, 0, sizeof m->lm_head);
    else
        wri_tensor_free(&m->lm_head);

    wri_tensor_free(&m->embed_table);
    wri_tensor_free(&m->final_norm);
    wri_tensor_free(&m->per_layer_token_embd);
    wri_tensor_free(&m->per_layer_model_proj);
    wri_tensor_free(&m->per_layer_proj_norm);

    for (uint32_t l = 0; l < WR_MAX_LAYERS; l++) {
        wri_tensor_free(&m->attn_norm[l]);
        wri_tensor_free(&m->wq[l]);
        wri_tensor_free(&m->wk[l]);
        wri_tensor_free(&m->wv[l]);
        wri_tensor_free(&m->wo[l]);
        wri_tensor_free(&m->ffn_norm[l]);
        wri_tensor_free(&m->w_gate[l]);
        wri_tensor_free(&m->w_up[l]);
        wri_tensor_free(&m->w_down[l]);
        wri_tensor_free(&m->q_norm[l]);
        wri_tensor_free(&m->k_norm[l]);
        wri_tensor_free(&m->post_attn_norm[l]);
        wri_tensor_free(&m->post_ffw_norm[l]);
        wri_tensor_free(&m->post_norm[l]);
        wri_tensor_free(&m->inp_gate[l]);
        wri_tensor_free(&m->pl_proj[l]);
        wri_tensor_free(&m->layer_output_scale[l]);
    }
}

int wri_model_destroy(wr_model *m)
{
    if (!m)
        return WR_OK;

    /* Refcounts and the cached tokenizer are read — and the cache taken
     * out — under the one lock that guards them (internal.h), so the
     * check and the act see the same state.  The lazily cached tokenizer
     * (wr_generate) holds one tokenizer_ref of its own; anything beyond
     * that is a live caller-owned child. */
    uint32_t srefs, trefs, cached;
    wr_tokenizer *cached_tok;
    wr_mutex_lock(m->lock);
    srefs = m->session_refs;
    trefs = m->tokenizer_refs;
    cached_tok = m->cached_tok;
    cached = cached_tok ? 1u : 0u;
    if (srefs == 0 && trefs <= cached)
        m->cached_tok = NULL;           /* taken: freed below */
    wr_mutex_unlock(m->lock);

    if (srefs != 0 || trefs > cached) {
        wri_log_msg(0, "model: destroy refused — %u session(s) and %u "
                "tokenizer(s) still alive (WR_ERR_BUSY)", srefs, trefs);
        return WR_ERR_BUSY;
    }

    if (cached_tok) {
        wr_tokenizer_free(cached_tok);
    }

    free_all_slots(m);

    if (m->arena) {
        wri_arena_destroy(m->arena);
        m->arena = NULL;
    }
    /* Unmap/close last: mapped weight tensors referenced the file mapping
     * until free_all_slots zeroed their descriptors. */
    if (m->gguf) {
        wri_gguf_close(m->gguf);
        m->gguf = NULL;
    }

    free(m->gguf_path);
    m->gguf_path = NULL;

    if (m->engine && m->engine->models_live)
        m->engine->models_live--;

    wr_mutex_destroy(m->lock);
    free(m);
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Public API: free + info
 * -------------------------------------------------------------------------- */

void wr_model_free(wr_model *m)
{
    if (!m)
        return;
    int rc = wri_model_destroy(m);
    if (rc != WR_OK)
        wri_log_msg(0, "wr_model_free: refused (%s) — the model leaks rather "
                "than freeing memory a child still references",
                wr_status_str((wr_status)rc));
}

/* Dominant loaded weight dtype by byte volume (what actually fills the
 * arena — reports the kept-quantized dtype when weights stayed in their
 * file format, F32 when the load policy dequantized them). */
static const char *dominant_dtype_name(const wr_model *m)
{
    uint64_t bytes[WR_DTYPE_COUNT] = { 0 };

    const wr_tensor *globals[5] = {
        &m->embed_table, &m->final_norm, &m->per_layer_token_embd,
        &m->per_layer_model_proj, &m->per_layer_proj_norm
    };
    for (int i = 0; i < 5; i++)
        if (globals[i]->data && globals[i]->dtype < WR_DTYPE_COUNT)
            bytes[globals[i]->dtype] += globals[i]->size_bytes;
    if (!m->tied_lm_head && m->lm_head.data &&
        m->lm_head.dtype < WR_DTYPE_COUNT)
        bytes[m->lm_head.dtype] += m->lm_head.size_bytes;

    for (uint32_t l = 0; l < m->n_layers && l < WR_MAX_LAYERS; l++) {
        const wr_tensor *w[17] = {
            &m->attn_norm[l], &m->wq[l], &m->wk[l], &m->wv[l], &m->wo[l],
            &m->ffn_norm[l], &m->w_gate[l], &m->w_up[l], &m->w_down[l],
            &m->q_norm[l], &m->k_norm[l], &m->post_attn_norm[l],
            &m->post_ffw_norm[l], &m->post_norm[l], &m->inp_gate[l],
            &m->pl_proj[l], &m->layer_output_scale[l]
        };
        for (int i = 0; i < 17; i++)
            if (w[i]->data && w[i]->dtype < WR_DTYPE_COUNT)
                bytes[w[i]->dtype] += w[i]->size_bytes;
    }

    int best = WR_DTYPE_F32;
    for (int d = 0; d < WR_DTYPE_COUNT; d++)
        if (bytes[d] > bytes[best])
            best = d;
    return wri_dtype_name((wr_dtype)best);
}

wr_status wr_model_get_info(const wr_model *m, wr_model_info *out)
{
    if (!m || !out)
        return WR_ERR_INVAL;

    memset(out, 0, sizeof *out);
    memcpy(out->arch, m->arch, sizeof out->arch);
    out->arch[sizeof out->arch - 1] = '\0';

    const char *qn = dominant_dtype_name(m);
    size_t i = 0;
    for (; qn[i] && i + 1 < sizeof out->quant; i++)
        out->quant[i] = qn[i];
    out->quant[i] = '\0';

    out->n_layers       = m->n_layers;
    out->n_q_heads      = m->n_q_heads;
    out->n_kv_heads     = m->n_kv_heads;
    out->head_dim       = m->head_dim;
    out->head_dim_swa   = m->head_dim_swa;
    out->hidden_dim     = m->hidden_dim;
    out->ffn_dim        = m->ffn_dim;
    out->vocab_size     = m->vocab_size;
    out->train_context  = m->train_context;
    out->max_context    = m->max_context;
    out->rope_freq_base = m->rope_freq_base;
    out->rms_eps        = m->rms_eps;
    out->logit_softcap  = m->logit_softcap;
    out->sliding_window = m->sliding_window;
    out->is_gemma       = m->is_gemma;
    out->has_qk_norm    = m->has_qk_norm;
    out->tied_lm_head   = m->tied_lm_head;
    out->has_swa        = m->has_swa;
    return WR_OK;
}
