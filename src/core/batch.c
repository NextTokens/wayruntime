/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * batch.c — batched decode across sessions of one model.
 *
 * The batching primitive: run N sessions' layer stacks (sequentially —
 * KV and attention are inherently per-session), then COALESCE the N
 * LM-head projections into a single GEMM.  Each session contributes its
 * post-final-norm [1, hidden] row; the rows are concatenated into one
 * [N, hidden] staging tensor, multiplied once against the shared LM-head
 * view [hidden, vocab], and the [N, vocab] result is demuxed back into
 * each session's logits scratch.  For N sessions this replaces the N
 * most expensive matmuls of decode with one.
 *
 * Bit-exactness versus N serial steps is a hard contract (the golden
 * self-test compares raw bytes): the coalesced GEMM computes every
 * output row with the same per-column reduction order as the M=1 case,
 * because the matmul partitioning never splits a column's k-reduction
 * (see pool.h).
 *
 * The origin OS's cross-process defenses (per-entry owner ids) died with
 * the process boundary, and its 65536-logit output cap died with the
 * copy-out syscall — the cap barred large vocabularies from batching at
 * all.  There is NO vocab cap here: logits stay in session-owned scratch
 * at full width.
 */

#include <inttypes.h>
#include <string.h>

#include "core/internal.h"
#include "core/tensor.h"
#include "core/quant.h"

/* Internal seams shared with session.c (deliberately not in internal.h:
 * the decode body below the LM head is a private split between the
 * single-step and batched paths, exactly as in the origin engine). */
wr_status wri_session_decode_body(wr_session *s, uint32_t token);
void      wri_session_kv_rollback(wr_session *s);

/* --------------------------------------------------------------------------
 * Coalesced LM head: concat N [1, hidden] rows → ONE GEMM against the
 * shared LM-head view → demux [1, vocab] rows.
 * -------------------------------------------------------------------------- */
static wr_status coalesce_lm_head(wr_session *const *ss, uint32_t n)
{
    const wr_model *m = ss[0]->model;
    const uint32_t hidden = m->hidden_dim;
    const uint32_t vocab  = m->vocab_size;
    wr_status st = WR_OK;
    int rc;

    /* The shared LM head is stored [vocab, hidden] (GGML layout, aliases
     * the embedding when tied).  Build the per-call transposed view —
     * the shared descriptor is never mutated (the origin OS reshaped it
     * in place, racing concurrent sessions; that is the bug this design
     * removes). */
    wr_tensor lmv;
    rc = wri_tensor_view_2d(&m->lm_head, hidden, vocab, &lmv);
    if (rc != WR_OK) return (wr_status)rc;

    /* Staging tensors: big_in [n, hidden], big_out [n, vocab], F32. */
    wr_tensor big_in, big_out;
    memset(&big_in, 0, sizeof(big_in));
    memset(&big_out, 0, sizeof(big_out));
    {
        uint32_t shp_in[2] = { n, hidden };
        rc = wri_tensor_init(&big_in, WR_DTYPE_F32, 2, shp_in, 0, NULL);
        if (rc != WR_OK) return (wr_status)rc;
        uint32_t shp_out[2] = { n, vocab };
        rc = wri_tensor_init(&big_out, WR_DTYPE_F32, 2, shp_out, 0, NULL);
        if (rc != WR_OK) { wri_tensor_free(&big_in); return (wr_status)rc; }
    }

    /* Concat: session i's post-final-norm hidden row → big_in[i]. */
    for (uint32_t i = 0; i < n; i++)
        memcpy((float *)big_in.data + (uint64_t)i * hidden,
               ss[i]->normed.data, (size_t)hidden * sizeof(float));

    /* ONE matmul for all n sessions. */
    rc = wri_op_matmul(&big_in, &lmv, &big_out);
    if (rc != WR_OK) {
        wri_log_msg(0, "batch: coalesced lm-head matmul failed (rc=%d)", rc);
        st = (wr_status)rc;
        goto out;
    }

    /* Demux row i into session i's logits scratch; soft-cap per session
     * (the batched path must match the single path bit for bit, softcap
     * included). */
    for (uint32_t i = 0; i < n; i++) {
        memcpy(ss[i]->logits.data,
               (const float *)big_out.data + (uint64_t)i * vocab,
               (size_t)vocab * sizeof(float));
        if (m->is_gemma && m->logit_softcap > 0.0f)
            wri_softcap((float *)ss[i]->logits.data, vocab,
                        m->logit_softcap);
    }

out:
    wri_tensor_free(&big_in);
    wri_tensor_free(&big_out);
    return st;
}

/* Lock/unlock in a globally consistent order (ascending address) so two
 * overlapping batch calls cannot deadlock even though overlapping calls
 * already violate the one-driver-per-session contract. */
static void sort_by_address(wr_session **arr, uint32_t n)
{
    for (uint32_t i = 1; i < n; i++) {
        wr_session *key = arr[i];
        uint32_t j = i;
        while (j > 0 && (uintptr_t)arr[j - 1] > (uintptr_t)key) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = key;
    }
}

/* --------------------------------------------------------------------------
 * Batched decode step (internal.h contract)
 * -------------------------------------------------------------------------- */
int wri_batch_decode_step(wr_session *const *sessions, const uint32_t *tokens,
                          uint32_t n)
{
    if (!sessions || !tokens) return WR_ERR_INVAL;
    if (n == 0 || n > WR_BATCH_MAX) return WR_ERR_LIMIT;

    wr_model *m = sessions[0] ? sessions[0]->model : NULL;
    if (!m || !m->weights_committed) return WR_ERR_STATE;
    for (uint32_t i = 0; i < n; i++) {
        if (!sessions[i] || sessions[i]->model != m) return WR_ERR_INVAL;
        for (uint32_t j = 0; j < i; j++)
            if (sessions[i] == sessions[j]) return WR_ERR_INVAL;
    }

    wr_session *locked[WR_BATCH_MAX];
    for (uint32_t i = 0; i < n; i++) locked[i] = sessions[i];
    sort_by_address(locked, n);
    for (uint32_t i = 0; i < n; i++) wr_mutex_lock(locked[i]->lock);

    wr_status st = WR_OK;

    /* All-or-nothing context check across the whole batch: on failure no
     * session's KV cache has advanced. */
    for (uint32_t i = 0; i < n && st == WR_OK; i++)
        if (sessions[i]->pos >= sessions[i]->max_context)
            st = WR_ERR_CTX_FULL;

    /* Per-session decode bodies (sequential — KV/attention are
     * per-session state; only the LM head coalesces). */
    uint32_t advanced = 0;
    if (st == WR_OK) {
        for (uint32_t i = 0; i < n; i++) {
            st = wri_session_decode_body(sessions[i], tokens[i]);
            if (st != WR_OK) {
                wri_session_kv_rollback(sessions[i]);
                break;
            }
            advanced++;
        }
    }

    if (st == WR_OK)
        st = coalesce_lm_head(sessions, n);

    if (st == WR_OK) {
        for (uint32_t i = 0; i < n; i++) sessions[i]->pos++;
    } else {
        /* Unwind every session whose body completed this call. */
        for (uint32_t i = 0; i < advanced; i++)
            wri_session_kv_rollback(sessions[i]);
    }

    for (uint32_t i = 0; i < n; i++) sessions[i]->last_status = st;
    for (uint32_t i = n; i > 0; i--) wr_mutex_unlock(locked[i - 1]->lock);

    return (st == WR_OK) ? (int)n : (int)st;
}

/* --------------------------------------------------------------------------
 * Public surface
 * -------------------------------------------------------------------------- */
int wr_batch_step(wr_session *const *sessions, const uint32_t *tokens,
                  uint32_t n, const float **logits_out)
{
    int rc = wri_batch_decode_step(sessions, tokens, n);
    if (rc < 0) return rc;
    if (logits_out)
        for (uint32_t i = 0; i < n; i++)
            logits_out[i] = (const float *)sessions[i]->logits.data;
    return rc;
}

/* --------------------------------------------------------------------------
 * Golden self-test: batched == serial, bit-exact.
 *
 * Ported from the origin engine's batching oracle: one tiny synthetic
 * model (1 layer, 2 q-heads, 1 kv-head, head_dim 8, hidden 16, vocab 32,
 * ffn 32) whose embedding and LM head carry a deterministic non-trivial
 * pattern (zero weights would make the bit-exact check trivially true),
 * norms = 1.0, attention/FFN weights zero so the residual stream equals
 * the embedding and per-token logits are distinct.  Three sessions are
 * stepped one-by-one for reference logits, then three fresh sessions are
 * stepped through the batched path.  Proves:
 *   - per-session batched logits are BIT-EXACT to the serial path
 *     (raw-byte compare);
 *   - the LM head really coalesced: the batched phase issues exactly
 *     n-1 fewer matmuls than the serial phase (counter delta).
 * -------------------------------------------------------------------------- */
#define WRI_BSTEP_N 3u

static int fx_tensor(wr_tensor *t, wr_dtype dt, uint32_t ndim,
                     uint32_t d0, uint32_t d1, uint16_t flags)
{
    uint32_t shape[2] = { d0, d1 };
    return wri_tensor_init(t, dt, ndim, shape, flags, NULL);
}

/* Deterministic bounded values in [-0.5, 0.5) — keeps the matmuls
 * well-conditioned.  The formula is pinned by the origin fixture. */
static void fx_fill_pattern(wr_tensor *t, uint64_t count, uint32_t salt)
{
    float *p = (float *)t->data;
    for (uint64_t i = 0; i < count; i++) {
        int32_t v = (int32_t)(((i * 131u) + salt * 17u) % 101u) - 50;
        p[i] = (float)v * 0.01f;
    }
}

static void fx_fill_const(wr_tensor *t, uint64_t count, float val)
{
    float *p = (float *)t->data;
    for (uint64_t i = 0; i < count; i++) p[i] = val;
}

static void fx_model_teardown(wr_model *m)
{
    wri_tensor_free(&m->embed_table);
    wri_tensor_free(&m->final_norm);
    wri_tensor_free(&m->lm_head);
    wri_tensor_free(&m->attn_norm[0]);
    wri_tensor_free(&m->ffn_norm[0]);
    wri_tensor_free(&m->wq[0]);
    wri_tensor_free(&m->wk[0]);
    wri_tensor_free(&m->wv[0]);
    wri_tensor_free(&m->wo[0]);
    wri_tensor_free(&m->w_gate[0]);
    wri_tensor_free(&m->w_up[0]);
    wri_tensor_free(&m->w_down[0]);
    if (m->lock) { wr_mutex_destroy(m->lock); m->lock = NULL; }
}

int wri_self_test_batch_step(void)
{
    enum { HID = 16, VOC = 32, FFN = 32, HD = 8, QH = 2, KVH = 1, CTX = 16 };
    const uint32_t toks[WRI_BSTEP_N] = { 1, 7, 13 };
    float ref[WRI_BSTEP_N][VOC];
    float got[WRI_BSTEP_N][VOC];
    uint64_t c0[WR_COUNTER_COUNT], c1[WR_COUNTER_COUNT], c2[WR_COUNTER_COUNT];
    wr_session *bs[WRI_BSTEP_N] = { NULL, NULL, NULL };
    int ok = 0;

    wr_model fm;
    memset(&fm, 0, sizeof(fm));
    fm.engine        = wri_g_engine;
    fm.n_layers      = 1;
    fm.n_q_heads     = QH;
    fm.n_kv_heads    = KVH;
    fm.head_dim      = HD;
    fm.hidden_dim    = HID;
    fm.ffn_dim       = FFN;
    fm.vocab_size    = VOC;
    fm.train_context = CTX;
    fm.max_context   = CTX;
    fm.rms_eps       = 1e-5f;
    fm.rope_freq_base = 10000.0f;

    fm.lock = wr_mutex_create();
    if (!fm.lock) return -1;

    if (fx_tensor(&fm.embed_table, WR_DTYPE_F32, 2, VOC, HID, 0) != WR_OK ||
        fx_tensor(&fm.final_norm, WR_DTYPE_F32, 1, HID, 0, 0) != WR_OK ||
        fx_tensor(&fm.lm_head, WR_DTYPE_F32, 2, VOC, HID,
                  WR_TENSOR_GGML_WEIGHT) != WR_OK ||
        fx_tensor(&fm.attn_norm[0], WR_DTYPE_F32, 1, HID, 0, 0) != WR_OK ||
        fx_tensor(&fm.ffn_norm[0], WR_DTYPE_F32, 1, HID, 0, 0) != WR_OK ||
        fx_tensor(&fm.wq[0], WR_DTYPE_F32, 2, HID, QH * HD,
                  WR_TENSOR_GGML_WEIGHT) != WR_OK ||
        fx_tensor(&fm.wk[0], WR_DTYPE_F32, 2, HID, KVH * HD,
                  WR_TENSOR_GGML_WEIGHT) != WR_OK ||
        fx_tensor(&fm.wv[0], WR_DTYPE_F32, 2, HID, KVH * HD,
                  WR_TENSOR_GGML_WEIGHT) != WR_OK ||
        fx_tensor(&fm.wo[0], WR_DTYPE_F32, 2, QH * HD, HID,
                  WR_TENSOR_GGML_WEIGHT) != WR_OK ||
        fx_tensor(&fm.w_gate[0], WR_DTYPE_F32, 2, HID, FFN,
                  WR_TENSOR_GGML_WEIGHT) != WR_OK ||
        fx_tensor(&fm.w_up[0], WR_DTYPE_F32, 2, HID, FFN,
                  WR_TENSOR_GGML_WEIGHT) != WR_OK ||
        fx_tensor(&fm.w_down[0], WR_DTYPE_F32, 2, FFN, HID,
                  WR_TENSOR_GGML_WEIGHT) != WR_OK)
        goto out;

    /* Non-trivial embed + LM head so logits are distinct and non-zero;
     * norms = 1.0 so RMSNorm doesn't zero the stream; attention/FFN
     * weights stay zero so residual == embedding, giving deterministic
     * per-token logits to compare. */
    fx_fill_pattern(&fm.embed_table, (uint64_t)VOC * HID, 1);
    fx_fill_pattern(&fm.lm_head, (uint64_t)HID * VOC, 2);
    fx_fill_const(&fm.final_norm, HID, 1.0f);
    fx_fill_const(&fm.attn_norm[0], HID, 1.0f);
    fx_fill_const(&fm.ffn_norm[0], HID, 1.0f);
    fm.weights_committed = 1;

    /* Reference: N independent one-by-one steps. */
    wri_counters_snapshot(c0);
    for (uint32_t i = 0; i < WRI_BSTEP_N; i++) {
        wr_session *s = NULL;
        if (wr_session_create(&fm, NULL, &s) != WR_OK) {
            wri_log_msg(0, "batch self-test: ref session %u create failed", i);
            goto out;
        }
        if (wri_session_step(s, toks[i], 1) != WR_OK) {
            wri_log_msg(0, "batch self-test: ref step %u failed", i);
            wr_session_destroy(s);
            goto out;
        }
        memcpy(ref[i], s->logits.data, (size_t)VOC * sizeof(float));
        wr_session_destroy(s);
    }
    wri_counters_snapshot(c1);

    /* Batched: N fresh sessions stepped together. */
    for (uint32_t i = 0; i < WRI_BSTEP_N; i++) {
        if (wr_session_create(&fm, NULL, &bs[i]) != WR_OK) {
            wri_log_msg(0, "batch self-test: batch session %u create failed", i);
            goto out;
        }
    }
    {
        const float *louts[WRI_BSTEP_N] = { NULL, NULL, NULL };
        int brc = wr_batch_step(bs, toks, WRI_BSTEP_N, louts);
        if (brc != (int)WRI_BSTEP_N) {
            wri_log_msg(0, "batch self-test: wr_batch_step rc=%d (want %u)",
                    brc, WRI_BSTEP_N);
            goto out;
        }
        for (uint32_t i = 0; i < WRI_BSTEP_N; i++) {
            if (!louts[i] || bs[i]->pos != 1 ||
                bs[i]->kv_k[0].valid_rows != 1) {
                wri_log_msg(0, "batch self-test: session %u state wrong", i);
                goto out;
            }
            memcpy(got[i], louts[i], (size_t)VOC * sizeof(float));
        }
    }
    wri_counters_snapshot(c2);

    /* Bit-exact (raw bytes): batched logits == one-by-one logits. */
    for (uint32_t i = 0; i < WRI_BSTEP_N; i++) {
        if (memcmp(ref[i], got[i], (size_t)VOC * sizeof(float)) != 0) {
            wri_log_msg(0, "batch self-test: session %u logits NOT bit-exact", i);
            goto out;
        }
    }

    /* Coalescing witness: the batched phase must have issued exactly
     * n-1 fewer matmuls than the serial phase (n LM heads became one). */
    {
        uint64_t serial_mm = c1[WR_CTR_MATMUL_CALLS] - c0[WR_CTR_MATMUL_CALLS];
        uint64_t batch_mm  = c2[WR_CTR_MATMUL_CALLS] - c1[WR_CTR_MATMUL_CALLS];
        if (batch_mm + (WRI_BSTEP_N - 1) != serial_mm) {
            wri_log_msg(0, "batch self-test: LM head did not coalesce "
                    "(serial=%" PRIu64 " batched=%" PRIu64 " matmuls)",
                    serial_mm, batch_mm);
            goto out;
        }
    }

    ok = 1;

out:
    for (uint32_t i = 0; i < WRI_BSTEP_N; i++)
        if (bs[i]) wr_session_destroy(bs[i]);
    fx_model_teardown(&fm);
    return ok ? 0 : -1;
}
