/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * sample.c — token sampling.
 *
 * NEW CODE, not a port.  The origin OS transported sampling parameters
 * end-to-end but never read them: every live decode path there was
 * greedy argmax.  The semantics implemented here are fixed by sample.h
 * and differential-tested against llama.cpp's sampler:
 *
 *   chain:   repetition penalty → temperature → top-k → top-p → draw
 *   greedy:  temperature == 0 skips temperature/top-k/top-p/draw and
 *            takes argmax over the (penalized) logits — no RNG state is
 *            consumed, so the greedy stream is bit-stable regardless of
 *            seed.
 *   penalty: HF-style over a bounded recent-token window: positive
 *            logit /= penalty, negative logit *= penalty, applied at
 *            most ONCE per distinct token id in the window.
 *   draw:    categorical over the survivors' max-shifted softmax using
 *            the sampler's own xorshift64* stream.  All arithmetic is
 *            wri_exp + double accumulation in a fixed order, so equal
 *            seeds replay equal sequences on every supported host.
 *   mask:    the installed callback filters candidates before the
 *            draw; pieces handed to it are the tokenizer's normalized
 *            REAL bytes (the byte-level alphabet's whitespace
 *            surrogates are already reversed by wr_token_piece).  If
 *            every candidate is forbidden, the best forbidden token is
 *            returned — the grammar consumer is expected to stop.
 *
 * Edge cases by construction: top_k >= vocab and top_p >= 1 are
 * no-ops, at least one token always survives top-p, and every path
 * returns a valid id in [0, vocab) or a negative wr_status — never an
 * out-of-range id, never a crash.
 *
 * wr_sample never modifies the caller's logits: the chain operates on
 * a sampler-owned copy grown once to vocab size (no steady-state
 * allocation).
 */
#include <stdlib.h>
#include <string.h>

#include "core/internal.h"
#include "core/sample.h"

#include "core/mathx.h"

/* Fixed odd constant substituted for a zero RNG state/seed (xorshift
 * has no zero orbit). */
#define WRI_RNG_DEFAULT_SEED 0x9E3779B97F4A7C15ULL

/* Piece scratch for mask callbacks.  Pieces longer than this are
 * truncated before the callback sees them (vocab entries this long do
 * not occur in supported models). */
#define WRI_PIECE_BUF 4096

/* --------------------------------------------------------------------------
 * RNG — xorshift64*
 * -------------------------------------------------------------------------- */

uint64_t wri_rng_next(uint64_t *rng_state)
{
    uint64_t x = *rng_state;
    if (x == 0) x = WRI_RNG_DEFAULT_SEED;   /* zero has no xorshift orbit */
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Uniform double in [0, 1) from the top 53 bits of one RNG step. */
static double wri_rng_uniform(uint64_t *rng_state)
{
    return (double)(wri_rng_next(rng_state) >> 11) *
           (1.0 / 9007199254740992.0);      /* 2^-53 */
}

/* --------------------------------------------------------------------------
 * Argmax — the greedy path
 * -------------------------------------------------------------------------- */

int32_t wri_sample_argmax(const float *logits, uint32_t vocab)
{
    if (!logits || vocab == 0) return (int32_t)WR_ERR_INVAL;
    uint32_t best = 0;
    float best_v = logits[0];
    for (uint32_t i = 1; i < vocab; i++) {
        if (logits[i] > best_v) {           /* strict >: ties keep lower id */
            best_v = logits[i];
            best = i;
        }
    }
    return (int32_t)best;
}

/* --------------------------------------------------------------------------
 * Internals
 * -------------------------------------------------------------------------- */

/* Effective penalty window (ring capacity). */
static uint32_t wri_hist_window(const wr_sampler *s)
{
    uint32_t w = (uint32_t)s->p.repeat_last_n;
    if (w > WRI_SAMPLE_HISTORY_MAX) w = WRI_SAMPLE_HISTORY_MAX;
    return w;
}

/* Record a pick in the penalty ring (no-op when the window is 0). */
static void wri_hist_push(wr_sampler *s, uint32_t id)
{
    uint32_t w = wri_hist_window(s);
    if (w == 0) return;
    if (s->hist_len < w) {
        s->history[(s->hist_head + s->hist_len) % w] = id;
        s->hist_len++;
    } else {
        s->history[s->hist_head] = id;
        s->hist_head = (s->hist_head + 1) % w;
    }
}

/* HF-style repetition penalty over the history window, each distinct id
 * penalized once (a token seen N times is not penalized N-fold). */
static void wri_apply_repeat_penalty(const wr_sampler *s, float *l,
                                     uint32_t vocab)
{
    float pen = s->p.repeat_penalty;
    uint32_t w = wri_hist_window(s);
    if (pen == 1.0f || w == 0 || s->hist_len == 0) return;
    for (uint32_t i = 0; i < s->hist_len; i++) {
        uint32_t id = s->history[(s->hist_head + i) % w];
        int dup = 0;
        for (uint32_t j = 0; j < i && !dup; j++)
            if (s->history[(s->hist_head + j) % w] == id) dup = 1;
        if (dup || id >= vocab) continue;
        float v = l[id];
        l[id] = (v > 0.0f) ? v / pen : v * pen;
    }
}

/* Grow the working buffers to `vocab` elements (kept across calls). */
static wr_status wri_work_reserve(wr_sampler *s, uint32_t vocab)
{
    if (s->work_cap >= vocab) return WR_OK;
    float    *wl = (float *)malloc((size_t)vocab * sizeof(float));
    uint32_t *wi = (uint32_t *)malloc((size_t)vocab * sizeof(uint32_t));
    if (!wl || !wi) {
        free(wl);
        free(wi);
        return WR_ERR_NOMEM;
    }
    free(s->work_logits);
    free(s->work_ids);
    s->work_logits = wl;
    s->work_ids = wi;
    s->work_cap = vocab;
    return WR_OK;
}

/* Consult the mask callback for one token.  A piece that cannot be
 * decoded is treated as forbidden (it could never be matched by a
 * byte-level grammar anyway). */
static int wri_mask_allows(const wr_sampler *s, uint32_t id, char *piece,
                           int piece_cap)
{
    int n = wr_token_piece(s->mask_tok, id, piece, piece_cap);
    if (n < 0) return 0;
    return s->mask_fn(id, piece, s->mask_user) != 0;
}

/* Greedy argmax with the mask consulted for every id: highest-logit
 * ALLOWED token, or the highest-logit token overall when the mask
 * forbids everything (ties toward the lower id in both). */
static int32_t wri_masked_argmax(const wr_sampler *s, const float *l,
                                 uint32_t vocab)
{
    char piece[WRI_PIECE_BUF];
    int32_t best_allowed = -1;
    float best_allowed_v = 0.0f;
    uint32_t best_any = 0;
    float best_any_v = l[0];
    for (uint32_t id = 0; id < vocab; id++) {
        if (l[id] > best_any_v) {
            best_any_v = l[id];
            best_any = id;
        }
        if (!wri_mask_allows(s, id, piece, (int)sizeof(piece))) continue;
        if (best_allowed < 0 || l[id] > best_allowed_v) {
            best_allowed = (int32_t)id;
            best_allowed_v = l[id];
        }
    }
    return (best_allowed >= 0) ? best_allowed : (int32_t)best_any;
}

/* Descending in-place heapsort of ids[0..n) keyed by l[id]; equal
 * logits order by ascending id.  Deterministic, no allocation, no
 * comparator context (qsort_r is not portable C11). */
static int wri_cand_less(const float *l, uint32_t a, uint32_t b)
{
    /* "less" = sorts LATER in the descending output. */
    if (l[a] != l[b]) return l[a] < l[b];
    return a > b;
}

static void wri_sift_down(uint32_t *ids, const float *l, uint32_t root,
                          uint32_t last)
{
    for (;;) {
        uint32_t child = 2 * root + 1;
        if (child > last) break;
        if (child + 1 <= last && wri_cand_less(l, ids[child + 1], ids[child]))
            child++;
        if (!wri_cand_less(l, ids[child], ids[root])) break;
        uint32_t tmp = ids[root];
        ids[root] = ids[child];
        ids[child] = tmp;
        root = child;
    }
}

static void wri_sort_desc(uint32_t *ids, const float *l, uint32_t n)
{
    if (n < 2) return;
    for (uint32_t start = n / 2; start-- > 0;)
        wri_sift_down(ids, l, start, n - 1);     /* min-heapify */
    for (uint32_t end = n - 1; end > 0; end--) { /* min to the back → desc */
        uint32_t tmp = ids[0];
        ids[0] = ids[end];
        ids[end] = tmp;
        wri_sift_down(ids, l, 0, end - 1);
    }
}

/* --------------------------------------------------------------------------
 * Full chain
 * -------------------------------------------------------------------------- */

int32_t wri_sample_step(wr_sampler *s, const float *logits, uint32_t vocab)
{
    if (!s || !logits || vocab == 0) return (int32_t)WR_ERR_INVAL;

    const int mask_on = (s->mask_fn != NULL && s->mask_tok != NULL);
    const int greedy = (s->p.temperature == 0.0f);
    const int penalty_on = (s->p.repeat_penalty != 1.0f &&
                            wri_hist_window(s) > 0 && s->hist_len > 0);

    /* Greedy without a penalty needs no working copy at all. */
    if (greedy && !penalty_on) {
        int32_t id = mask_on ? wri_masked_argmax(s, logits, vocab)
                             : wri_sample_argmax(logits, vocab);
        if (id >= 0) wri_hist_push(s, (uint32_t)id);
        return id;
    }

    if (wri_work_reserve(s, vocab) != WR_OK) return (int32_t)WR_ERR_NOMEM;
    float *l = s->work_logits;
    memcpy(l, logits, (size_t)vocab * sizeof(float));

    wri_apply_repeat_penalty(s, l, vocab);

    if (greedy) {
        int32_t id = mask_on ? wri_masked_argmax(s, l, vocab)
                             : wri_sample_argmax(l, vocab);
        if (id >= 0) wri_hist_push(s, (uint32_t)id);
        return id;
    }

    /* Temperature scale (temperature > 0 was validated at create). */
    float inv_t = 1.0f / s->p.temperature;
    for (uint32_t i = 0; i < vocab; i++) l[i] *= inv_t;

    /* Candidate set starts as the whole vocab in id order. */
    uint32_t *ids = s->work_ids;
    for (uint32_t i = 0; i < vocab; i++) ids[i] = i;
    uint32_t n_cand = vocab;

    uint32_t k = (s->p.top_k > 0) ? (uint32_t)s->p.top_k : 0;
    int k_on = (k > 0 && k < vocab);                     /* k>=vocab: no-op */
    int p_on = (s->p.top_p > 0.0f && s->p.top_p < 1.0f); /* p>=1:     no-op */

    if (k_on || p_on) {
        wri_sort_desc(ids, l, n_cand);
        if (k_on) n_cand = k;
        if (p_on && n_cand > 1) {
            /* Smallest prefix of the sorted survivors whose softmax
             * mass reaches top_p; at least one token always survives. */
            float mx = l[ids[0]];            /* sorted: front is the max */
            double sum = 0.0;
            for (uint32_t i = 0; i < n_cand; i++)
                sum += (double)wri_exp(l[ids[i]] - mx);
            double target = (double)s->p.top_p * sum;
            double cum = 0.0;
            uint32_t keep = n_cand;
            for (uint32_t i = 0; i < n_cand; i++) {
                cum += (double)wri_exp(l[ids[i]] - mx);
                if (cum >= target) {
                    keep = i + 1;
                    break;
                }
            }
            n_cand = keep;
        }
    }

    /* Mask filter over the survivors (compacted in place, order kept). */
    if (mask_on) {
        char piece[WRI_PIECE_BUF];
        uint32_t n_allow = 0;
        uint32_t best_any = ids[0];
        float best_any_v = l[ids[0]];
        for (uint32_t i = 0; i < n_cand; i++) {
            uint32_t id = ids[i];
            if (l[id] > best_any_v) {
                best_any_v = l[id];
                best_any = id;
            }
            if (wri_mask_allows(s, id, piece, (int)sizeof(piece)))
                ids[n_allow++] = id;
        }
        if (n_allow == 0) {
            /* Everything forbidden: return the best forbidden token
             * (documented in the public header — the grammar consumer
             * stops instead of emitting it). */
            wri_hist_push(s, best_any);
            return (int32_t)best_any;
        }
        n_cand = n_allow;
    }

    /* Max-shifted softmax over the final candidate set + one draw. */
    float mx = l[ids[0]];
    for (uint32_t i = 1; i < n_cand; i++)
        if (l[ids[i]] > mx) mx = l[ids[i]];
    double sum = 0.0;
    for (uint32_t i = 0; i < n_cand; i++)
        sum += (double)wri_exp(l[ids[i]] - mx);
    double r = wri_rng_uniform(&s->rng) * sum;
    uint32_t chosen = ids[n_cand - 1];       /* rounding fallback */
    double cum = 0.0;
    for (uint32_t i = 0; i < n_cand; i++) {
        cum += (double)wri_exp(l[ids[i]] - mx);
        if (cum > r) {
            chosen = ids[i];
            break;
        }
    }

    wri_hist_push(s, chosen);
    return (int32_t)chosen;
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

wr_sample_params wr_sample_params_default(void)
{
    wr_sample_params p;
    memset(&p, 0, sizeof(p));
    p.temperature = 0.0f;        /* greedy — the parity-proven default */
    p.top_k = 0;
    p.top_p = 0.0f;
    p.repeat_penalty = 1.0f;
    p.repeat_last_n = 0;
    p.seed = 0;
    return p;
}

static wr_status wri_validate_params(wr_sample_params *p)
{
    if (p->temperature != p->temperature || p->temperature < 0.0f)
        return WR_ERR_INVAL;                 /* NaN or negative */
    if (p->top_p != p->top_p)
        return WR_ERR_INVAL;                 /* NaN (any real value is legal) */
    if (p->top_k < 0)
        return WR_ERR_INVAL;
    if (p->repeat_last_n < 0)
        return WR_ERR_INVAL;
    if (p->repeat_penalty != p->repeat_penalty || p->repeat_penalty < 0.0f)
        return WR_ERR_INVAL;
    if (p->repeat_penalty == 0.0f)
        p->repeat_penalty = 1.0f;            /* zero field = default = off */
    return WR_OK;
}

wr_status wr_sampler_create(const wr_sample_params *p, wr_sampler **out)
{
    if (!out) return WR_ERR_INVAL;
    *out = NULL;
    wr_sample_params v = p ? *p : wr_sample_params_default();
    wr_status st = wri_validate_params(&v);
    if (st != WR_OK) return st;

    wr_sampler *s = (wr_sampler *)calloc(1, sizeof(*s));
    if (!s) return WR_ERR_NOMEM;
    s->p = v;
    s->rng = v.seed ? v.seed : WRI_RNG_DEFAULT_SEED;
    *out = s;
    return WR_OK;
}

void wr_sampler_free(wr_sampler *s)
{
    if (!s) return;
    free(s->work_logits);
    free(s->work_ids);
    free(s);
}

void wr_sampler_reset(wr_sampler *s, uint64_t seed)
{
    if (!s) return;
    s->rng = seed ? seed : WRI_RNG_DEFAULT_SEED;
    s->hist_len = 0;
    s->hist_head = 0;
}

int32_t wr_sample(wr_sampler *s, const float *logits, uint32_t vocab)
{
    return wri_sample_step(s, logits, vocab);
}

void wr_sampler_set_mask(wr_sampler *s, const wr_tokenizer *tok,
                         wr_token_mask_fn allow, void *user)
{
    if (!s) return;
    if (!allow) {                            /* clear */
        s->mask_fn = NULL;
        s->mask_user = NULL;
        s->mask_tok = NULL;
        return;
    }
    if (!tok) {
        /* Contract violation (mask without a piece source); refuse the
         * installation rather than crash inside wr_sample later. */
        wri_log_msg(1, "wr_sampler_set_mask: mask callback without a tokenizer "
                   "— not installed");
        return;
    }
    s->mask_fn = allow;
    s->mask_user = user;
    s->mask_tok = tok;
}
