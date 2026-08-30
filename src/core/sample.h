/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * sample.h — sampler internals.
 *
 * NEW CODE, not a port: the origin transported sampling parameters
 * end-to-end but every live decode path was greedy argmax.  There is no
 * reference behavior to preserve, so the semantics below are fixed by
 * this contract and differential-tested against llama.cpp's sampler on
 * shared models:
 *
 *   chain:  repetition penalty → temperature → top-k → top-p → draw
 *   greedy: temperature == 0 bypasses the whole chain (argmax over the
 *           penalized logits) — the parity-proven default
 *   penalty: HF-style — for each token in the recent-history window,
 *           positive logit /= penalty, negative logit *= penalty
 *   top-k:  keep the k highest logits
 *   top-p:  keep the smallest prefix of the sorted survivors whose
 *           softmax mass reaches p (at least one token always survives)
 *   draw:   categorical over the survivors' renormalized softmax using
 *           the sampler's own xorshift64* stream — no libc rand, no
 *           global state, identical sequences for identical seeds on
 *           both platforms
 *   mask:   the callback filters candidates before the draw (greedy:
 *           argmax over allowed candidates); if nothing is allowed, the
 *           best forbidden token is returned (documented in the public
 *           header — the grammar consumer stops instead)
 */
#ifndef WR_SAMPLE_H
#define WR_SAMPLE_H

#include "core/internal.h"

/* Recent-token history for the repetition penalty. */
#define WRI_SAMPLE_HISTORY_MAX 256

struct wr_sampler {
    wr_sample_params p;          /* validated copy                        */

    uint64_t rng;                /* xorshift64* state, never 0            */

    /* Penalty ring buffer of the last min(p.repeat_last_n,
     * WRI_SAMPLE_HISTORY_MAX) sampled ids. */
    uint32_t history[WRI_SAMPLE_HISTORY_MAX];
    uint32_t hist_len;
    uint32_t hist_head;

    /* Optional token mask. */
    wr_token_mask_fn    mask_fn;
    void               *mask_user;
    const wr_tokenizer *mask_tok;  /* piece source for mask_fn            */

    /* Working buffers grown on demand to vocab size (kept across calls
     * so wr_sample does no steady-state allocation). */
    float    *work_logits;
    uint32_t *work_ids;
    uint32_t  work_cap;
};

/* xorshift64* step (seed 0 is remapped to a fixed odd constant). */
uint64_t wri_rng_next(uint64_t *state);

/* Plain argmax over logits[0..vocab) — the greedy path, also used by
 * callers that bypass wr_sampler entirely.  Ties break toward the lower
 * id (deterministic). */
int32_t wri_sample_argmax(const float *logits, uint32_t vocab);

/* Full-chain sampling step backing wr_sample (validates args, applies
 * the chain above, records the pick in the history).  Returns token id
 * or negative wr_status. */
int32_t wri_sample_step(wr_sampler *s, const float *logits, uint32_t vocab);

#endif /* WR_SAMPLE_H */
