/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * generate.c — one-call generation facade:
 *   tokenize → (chat template) → prefill(n-1) → step/sample loop →
 *   one-pass detokenize.
 *
 * Ported from the origin OS's serving loop, with its process machinery
 * (IPC framing, model hot-attach records, response spill files, fixed
 * static buffers) removed.  Everything here is per-call state — no
 * statics, no truncating caps on prompt or output beyond the session's
 * context.
 *
 * Load-bearing details preserved from the origin loop:
 *
 *   - PREFILL DISCIPLINE: the first n_prompt-1 tokens go through
 *     wr_prefill; the LAST prompt token seeds the first wr_step.
 *     Feeding all n tokens to prefill and the last one again to step
 *     appended it to the KV cache twice (the duplicated ChatML newline
 *     bug) and corrupted the first prediction.
 *
 *   - CHAT-TEMPLATE AUTODETECT: probes the vocab for the turn-marker
 *     pair "<|turn>"/"<turn|>" (Gemma-style, roles " user"/" model" —
 *     the leading space becomes the SPM word marker) before the ChatML
 *     pair "<|im_start|>"/"<|im_end|>" (roles "user"/"assistant").
 *     Markers are emitted as ids (they never match as plain text);
 *     role and prompt text are encoded.
 *
 *   - ChatML NEWLINE VIA ENCODE: byte-level BPE vocabs hold '\n' as
 *     the U+010A surrogate, so an exact-string "\n" lookup misses and
 *     would silently drop every structural newline — the model then
 *     sees malformed ChatML and emits junk.  The newline id is
 *     obtained by ENCODING "\n" (Gemma-style vocabs keep a literal
 *     "\n" entry, looked up directly).
 *
 *   - "/no_think" INJECTION: vocab == 151936 identifies Qwen3, whose
 *     reasoning channel otherwise eats a bounded token budget; its
 *     documented chat control is appended to the user turn unless the
 *     caller already chose one.
 *
 *   - TEMPLATE STOP WITH MARKER DROP: the turn-close id can differ
 *     from the model's EOS; generation stops on it and the marker is
 *     dropped from both the text and the token count.
 *
 *   - ONE-PASS DETOKENIZE: the whole generated id stream is decoded
 *     in a single wr_detokenize call so pieces carrying spaces or
 *     split UTF-8 sequences concatenate correctly.
 */
#include <stdlib.h>
#include <string.h>

#include "core/internal.h"
#include "core/sample.h"

/* Piece scratch bound for measurement and streaming callbacks. */
#define WRI_GEN_PIECE_BUF 4096

/* --------------------------------------------------------------------------
 * Growable id vector (per-call; replaces the origin's fixed 512/256
 * static arrays)
 * -------------------------------------------------------------------------- */

typedef struct wri_u32vec {
    uint32_t *v;
    uint32_t n;
    uint32_t cap;
} wri_u32vec;

static wr_status wri_vec_reserve(wri_u32vec *a, uint32_t need_total)
{
    if (need_total <= a->cap) return WR_OK;
    uint32_t cap = a->cap ? a->cap : 64;
    while (cap < need_total) {
        if (cap > UINT32_MAX / 2) {
            cap = need_total;
            break;
        }
        cap *= 2;
    }
    uint32_t *nv = (uint32_t *)realloc(a->v, (size_t)cap * sizeof(uint32_t));
    if (!nv) return WR_ERR_NOMEM;
    a->v = nv;
    a->cap = cap;
    return WR_OK;
}

static wr_status wri_vec_push(wri_u32vec *a, uint32_t x)
{
    if (a->n == UINT32_MAX) return WR_ERR_LIMIT;
    wr_status st = wri_vec_reserve(a, a->n + 1);
    if (st != WR_OK) return st;
    a->v[a->n++] = x;
    return WR_OK;
}

static void wri_vec_free(wri_u32vec *a)
{
    free(a->v);
    a->v = NULL;
    a->n = 0;
    a->cap = 0;
}

/* Append one special-token id (skipped when the vocab lacks it). */
static wr_status wri_emit_id(wri_u32vec *ids, int id)
{
    if (id < 0) return WR_OK;
    return wri_vec_push(ids, (uint32_t)id);
}

/* Encode `text` and append the ids. */
static wr_status wri_emit_text(const wr_tokenizer *tok, const char *text,
                               wri_u32vec *ids)
{
    int need = wr_tokenize(tok, text, NULL, 0);
    if (need < 0) return (wr_status)need;
    if (need == 0) return WR_OK;
    wr_status st = wri_vec_reserve(ids, ids->n + (uint32_t)need);
    if (st != WR_OK) return st;
    int got = wr_tokenize(tok, text, ids->v + ids->n, need);
    if (got < 0) return (wr_status)got;
    ids->n += (uint32_t)got;
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Chat-template framing
 *
 * Frame:  <bos> <open> {role_user}\n {prompt} <close>\n <open> {role_asst}\n
 *
 * Returns the framed token count (> 0), 0 when no template family is
 * recognized (caller falls back to the plain encode), or a negative
 * wr_status.  On success *stop_tok is the turn-close id to stop the
 * generated turn on.
 * -------------------------------------------------------------------------- */

static int wri_apply_chat_template(const wr_tokenizer *tok,
                                   uint32_t vocab_size, const char *prompt,
                                   wri_u32vec *ids, int32_t *stop_tok)
{
    *stop_tok = -1;

    /* Canonical turn markers first; some third-party conversions rename
     * them, so the renamed pair is probed as a fallback. */
    int g_open = wr_token_id(tok, "<start_of_turn>");
    int g_close = wr_token_id(tok, "<end_of_turn>");
    if (g_open < 0 || g_close < 0) {
        g_open = wr_token_id(tok, "<|turn>");
        g_close = wr_token_id(tok, "<turn|>");
    }
    int c_open = wr_token_id(tok, "<|im_start|>");
    int c_close = wr_token_id(tok, "<|im_end|>");

    int open_id, close_id, nl_id;
    const char *role_user, *role_asst;
    int qwen3_chatml = 0;

    if (g_open >= 0 && g_close >= 0) {
        /* Turn-marker family: exact "\n" lookup works (the vocab keeps
         * a literal newline entry); roles carry the leading space that
         * the SPM alphabet turns into its word marker. */
        open_id = g_open;
        close_id = g_close;
        nl_id = wr_token_id(tok, "\n");
        role_user = " user";
        role_asst = " model";
    } else if (c_open >= 0 && c_close >= 0) {
        /* ChatML family: the newline id MUST come from an encode (see
         * the file header — byte-level vocabs have no literal "\n"). */
        open_id = c_open;
        close_id = c_close;
        qwen3_chatml = (vocab_size == 151936u);
        uint32_t nltmp[2];
        int nlc = wr_tokenize(tok, "\n", nltmp, 2);
        nl_id = (nlc > 0) ? (int)nltmp[0] : -1;
        role_user = "user";
        role_asst = "assistant";
    } else {
        return 0;                        /* no recognized template */
    }

    /* Qwen3: request a direct answer unless the caller already made an
     * explicit thinking-mode choice anywhere in the prompt. */
    const char *body = prompt;
    char *owned = NULL;
    if (qwen3_chatml && strstr(prompt, "/no_think") == NULL &&
        strstr(prompt, "/think") == NULL) {
        static const char suffix[] = " /no_think";
        size_t plen = strlen(prompt);
        owned = (char *)malloc(plen + sizeof(suffix));
        if (!owned) return (int)WR_ERR_NOMEM;
        memcpy(owned, prompt, plen);
        memcpy(owned + plen, suffix, sizeof(suffix));   /* includes NUL */
        body = owned;
    }

    wr_status st = WR_OK;
#define WRI_CT_ID(x)  do { st = wri_emit_id(ids, (x));         \
                           if (st != WR_OK) goto done; } while (0)
#define WRI_CT_TXT(t) do { st = wri_emit_text(tok, (t), ids);  \
                           if (st != WR_OK) goto done; } while (0)
    WRI_CT_ID(wr_token_bos(tok));
    WRI_CT_ID(open_id);
    WRI_CT_TXT(role_user);
    WRI_CT_ID(nl_id);
    WRI_CT_TXT(body);
    WRI_CT_ID(close_id);
    WRI_CT_ID(nl_id);
    WRI_CT_ID(open_id);
    WRI_CT_TXT(role_asst);
    WRI_CT_ID(nl_id);
#undef WRI_CT_ID
#undef WRI_CT_TXT

done:
    free(owned);
    if (st != WR_OK) return (int)st;
    *stop_tok = close_id;
    return (int)ids->n;
}

/* --------------------------------------------------------------------------
 * Model-cached tokenizer
 * -------------------------------------------------------------------------- */

/* Get (building on first use) the tokenizer cached on the model.  The
 * build runs outside model->lock — wr_tokenizer_from_model takes that
 * lock itself for its refcount — so a concurrent first call from a
 * sibling session may race; the loser's instance is freed. */
static wr_status wri_cached_tokenizer(wr_model *m, const wr_tokenizer **out)
{
    wr_mutex_lock(m->lock);
    wr_tokenizer *tok = m->cached_tok;
    wr_mutex_unlock(m->lock);
    if (tok) {
        *out = tok;
        return WR_OK;
    }

    wr_tokenizer *fresh = NULL;
    wr_status st = wr_tokenizer_from_model(m, &fresh);
    if (st != WR_OK) return st;

    wr_tokenizer *loser = NULL;
    wr_mutex_lock(m->lock);
    if (m->cached_tok) {
        tok = m->cached_tok;
        loser = fresh;
    } else {
        m->cached_tok = fresh;
        tok = fresh;
    }
    wr_mutex_unlock(m->lock);
    if (loser) wr_tokenizer_free(loser);

    *out = tok;
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * wr_generate
 * -------------------------------------------------------------------------- */

wr_status wr_generate(wr_session *s, const wr_generate_params *p,
                      wr_generate_result *out)
{
    if (!out) return WR_ERR_INVAL;
    memset(out, 0, sizeof(*out));
    if (!s || !p || !p->prompt) return WR_ERR_INVAL;

    wr_model *m = s->model;
    if (!m) return WR_ERR_STATE;
    WRI_ASSERT(m->lock != NULL);

    const wr_tokenizer *tok = NULL;
    wr_status st = wri_cached_tokenizer(m, &tok);
    if (st != WR_OK) return st;

    /* ---- prompt framing ------------------------------------------------ */
    wri_u32vec ids = {NULL, 0, 0};
    int32_t template_stop = -1;
    int framed = 0;
    if (p->flags & WR_GEN_CHAT_TEMPLATE) {
        int r = wri_apply_chat_template(tok, m->vocab_size, p->prompt, &ids,
                                        &template_stop);
        if (r < 0) {
            wri_vec_free(&ids);
            return (wr_status)r;
        }
        framed = (r > 0);
    }
    if (!framed) {
        /* Plain path.  Default injects the model's BOS (when declared);
         * WR_GEN_RAW sends the prompt verbatim. */
        if (!(p->flags & WR_GEN_RAW)) {
            st = wri_emit_id(&ids, wr_token_bos(tok));
            if (st != WR_OK) {
                wri_vec_free(&ids);
                return st;
            }
        }
        st = wri_emit_text(tok, p->prompt, &ids);
        if (st != WR_OK) {
            wri_vec_free(&ids);
            return st;
        }
    }
    if (ids.n == 0) {
        /* Empty prompt: a lone BOS still seeds a step.  RAW forbids the
         * injection, so an empty RAW prompt has nothing to feed. */
        int bos = wr_token_bos(tok);
        if ((p->flags & WR_GEN_RAW) || bos < 0) {
            wri_vec_free(&ids);
            return WR_ERR_INVAL;
        }
        st = wri_vec_push(&ids, (uint32_t)bos);
        if (st != WR_OK) {
            wri_vec_free(&ids);
            return st;
        }
    }

    /* ---- prefill n-1, step seeds the last prompt token ----------------- */
    uint32_t n_prompt = ids.n;
    if (n_prompt > 1) {
        int r = wr_prefill(s, ids.v, n_prompt - 1);
        if (r < 0) {
            wri_vec_free(&ids);
            return (wr_status)r;
        }
        if ((uint32_t)r != n_prompt - 1) {
            wri_vec_free(&ids);
            return WR_ERR_INTERNAL;      /* prefill is all-or-nothing */
        }
    }

    /* ---- decode loop ---------------------------------------------------- */
    uint32_t vocab = m->vocab_size;
    int32_t eos = wr_token_eos(tok);
    uint32_t cur = ids.v[n_prompt - 1];
    /* Prompt tokens actually in the KV cache: the prefill batch now, plus
     * the seed token once its step succeeds. */
    uint32_t prompt_consumed = (n_prompt > 1) ? n_prompt - 1 : 0;
    /* The last sampled token not yet fed to the session (>= 0 after a
     * stop); it is appended after the loop so the retained context
     * matches the transcript the caller receives. */
    int32_t tail = -1;
    wri_u32vec text_ids = {NULL, 0, 0};  /* ids included in the text */
    uint32_t n_gen = 0;                  /* tokens generated (tokens_out) */
    int32_t stop_reason = WR_STOP_MAX_TOKENS;
    wr_status err = WR_OK;

    while (p->max_tokens == 0 || n_gen < p->max_tokens) {
        const float *logits = wr_step(s, cur);
        if (!logits) {
            wr_status ss = wr_session_status(s);
            if (ss == WR_ERR_CTX_FULL) {
                stop_reason = WR_STOP_CTX_FULL;
                break;
            }
            err = (ss == WR_OK) ? WR_ERR_INTERNAL : ss;
            break;
        }
        prompt_consumed = n_prompt;      /* the seed step consumed */
        tail = -1;                       /* cur is now in the KV cache */

        int32_t next = p->sampler ? wr_sample(p->sampler, logits, vocab)
                                  : wri_sample_argmax(logits, vocab);
        if (next < 0) {
            err = (wr_status)next;
            break;
        }

        /* Template turn-close: dropped from text AND count (its id can
         * equal the model EOS — ChatML — so it is checked first and
         * reported as the template stop). */
        if (template_stop >= 0 && next == template_stop) {
            stop_reason = WR_STOP_TEMPLATE;
            tail = next;
            break;
        }
        /* Model EOS or the caller's extra stop token: generated (so it
         * counts) but excluded from the text — stop markers never
         * bleed into the response. */
        if ((eos >= 0 && next == eos) ||
            (p->stop_token > 0 && next == p->stop_token)) {
            n_gen++;
            stop_reason = WR_STOP_EOS;
            tail = next;
            break;
        }

        st = wri_vec_push(&text_ids, (uint32_t)next);
        if (st != WR_OK) {
            err = st;
            break;
        }
        n_gen++;
        cur = (uint32_t)next;
        tail = next;

        if (p->on_token) {
            char piece[WRI_GEN_PIECE_BUF];
            int pl = wr_token_piece(tok, (uint32_t)next, piece,
                                    (int)sizeof(piece));
            if (pl < 0) piece[0] = '\0';
            if (p->on_token((uint32_t)next, piece, p->token_user) != 0) {
                stop_reason = WR_STOP_CALLBACK;
                break;
            }
        }
    }

    if (err != WR_OK) {
        wri_vec_free(&ids);
        wri_vec_free(&text_ids);
        return err;
    }

    /* ---- retain the final token ---------------------------------------- */
    /* The last sampled token (final text token, model EOS, or the
     * template close marker) has not been fed to the session yet.  Append
     * it - without the LM-head projection - so the KV cache holds exactly
     * the transcript the caller receives and a following wr_generate on
     * this session continues the conversation correctly.  A full context
     * is the one case where it cannot be retained. */
    if (tail >= 0) {
        uint32_t t = (uint32_t)tail;
        int r = wr_prefill(s, &t, 1);
        if (r < 0 && r != WR_ERR_CTX_FULL) {
            wri_vec_free(&ids);
            wri_vec_free(&text_ids);
            return (wr_status)r;
        }
    }

    /* ---- one-pass detokenize ------------------------------------------- */
    /* Exact sizing first: the decode length is the sum of the per-token
     * piece lengths (piece and decode apply the same normalization), so
     * the single wr_detokenize pass below never truncates. */
    uint64_t total = 0;
    {
        char piece[WRI_GEN_PIECE_BUF];
        for (uint32_t i = 0; i < text_ids.n; i++) {
            int pl = wr_token_piece(tok, text_ids.v[i], piece,
                                    (int)sizeof(piece));
            if (pl == WR_ERR_LIMIT) {
                /* A vocab piece longer than the measuring buffer: refuse
                 * rather than under-size the output and truncate. */
                wri_log_msg(0, "generate: token %u piece exceeds %d bytes "
                        "(WR_ERR_LIMIT)", text_ids.v[i], WRI_GEN_PIECE_BUF);
                wri_vec_free(&ids);
                wri_vec_free(&text_ids);
                return WR_ERR_LIMIT;
            }
            if (pl > 0) total += (uint64_t)pl;
        }
    }
    if (total > (uint64_t)INT32_MAX - 1) {
        wri_vec_free(&ids);
        wri_vec_free(&text_ids);
        return WR_ERR_LIMIT;
    }

    char *text = (char *)malloc((size_t)total + 1);
    if (!text) {
        wri_vec_free(&ids);
        wri_vec_free(&text_ids);
        return WR_ERR_NOMEM;
    }
    int written = 0;
    if (text_ids.n > 0) {
        written = wr_detokenize(tok, text_ids.v, (int)text_ids.n, text,
                                (int)(total + 1));
        if (written < 0) {
            free(text);
            wri_vec_free(&ids);
            wri_vec_free(&text_ids);
            return (wr_status)written;
        }
    }
    text[written] = '\0';

    out->text = text;                    /* caller releases with wr_free */
    out->tokens_in = prompt_consumed;
    out->tokens_out = n_gen;
    out->stop_reason = stop_reason;

    wri_vec_free(&ids);
    wri_vec_free(&text_ids);
    return WR_OK;
}
