/* SPDX-License-Identifier: Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * grammar_mask — constrained decoding with wr_sampler_set_mask.
 *
 * A self-contained example of the token-mask callback API: a byte-level
 * grammar automaton restricts sampling so the model can only emit a
 * tiny JSON-ish object,
 *
 *     object := ws '{' pair (ws ',' pair)* ws '}' ws
 *     pair   := ws '"' key '"' ':' ' '* '"' value '"'
 *     key    := [a-z0-9_]{1..24}
 *     value  := (printable ASCII except '"' and '\'){1..48}
 *     ws     := (' ' | '\n')*
 *
 * so the output is well-formed BY CONSTRUCTION, whatever the model
 * would rather say.  The automaton design descends from the origin OS's
 * grammar-masked plan decoder, the production-proven consumer of this
 * callback style; the grammar surface here is re-targeted to JSON-ish
 * output for SDK-example purposes.
 *
 * Mechanics worth copying into real consumers:
 *
 *   - the MASK callback (gm_allow) trial-feeds a candidate token's
 *     bytes WITHOUT committing: wr_sample consults it for many
 *     candidates per step, so it must never mutate the live state;
 *   - the STREAM callback (gm_commit) advances the automaton with the
 *     token that actually won the sample, and stops generation once the
 *     grammar reaches its accept state;
 *   - EOS is special-cased by token id: it carries no grammar bytes and
 *     is legal exactly when the automaton would accept end-of-stream;
 *   - zero-length pieces are masked out: they make no grammar progress,
 *     so allowing them would let greedy decoding spin forever;
 *   - if the mask forbids every candidate, wr_sample still returns the
 *     best forbidden token (documented contract) — the consumer detects
 *     this when its commit fails, and stops;
 *   - piece strings arrive as REAL bytes: the byte-level alphabet's
 *     space/newline surrogates are already normalized by the library.
 *
 * Build (library already built, from the repository root):
 *   gcc -std=c11 -O2 -Iinclude examples/grammar_mask.c \
 *       build/posix/libwayruntime.a -lpthread -lm -o grammar_mask
 * Run:
 *   ./grammar_mask path/to/model.gguf ["prompt override"]
 */

#include <wayruntime/wayruntime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GM_KEY_MAX 24 /* max chars in a key   */
#define GM_VAL_MAX 48 /* max chars in a value */

/* Automaton states. */
enum {
    GM_S_OBJ_OPEN = 0, /* expecting '{' (leading ws ok)                  */
    GM_S_PAIR_OPEN,    /* expecting the '"' that opens a key (ws ok)     */
    GM_S_KEY,          /* inside the key                                 */
    GM_S_COLON,        /* expecting ':'                                  */
    GM_S_VAL_OPEN,     /* expecting the '"' that opens a value (' ' ok)  */
    GM_S_VAL,          /* inside the value                               */
    GM_S_NEXT,         /* expecting ',' or '}' (ws ok)                   */
    GM_S_DONE          /* object closed; only trailing ws is legal       */
};

typedef struct {
    uint8_t  state;
    uint16_t len;       /* chars in the current key/value            */
    uint16_t pairs;     /* completed "key": "value" pairs             */
    uint16_t min_pairs; /* pairs required before '}' / EOS are legal  */
    int32_t  eos_id;    /* tokenizer EOS id, -1 if not declared       */
} gm_state_t;

static void gm_init(gm_state_t *g, int32_t eos_id)
{
    g->state     = GM_S_OBJ_OPEN;
    g->len       = 0;
    g->pairs     = 0;
    g->min_pairs = 1;
    g->eos_id    = eos_id;
}

/* Demand a minimum amount of structure before the object may close. */
static void gm_require_pairs(gm_state_t *g, uint16_t min_pairs)
{
    g->min_pairs = min_pairs > 0 ? min_pairs : 1;
}

static int gm_key_char(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

/* Feed one byte.  Returns 1 if the byte keeps the object well-formed,
 * 0 if it would break the grammar (the caller masks the token).  On
 * rejection the state is left unchanged — trial copies just discard. */
static int gm_feed(gm_state_t *g, char c)
{
    switch (g->state) {
    case GM_S_OBJ_OPEN:
        if (c == ' ' || c == '\n')
            return 1;
        if (c == '{') {
            g->state = GM_S_PAIR_OPEN;
            return 1;
        }
        return 0;
    case GM_S_PAIR_OPEN:
        if (c == ' ' || c == '\n')
            return 1;
        if (c == '"') {
            g->state = GM_S_KEY;
            g->len   = 0;
            return 1;
        }
        return 0;
    case GM_S_KEY:
        if (c == '"' && g->len > 0) {
            g->state = GM_S_COLON;
            return 1;
        }
        if (gm_key_char(c) && g->len < GM_KEY_MAX) {
            g->len++;
            return 1;
        }
        return 0;
    case GM_S_COLON:
        if (c == ':') {
            g->state = GM_S_VAL_OPEN;
            return 1;
        }
        return 0;
    case GM_S_VAL_OPEN:
        if (c == ' ')
            return 1;
        if (c == '"') {
            g->state = GM_S_VAL;
            g->len   = 0;
            return 1;
        }
        return 0;
    case GM_S_VAL:
        if (c == '"' && g->len > 0) {
            g->state = GM_S_NEXT;
            g->pairs++;
            return 1;
        }
        /* Printable ASCII only; '"' would close, '\' would need escape
         * handling this tiny grammar deliberately leaves out. */
        if (c >= 32 && c <= 126 && c != '"' && c != '\\' &&
            g->len < GM_VAL_MAX) {
            g->len++;
            return 1;
        }
        return 0;
    case GM_S_NEXT:
        if (c == ' ' || c == '\n')
            return 1;
        if (c == ',') {
            g->state = GM_S_PAIR_OPEN;
            return 1;
        }
        if (c == '}' && g->pairs >= g->min_pairs) {
            g->state = GM_S_DONE;
            return 1;
        }
        return 0;
    default: /* GM_S_DONE */
        return c == ' ' || c == '\n';
    }
}

/* Trial-feed a token's bytes WITHOUT committing: returns 1 when every
 * byte is legal (the advanced state is stored into *out when non-NULL,
 * which is how the winner is committed), 0 when any byte is illegal. */
static int gm_token_ok(const gm_state_t *g, const char *bytes, size_t n,
                       gm_state_t *out)
{
    gm_state_t t = *g;
    for (size_t i = 0; i < n; i++)
        if (!gm_feed(&t, bytes[i]))
            return 0;
    if (out)
        *out = t;
    return 1;
}

/* End-of-stream acceptability: the object must be closed. */
static int gm_eos_ok(const gm_state_t *g)
{
    return g->state == GM_S_DONE && g->pairs >= g->min_pairs;
}

/* wr_token_mask_fn: consulted per CANDIDATE token during wr_sample.
 * Read-only against the live automaton state. */
static int gm_allow(uint32_t token_id, const char *piece, void *user)
{
    const gm_state_t *g = user;
    size_t            n;

    if (g->eos_id >= 0 && (int32_t)token_id == g->eos_id)
        return gm_eos_ok(g);
    n = strlen(piece);
    if (n == 0)
        return 0; /* no grammar progress — would loop forever */
    return gm_token_ok(g, piece, n, NULL);
}

/* wr_token_cb: called once for the token that actually won.  Commits
 * its bytes into the automaton, streams it, and stops generation when
 * the object is complete. */
static int gm_commit(uint32_t token_id, const char *piece, void *user)
{
    gm_state_t *g = user;

    if (g->eos_id >= 0 && (int32_t)token_id == g->eos_id)
        return 0; /* generation is about to stop on EOS anyway */
    if (!gm_token_ok(g, piece, strlen(piece), g))
        return 1; /* every candidate was forbidden and wr_sample fell
                   * back to the best forbidden token — stop, per the
                   * mask contract */
    fputs(piece, stdout);
    fflush(stdout);
    return g->state == GM_S_DONE; /* accept state: object complete */
}

int main(int argc, char **argv)
{
    const char *model_path;
    const char *prompt;

    wr_engine    *engine  = NULL;
    wr_model     *model   = NULL;
    wr_tokenizer *tok     = NULL;
    wr_session   *session = NULL;
    wr_sampler   *sampler = NULL;
    gm_state_t    g;
    wr_status     st;
    int           rc = 3;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: grammar_mask <model.gguf> [prompt]\n");
        return 1;
    }
    model_path = argv[1];
    prompt     = (argc == 3)
        ? argv[2]
        : "Here is a small JSON object with string values describing "
          "a cat:\n";

    st = wr_engine_create(NULL, &engine);
    if (st != WR_OK) {
        fprintf(stderr, "grammar_mask: engine create failed: %s\n",
                wr_status_str(st));
        return 3;
    }
    st = wr_model_load(engine, model_path, NULL, &model);
    if (st != WR_OK) {
        fprintf(stderr, "grammar_mask: loading %s failed: %s\n",
                model_path, wr_status_str(st));
        rc = 2;
        goto out;
    }
    st = wr_tokenizer_from_model(model, &tok);
    if (st != WR_OK) {
        fprintf(stderr, "grammar_mask: tokenizer build failed: %s\n",
                wr_status_str(st));
        rc = 2;
        goto out;
    }
    st = wr_session_create(model, NULL, &session);
    if (st != WR_OK) {
        fprintf(stderr, "grammar_mask: session create failed: %s\n",
                wr_status_str(st));
        goto out;
    }

    /* Greedy sampling (deterministic) with the grammar mask installed.
     * The tokenizer handed to wr_sampler_set_mask supplies the piece
     * strings the callbacks see; it must outlive the installation. */
    {
        wr_sample_params sp = wr_sample_params_default();
        st = wr_sampler_create(&sp, &sampler);
        if (st != WR_OK) {
            fprintf(stderr, "grammar_mask: sampler create failed: %s\n",
                    wr_status_str(st));
            goto out;
        }
    }
    gm_init(&g, wr_token_eos(tok));
    gm_require_pairs(&g, 2); /* insist on at least two pairs */
    wr_sampler_set_mask(sampler, tok, gm_allow, &g);

    {
        wr_generate_params gp  = {0};
        wr_generate_result res;

        gp.prompt     = prompt;
        gp.max_tokens = 96;
        gp.sampler    = sampler;
        gp.stop_token = -1;
        gp.on_token   = gm_commit; /* streams + commits + stops on DONE */
        gp.token_user = &g;

        st = wr_generate(session, &gp, &res);
        if (st != WR_OK) {
            fprintf(stderr, "grammar_mask: generate failed: %s\n",
                    wr_status_str(st));
            goto out;
        }
        fputc('\n', stdout);
        fprintf(stderr,
                "grammar_mask: %u prompt tokens, %u generated, stop=%d, "
                "%u pairs, grammar %s\n",
                res.tokens_in, res.tokens_out, (int)res.stop_reason,
                (unsigned)g.pairs,
                gm_eos_ok(&g) ? "complete" : "INCOMPLETE");
        wr_free(res.text);

        /* An incomplete object (for example max_tokens ran out mid
         * value) is reported honestly as a failure. */
        rc = gm_eos_ok(&g) ? 0 : 3;
    }

out:
    if (sampler)
        wr_sampler_free(sampler);
    if (session)
        wr_session_destroy(session);
    if (tok)
        wr_tokenizer_free(tok);
    if (model)
        wr_model_free(model);
    wr_engine_destroy(engine);
    return rc;
}
