/* SPDX-License-Identifier: Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * sdk_gen.c — SDK-level test harness, written against ONLY the public
 * API (<wayruntime/wayruntime.h>) and linked against libwayruntime.a.
 * Compiled on demand by test/integration.sh and test/realtest_diff.sh;
 * it is not part of the library build.
 *
 * Modes:
 *   sdk_gen sdk   <model.gguf>
 *       Public-contract tests on a real tiny model: session KV
 *       continuation accounting, wr_batch_step bit-exactness versus
 *       serial wr_step chains, honest WR_ERR_CTX_FULL, logits
 *       borrowed-view isolation, sampler seeded determinism, and the
 *       token-mask callback.  TAP-ish "ok"/"FAIL" lines, summary
 *       "sdk_gen: P passed, F failed", exit 0 iff F == 0.
 *
 *   sdk_gen stats
 *       Sampler statistical sanity on synthetic logits (no model):
 *       temperature draw distribution, top-k=1 / tiny top-p / temp 0
 *       argmax degeneration.  Same output discipline as `sdk`.
 *
 *   sdk_gen tok   <model.gguf>
 *       Reads UTF-8 lines from stdin, writes one line of
 *       space-separated token ids per input line (empty line for an
 *       empty tokenization).  Used by the llama.cpp tokenizer diff.
 *
 *   sdk_gen tf    <model.gguf> <prompt-ids-csv> <teacher-ids-csv>
 *       Teacher-forced argmax agreement: prefills the prompt ids,
 *       then at each step compares this runtime's argmax against the
 *       next reference id and feeds the REFERENCE id (teacher
 *       forcing).  Prints "tf: agree=K/N".  Exit 0 on completion
 *       (thresholding is the calling script's job), 3 on runtime
 *       error.
 */
#include <wayruntime/wayruntime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* TAP-ish result accounting                                          */
/* ------------------------------------------------------------------ */

static int g_pass, g_fail;

static void check(const char *name, int cond)
{
    if (cond) {
        g_pass++;
        printf("ok - %s\n", name);
    } else {
        g_fail++;
        printf("FAIL - %s\n", name);
    }
}

static int finish(const char *mode)
{
    printf("sdk_gen %s: %d passed, %d failed\n", mode, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

static uint32_t argmax_f32(const float *v, uint32_t n)
{
    uint32_t best = 0;
    float    bv   = v[0];
    for (uint32_t i = 1; i < n; i++)
        if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

/* ------------------------------------------------------------------ */
/* Mask callbacks                                                     */
/* ------------------------------------------------------------------ */

static int mask_forbid_one(uint32_t token_id, const char *piece, void *user)
{
    (void)piece;
    return token_id != *(const uint32_t *)user;
}

static int mask_forbid_all(uint32_t token_id, const char *piece, void *user)
{
    (void)token_id;
    (void)piece;
    (void)user;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Shared model bring-up for the model-backed modes                   */
/* ------------------------------------------------------------------ */

typedef struct {
    wr_engine    *engine;
    wr_model     *model;
    wr_tokenizer *tok;
    wr_model_info info;
} rig;

static int rig_up(rig *r, const char *path)
{
    wr_status st;

    memset(r, 0, sizeof *r);
    st = wr_engine_create(NULL, &r->engine);
    if (st != WR_OK) {
        fprintf(stderr, "sdk_gen: engine create failed: %s\n",
                wr_status_str(st));
        return -1;
    }
    st = wr_model_load(r->engine, path, NULL, &r->model);
    if (st != WR_OK) {
        fprintf(stderr, "sdk_gen: model load failed: %s\n",
                wr_status_str(st));
        return -1;
    }
    st = wr_model_get_info(r->model, &r->info);
    if (st != WR_OK) {
        fprintf(stderr, "sdk_gen: model info failed: %s\n",
                wr_status_str(st));
        return -1;
    }
    st = wr_tokenizer_from_model(r->model, &r->tok);
    if (st != WR_OK) {
        fprintf(stderr, "sdk_gen: tokenizer build failed: %s\n",
                wr_status_str(st));
        return -1;
    }
    return 0;
}

static void rig_down(rig *r)
{
    if (r->tok)
        wr_tokenizer_free(r->tok);
    if (r->model)
        wr_model_free(r->model);
    if (r->engine)
        wr_engine_destroy(r->engine);
}

/* ------------------------------------------------------------------ */
/* sdk mode                                                           */
/* ------------------------------------------------------------------ */

/* Session continuation: two wr_generate calls on ONE session must
 * share the KV cache.  Position accounting is exact: a generate call
 * prefills n_prompt-1 ids, then performs one wr_step per generated
 * token, so pos advances by tokens_in + tokens_out - 1 for both the
 * MAX_TOKENS and the EOS stop (the final sampled token is never fed). */
static void t_session_continuation(rig *r)
{
    wr_session *s = NULL;
    wr_status   st = wr_session_create(r->model, NULL, &s);

    check("continuation: session create", st == WR_OK);
    if (st != WR_OK)
        return;

    wr_generate_params gp = {0};
    wr_generate_result r1, r2;

    gp.prompt     = "Once upon a time";
    gp.max_tokens = 4;
    gp.flags      = WR_GEN_RAW;      /* verbatim; sampler NULL = greedy */
    gp.stop_token = -1;

    st = wr_generate(s, &gp, &r1);
    check("continuation: first generate ok", st == WR_OK);
    if (st != WR_OK) {
        wr_session_destroy(s);
        return;
    }
    check("continuation: first call produced text", r1.text != NULL);
    check("continuation: first call generated tokens", r1.tokens_out >= 1);
    uint32_t pos1 = wr_session_pos(s);
    check("continuation: pos == in + out - 1 after first call",
          pos1 == r1.tokens_in + r1.tokens_out - 1);

    gp.prompt = " and the little dog";
    st = wr_generate(s, &gp, &r2);
    check("continuation: second generate ok (KV retained)", st == WR_OK);
    if (st == WR_OK) {
        uint32_t pos2 = wr_session_pos(s);
        check("continuation: second call advanced the shared KV",
              pos2 > pos1);
        check("continuation: pos delta == in2 + out2 - 1",
              pos2 - pos1 == r2.tokens_in + r2.tokens_out - 1);
        wr_free(r2.text);
    }
    wr_free(r1.text);
    wr_session_destroy(s);
}

/* wr_batch_step over 2 sessions must be bit-exact versus 2 serial
 * wr_step chains (documented contract: the batched LM-head GEMM demux
 * equals N serial calls). */
static void t_batch_vs_serial(rig *r)
{
    static const char *prompts[2] = { "Once upon a time", "The little cat" };
    uint32_t ids[2][64];
    int      n[2];
    wr_session *b[2] = {0}, *ser[2] = {0};
    uint32_t vocab = r->info.vocab_size;
    int ok = 1;

    for (int i = 0; i < 2; i++) {
        n[i] = wr_tokenize(r->tok, prompts[i], ids[i], 64);
        if (n[i] < 2) {
            check("batch: prompts tokenized", 0);
            return;
        }
    }
    check("batch: prompts tokenized", 1);

    for (int i = 0; i < 2; i++) {
        if (wr_session_create(r->model, NULL, &b[i]) != WR_OK ||
            wr_session_create(r->model, NULL, &ser[i]) != WR_OK) {
            check("batch: sessions created", 0);
            goto out;
        }
    }
    check("batch: sessions created", 1);

    /* Identical prefill on the batched and the serial twin. */
    for (int i = 0; i < 2; i++) {
        if (wr_prefill(b[i], ids[i], (uint32_t)n[i] - 1) != n[i] - 1 ||
            wr_prefill(ser[i], ids[i], (uint32_t)n[i] - 1) != n[i] - 1) {
            check("batch: prefill", 0);
            goto out;
        }
    }
    check("batch: prefill", 1);

    /* 5 decode steps: first feeds the last prompt token, then the
     * (shared) greedy continuation.  Every step's logits must match
     * bit-for-bit between the batched and the serial path. */
    uint32_t feed[2] = { ids[0][n[0] - 1], ids[1][n[1] - 1] };
    for (int step = 0; step < 5 && ok; step++) {
        const float *bl[2] = {0};
        int rc = wr_batch_step(b, feed, 2, bl);
        if (rc != 2 || !bl[0] || !bl[1]) {
            ok = 0;
            break;
        }
        for (int i = 0; i < 2; i++) {
            const float *sl = wr_step(ser[i], feed[i]);
            if (!sl ||
                memcmp(bl[i], sl, (size_t)vocab * sizeof(float)) != 0) {
                ok = 0;
                break;
            }
            feed[i] = argmax_f32(sl, vocab);
        }
    }
    check("batch: 5-step wr_batch_step(2) bit-exact vs serial wr_step", ok);
    check("batch: positions advanced identically",
          wr_session_pos(b[0]) == wr_session_pos(ser[0]) &&
          wr_session_pos(b[1]) == wr_session_pos(ser[1]));

out:
    for (int i = 0; i < 2; i++) {
        if (b[i])
            wr_session_destroy(b[i]);
        if (ser[i])
            wr_session_destroy(ser[i]);
    }
}

/* Honest WR_ERR_CTX_FULL: exhausting a tiny max_context session must
 * fail loudly and consume nothing — never truncate silently. */
static void t_ctx_full(rig *r)
{
    wr_session_params sp = {0};
    wr_session *s = NULL;
    uint32_t bos = (uint32_t)(wr_token_bos(r->tok) >= 0
                              ? wr_token_bos(r->tok) : 1);
    wr_status st;

    /* Above the model bound: refused with WR_ERR_INVAL, never clamped. */
    sp.max_context = r->info.max_context + 1;
    st = wr_session_create(r->model, &sp, &s);
    check("ctx: max_context above model bound refused (WR_ERR_INVAL)",
          st == WR_ERR_INVAL && s == NULL);
    if (st == WR_OK)
        wr_session_destroy(s);

    sp.max_context = 8;
    s = NULL;
    st = wr_session_create(r->model, &sp, &s);
    check("ctx: 8-token session created", st == WR_OK);
    if (st != WR_OK)
        return;
    check("ctx: session reports its bound",
          wr_session_max_context(s) == 8);

    int steps = 0;
    for (int i = 0; i < 8; i++)
        if (wr_step(s, bos))
            steps++;
    check("ctx: 8 steps fit", steps == 8 && wr_session_pos(s) == 8);

    const float *l = wr_step(s, bos);
    check("ctx: 9th step returns NULL", l == NULL);
    check("ctx: failure cause is WR_ERR_CTX_FULL",
          wr_session_status(s) == WR_ERR_CTX_FULL);
    check("ctx: failed step consumed nothing", wr_session_pos(s) == 8);
    wr_session_destroy(s);

    /* Oversized prefill on a fresh tiny session: WR_ERR_CTX_FULL and
     * nothing consumed. */
    s = NULL;
    if (wr_session_create(r->model, &sp, &s) != WR_OK) {
        check("ctx: oversized prefill refused whole", 0);
        return;
    }
    uint32_t big[9];
    for (int i = 0; i < 9; i++)
        big[i] = bos;
    int pf = wr_prefill(s, big, 9);
    check("ctx: oversized prefill refused whole",
          pf == (int)WR_ERR_CTX_FULL && wr_session_pos(s) == 0);
    wr_session_destroy(s);
}

/* Logits borrowed-view rules: the pointer is session-owned scratch —
 * a decode on ANOTHER session must not disturb it, and wr_sample must
 * never modify the caller's logits. */
static void t_logits_stability(rig *r)
{
    wr_session *s1 = NULL, *s2 = NULL;
    uint32_t vocab = r->info.vocab_size;
    uint32_t bos = (uint32_t)(wr_token_bos(r->tok) >= 0
                              ? wr_token_bos(r->tok) : 1);
    float *copy = (float *)malloc((size_t)vocab * sizeof(float));

    if (!copy || wr_session_create(r->model, NULL, &s1) != WR_OK ||
        wr_session_create(r->model, NULL, &s2) != WR_OK) {
        check("logits: rig", 0);
        free(copy);
        if (s1)
            wr_session_destroy(s1);
        return;
    }
    check("logits: rig", 1);

    const float *l1 = wr_step(s1, bos);
    check("logits: step returns a view", l1 != NULL);
    if (l1) {
        int finite = 1;
        for (uint32_t i = 0; i < vocab; i++)
            if (l1[i] != l1[i]) {        /* NaN check */
                finite = 0;
                break;
            }
        check("logits: no NaN in the vector", finite);
        memcpy(copy, l1, (size_t)vocab * sizeof(float));

        /* Decode on a DIFFERENT session of the same model: s1's view
         * must be untouched (scratch is per-session, not shared). */
        const float *l2 = wr_step(s2, bos);
        check("logits: sibling session decode leaves the view intact",
              l2 != NULL &&
              memcmp(copy, l1, (size_t)vocab * sizeof(float)) == 0);

        /* wr_sample operates on its own copy. */
        wr_sampler *smp = NULL;
        wr_sample_params p = wr_sample_params_default();
        p.temperature = 0.7f;
        p.top_k = 40;
        p.seed = 11;
        if (wr_sampler_create(&p, &smp) == WR_OK) {
            int32_t id = wr_sample(smp, l1, vocab);
            check("logits: wr_sample returns a valid id",
                  id >= 0 && (uint32_t)id < vocab);
            check("logits: wr_sample does not modify the logits",
                  memcmp(copy, l1, (size_t)vocab * sizeof(float)) == 0);
            wr_sampler_free(smp);
        } else {
            check("logits: sampler create", 0);
        }
    }

    wr_session_destroy(s2);
    wr_session_destroy(s1);
    free(copy);
}

/* Seeded sampler determinism on synthetic logits: equal seeds replay
 * equal draw sequences; a different seed diverges.
 *
 * NOTE (theoretical flake): seeds 42 and 43 drive independent RNG
 * streams, so 32 identical draws are possible in principle; over a
 * ~50-candidate distribution the probability is astronomically small
 * (well under 2^-100).  If this ever fires, re-run once, then treat a
 * repeat as a real RNG defect. */
static void t_sampler_determinism(void)
{
    enum { VOCAB = 257, DRAWS = 32 };
    float logits[VOCAB];
    int32_t a[DRAWS], bfr[DRAWS], c[DRAWS];

    for (uint32_t i = 0; i < VOCAB; i++)
        logits[i] = (float)((i * 2654435761u) >> 24) * 0.03f;

    wr_sample_params p = wr_sample_params_default();
    p.temperature = 0.9f;
    p.top_k = 50;
    p.seed = 42;

    wr_sampler *s1 = NULL, *s2 = NULL;
    if (wr_sampler_create(&p, &s1) != WR_OK ||
        wr_sampler_create(&p, &s2) != WR_OK) {
        check("sampler: create", 0);
        if (s1)
            wr_sampler_free(s1);
        return;
    }
    check("sampler: create", 1);

    for (int i = 0; i < DRAWS; i++)
        a[i] = wr_sample(s1, logits, VOCAB);
    for (int i = 0; i < DRAWS; i++)
        bfr[i] = wr_sample(s2, logits, VOCAB);
    check("sampler: same seed, two samplers, identical sequences",
          memcmp(a, bfr, sizeof a) == 0);

    wr_sampler_reset(s1, 42);
    for (int i = 0; i < DRAWS; i++)
        bfr[i] = wr_sample(s1, logits, VOCAB);
    check("sampler: reset(42) replays the sequence",
          memcmp(a, bfr, sizeof a) == 0);

    wr_sampler_reset(s1, 43);
    for (int i = 0; i < DRAWS; i++)
        c[i] = wr_sample(s1, logits, VOCAB);
    check("sampler: seed 43 diverges from seed 42 "
          "(see flake note in source)",
          memcmp(a, c, sizeof a) != 0);

    int in_range = 1;
    for (int i = 0; i < DRAWS; i++)
        if (a[i] < 0 || a[i] >= VOCAB)
            in_range = 0;
    check("sampler: every draw is a valid id", in_range);

    wr_sampler_free(s2);
    wr_sampler_free(s1);
}

/* Token-mask callback: a forbidden token must never be produced while
 * an allowed alternative exists; when EVERYTHING is forbidden the best
 * forbidden token is returned (documented grammar-consumer contract). */
static void t_mask(rig *r)
{
    wr_session *s = NULL;
    uint32_t vocab = r->info.vocab_size;
    uint32_t bos = (uint32_t)(wr_token_bos(r->tok) >= 0
                              ? wr_token_bos(r->tok) : 1);

    if (wr_session_create(r->model, NULL, &s) != WR_OK) {
        check("mask: rig", 0);
        return;
    }
    const float *l = wr_step(s, bos);
    if (!l) {
        check("mask: rig", 0);
        wr_session_destroy(s);
        return;
    }
    check("mask: rig", 1);

    uint32_t a0 = argmax_f32(l, vocab);

    /* Expected greedy pick under the mask: best token skipping a0,
     * strict > with ties toward the lower id (mirrors the sampler's
     * documented tie rule). */
    int32_t expect = -1;
    float   ev = 0.0f;
    for (uint32_t i = 0; i < vocab; i++) {
        if (i == a0)
            continue;
        if (expect < 0 || l[i] > ev) {
            expect = (int32_t)i;
            ev = l[i];
        }
    }

    wr_sampler *greedy = NULL;
    if (wr_sampler_create(NULL, &greedy) != WR_OK) {
        check("mask: greedy sampler", 0);
        wr_session_destroy(s);
        return;
    }

    uint32_t forbidden = a0;
    wr_sampler_set_mask(greedy, r->tok, mask_forbid_one, &forbidden);
    int32_t got = wr_sample(greedy, l, vocab);
    check("mask: greedy avoids the forbidden argmax",
          got >= 0 && (uint32_t)got != a0);
    check("mask: greedy picks the best ALLOWED token", got == expect);

    wr_sampler_set_mask(greedy, r->tok, mask_forbid_all, NULL);
    got = wr_sample(greedy, l, vocab);
    check("mask: all-forbidden returns the best forbidden token",
          got == (int32_t)a0);

    /* Clear restores unmasked behavior. */
    wr_sampler_set_mask(greedy, NULL, NULL, NULL);
    got = wr_sample(greedy, l, vocab);
    check("mask: cleared mask restores plain argmax", got == (int32_t)a0);
    wr_sampler_free(greedy);

    /* Stochastic path: 200 draws, the forbidden id must never appear
     * (allowed alternatives always exist inside top-k 5). */
    wr_sample_params p = wr_sample_params_default();
    p.temperature = 1.0f;
    p.top_k = 5;
    p.seed = 7;
    wr_sampler *rnd = NULL;
    if (wr_sampler_create(&p, &rnd) == WR_OK) {
        wr_sampler_set_mask(rnd, r->tok, mask_forbid_one, &forbidden);
        int never = 1, valid = 1;
        for (int i = 0; i < 200; i++) {
            int32_t id = wr_sample(rnd, l, vocab);
            if (id < 0 || (uint32_t)id >= vocab)
                valid = 0;
            if (id == (int32_t)a0)
                never = 0;
        }
        check("mask: 200 stochastic draws never emit the forbidden id",
              never && valid);
        wr_sampler_free(rnd);
    } else {
        check("mask: stochastic sampler create", 0);
    }

    wr_session_destroy(s);
}

static int mode_sdk(const char *model_path)
{
    rig r;
    if (rig_up(&r, model_path) != 0) {
        rig_down(&r);
        return 3;
    }

    t_session_continuation(&r);
    t_batch_vs_serial(&r);
    t_ctx_full(&r);
    t_logits_stability(&r);
    t_sampler_determinism();
    t_mask(&r);

    rig_down(&r);
    return finish("sdk");
}

/* ------------------------------------------------------------------ */
/* stats mode — no model, synthetic logits                            */
/* ------------------------------------------------------------------ */

static int mode_stats(void)
{
    enum { VOCAB = 8 };
    float logits[VOCAB];

    /* Two live candidates: exp(10)/exp(9) => token 0 ~2.72x likelier
     * than token 1; the rest are numerically dead (-1e30). */
    for (uint32_t i = 0; i < VOCAB; i++)
        logits[i] = -1e30f;
    logits[0] = 10.0f;
    logits[1] = 9.0f;

    {
        wr_sample_params p = wr_sample_params_default();
        p.temperature = 1.0f;
        p.seed = 1234;
        wr_sampler *s = NULL;
        if (wr_sampler_create(&p, &s) != WR_OK) {
            check("stats: sampler create", 0);
            return finish("stats");
        }
        int c0 = 0, c1 = 0, other = 0;
        for (int i = 0; i < 10000; i++) {
            int32_t id = wr_sample(s, logits, VOCAB);
            if (id == 0)
                c0++;
            else if (id == 1)
                c1++;
            else
                other++;
        }
        wr_sampler_free(s);
        printf("# stats: temp 1.0, 10000 draws: id0=%d id1=%d other=%d\n",
               c0, c1, other);
        check("stats: dead candidates never drawn", other == 0);
        check("stats: both live tokens appear", c0 > 0 && c1 > 0);
        check("stats: argmax token drawn more often", c0 > c1);
        /* Expected c1 ~= 2689 (p = 1/(1+e)); < 100 is beyond any
         * plausible statistical fluctuation — a real defect. */
        check("stats: runner-up frequency plausible (>= 100/10000)",
              c1 >= 100);
    }

    {
        wr_sample_params p = wr_sample_params_default();
        p.temperature = 1.0f;
        p.top_k = 1;
        p.seed = 99;
        wr_sampler *s = NULL;
        int all0 = 1;
        if (wr_sampler_create(&p, &s) == WR_OK) {
            for (int i = 0; i < 1000; i++)
                if (wr_sample(s, logits, VOCAB) != 0)
                    all0 = 0;
            wr_sampler_free(s);
        } else {
            all0 = 0;
        }
        check("stats: top_k=1 always argmax", all0);
    }

    {
        wr_sample_params p = wr_sample_params_default();
        p.temperature = 1.0f;
        p.top_p = 0.01f;
        p.seed = 5;
        wr_sampler *s = NULL;
        int all0 = 1;
        if (wr_sampler_create(&p, &s) == WR_OK) {
            for (int i = 0; i < 1000; i++)
                if (wr_sample(s, logits, VOCAB) != 0)
                    all0 = 0;
            wr_sampler_free(s);
        } else {
            all0 = 0;
        }
        check("stats: tiny top_p always argmax", all0);
    }

    {
        /* Different argmax position so temp-0 is proven to track the
         * data, not slot 0; three seeds must all agree. */
        float l2[VOCAB];
        for (uint32_t i = 0; i < VOCAB; i++)
            l2[i] = (float)i * 0.25f;
        l2[3] = 100.0f;
        static const uint64_t seeds[3] = { 42, 43, 0xDEADBEEFu };
        int all3 = 1;
        for (int k = 0; k < 3; k++) {
            wr_sample_params p = wr_sample_params_default(); /* temp 0 */
            p.seed = seeds[k];
            wr_sampler *s = NULL;
            if (wr_sampler_create(&p, &s) != WR_OK) {
                all3 = 0;
                break;
            }
            for (int i = 0; i < 100; i++)
                if (wr_sample(s, l2, VOCAB) != 3)
                    all3 = 0;
            wr_sampler_free(s);
        }
        check("stats: temperature 0 always argmax regardless of seed",
              all3);
    }

    return finish("stats");
}

/* ------------------------------------------------------------------ */
/* tok mode — stdin lines to id lines                                 */
/* ------------------------------------------------------------------ */

static int mode_tok(const char *model_path)
{
    rig r;
    if (rig_up(&r, model_path) != 0) {
        rig_down(&r);
        return 3;
    }

    char line[8192];
    static uint32_t ids[16384];
    int rc = 0;

    while (fgets(line, (int)sizeof line, stdin)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        int n = wr_tokenize(r.tok, line, ids,
                            (int)(sizeof ids / sizeof ids[0]));
        if (n < 0) {
            printf("ERR %d\n", n);
            rc = 3;
            continue;
        }
        for (int i = 0; i < n; i++)
            printf(i ? " %u" : "%u", (unsigned)ids[i]);
        printf("\n");
    }
    fflush(stdout);
    rig_down(&r);
    return rc;
}

/* ------------------------------------------------------------------ */
/* tf mode — teacher-forced argmax agreement                          */
/* ------------------------------------------------------------------ */

static int parse_csv_u32(const char *s, uint32_t *out, int cap)
{
    int n = 0;
    while (*s) {
        char *end = NULL;
        unsigned long v = strtoul(s, &end, 10);
        if (end == s || n >= cap)
            return -1;
        out[n++] = (uint32_t)v;
        s = end;
        if (*s == ',')
            s++;
        else if (*s)
            return -1;
    }
    return n;
}

static int mode_tf(const char *model_path, const char *prompt_csv,
                   const char *teacher_csv)
{
    uint32_t prompt[512], teacher[512];
    int np = parse_csv_u32(prompt_csv, prompt, 512);
    int nt = parse_csv_u32(teacher_csv, teacher, 512);

    if (np < 1 || nt < 1) {
        fprintf(stderr, "sdk_gen tf: bad id list\n");
        return 2;
    }

    rig r;
    if (rig_up(&r, model_path) != 0) {
        rig_down(&r);
        return 3;
    }

    wr_session *s = NULL;
    if (wr_session_create(r.model, NULL, &s) != WR_OK) {
        fprintf(stderr, "sdk_gen tf: session create failed\n");
        rig_down(&r);
        return 3;
    }

    if (np > 1 && wr_prefill(s, prompt, (uint32_t)np - 1) != np - 1) {
        fprintf(stderr, "sdk_gen tf: prefill failed\n");
        wr_session_destroy(s);
        rig_down(&r);
        return 3;
    }

    const float *logits = wr_step(s, prompt[np - 1]);
    int agree = 0;
    for (int i = 0; i < nt; i++) {
        if (!logits) {
            fprintf(stderr, "sdk_gen tf: step failed: %s\n",
                    wr_status_str(wr_session_status(s)));
            wr_session_destroy(s);
            rig_down(&r);
            return 3;
        }
        uint32_t ours = argmax_f32(logits, r.info.vocab_size);
        int m = (ours == teacher[i]);
        agree += m;
        printf("tf step %2d  our=%6u  ref=%6u  %s\n",
               i + 1, (unsigned)ours, (unsigned)teacher[i],
               m ? "match" : "DIFF");
        if (i + 1 < nt)
            logits = wr_step(s, teacher[i]);   /* teacher forcing */
    }
    printf("tf: agree=%d/%d\n", agree, nt);

    wr_session_destroy(s);
    rig_down(&r);
    return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    if (argc >= 3 && strcmp(argv[1], "sdk") == 0)
        return mode_sdk(argv[2]);
    if (argc >= 2 && strcmp(argv[1], "stats") == 0)
        return mode_stats();
    if (argc >= 3 && strcmp(argv[1], "tok") == 0)
        return mode_tok(argv[2]);
    if (argc >= 5 && strcmp(argv[1], "tf") == 0)
        return mode_tf(argv[2], argv[3], argv[4]);

    fprintf(stderr,
            "usage: sdk_gen sdk <model.gguf>\n"
            "       sdk_gen stats\n"
            "       sdk_gen tok <model.gguf>   (lines on stdin)\n"
            "       sdk_gen tf <model.gguf> <prompt-ids-csv> "
            "<teacher-ids-csv>\n");
    return 2;
}
