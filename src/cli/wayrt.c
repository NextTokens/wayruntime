/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * wayrt — the wayruntime command-line tool.
 *
 * Deliberately written against ONLY the public SDK surface
 * (<wayruntime/wayruntime.h>), never the internal headers: this binary
 * is the first-party consumer of the API and doubles as its dogfooding
 * gate.  If a workflow below cannot be expressed through the public
 * header, the public header is missing something — fix the header, do
 * not reach into src/core.
 *
 * Subcommands:
 *   verify   <model.gguf>                    load, print metadata +
 *                                            tokenizer sanity, unload
 *   generate ... --prompt TEXT <model.gguf>  one-shot generation
 *   bench    [--tokens N] <model.gguf>       greedy decode timing +
 *                                            counter dump
 *   chat     <model.gguf>                    interactive chat with the
 *                                            auto-detected template
 *
 * Output discipline: `generate` writes GENERATED TEXT and nothing else
 * to stdout (scripts parse it); all diagnostics go to stderr.  `verify`
 * and `bench` write their reports to stdout.
 *
 * Exit codes: 0 ok, 1 usage, 2 model load error, 3 runtime error.
 *
 * test/realtest.sh depends on `verify <gguf>`, `generate --greedy
 * --max-tokens N --prompt S <gguf>` and `bench --tokens N <gguf>`;
 * those flags are stable interface.
 */

#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L /* clock_gettime under -std=c11 */
#endif

#include <wayruntime/wayruntime.h>

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <time.h>
#endif

/* ------------------------------------------------------------------ */
/* Exit codes (stable interface, documented in usage())               */
/* ------------------------------------------------------------------ */

enum {
    WRT_EXIT_OK      = 0,
    WRT_EXIT_USAGE   = 1,
    WRT_EXIT_LOAD    = 2,
    WRT_EXIT_RUNTIME = 3
};

/* ------------------------------------------------------------------ */
/* Global options                                                     */
/* ------------------------------------------------------------------ */

enum {
    SIMD_AUTO = 0,
    SIMD_SCALAR,
    SIMD_AVX2,
    SIMD_AVX512
};

typedef struct {
    uint32_t threads; /* 0 = library default (one per online CPU) */
    int      simd;    /* SIMD_* request from --simd                */
} cli_opts;

static void usage(FILE *out)
{
    fputs(
"usage: wayrt [--threads N] [--simd auto|scalar|avx2|avx512] <command> ...\n"
"\n"
"commands:\n"
"  verify <model.gguf>\n"
"      Load the model, print its metadata and tokenizer sanity checks,\n"
"      then unload cleanly.\n"
"  generate [--greedy | --temp T] [--top-k K] [--top-p P] [--seed N]\n"
"           [--max-tokens N] [--raw] --prompt TEXT <model.gguf>\n"
"      Generate a completion.  Only generated text is written to stdout;\n"
"      diagnostics go to stderr.  Sampling defaults to greedy;\n"
"      --max-tokens defaults to 128 (0 = until EOS or context).\n"
"      --raw feeds the prompt verbatim (no BOS injection).\n"
"  bench [--tokens N] <model.gguf>\n"
"      Time a greedy decode: prefill and decode tokens/s plus the engine\n"
"      counter dump.  N defaults to 32.\n"
"  chat <model.gguf>\n"
"      Interactive chat using the model's auto-detected chat template,\n"
"      streaming tokens as they are generated.  EOF (Ctrl-D / Ctrl-Z)\n"
"      ends the chat.\n"
"\n"
"global flags (accepted before or after the command name):\n"
"  --threads N   worker threads (0 = one per online CPU)\n"
"  --simd MODE   kernel selection; an explicit avx2/avx512 request the\n"
"                host cannot bind is an error, never a silent fallback\n"
"\n"
"exit codes: 0 ok, 1 usage, 2 model load error, 3 runtime error\n",
        out);
}

/* ------------------------------------------------------------------ */
/* Small helpers                                                      */
/* ------------------------------------------------------------------ */

static void fail_status(const char *what, wr_status st)
{
    fprintf(stderr, "wayrt: %s failed: %s (%d)\n",
            what, wr_status_str(st), (int)st);
}

static const char *simd_name(int v)
{
    switch (v) {
    case WR_SIMD_SCALAR: return "scalar";
    case WR_SIMD_AVX2:   return "avx2";
    case WR_SIMD_AVX512: return "avx512";
    case WR_SIMD_NEON:   return "neon";
    default:             return "?";
    }
}

static const char *stop_reason_name(int32_t r)
{
    switch (r) {
    case WR_STOP_EOS:        return "eos";
    case WR_STOP_TEMPLATE:   return "template";
    case WR_STOP_MAX_TOKENS: return "max_tokens";
    case WR_STOP_CTX_FULL:   return "ctx_full";
    case WR_STOP_CALLBACK:   return "callback";
    default:                 return "?";
    }
}

/* Monotonic-ish wall clock for bench timing.  The CLI cannot use the
 * library's internal platform layer (public API only), so it carries
 * its own two-line clock. */
static double now_sec(void)
{
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static double rate_tok_s(uint32_t n, double seconds)
{
    return seconds > 0.0 ? (double)n / seconds : 0.0;
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
/* Argument parsing                                                   */
/* ------------------------------------------------------------------ */

static int parse_u32(const char *flag, const char *s, uint32_t *out)
{
    char              *end = NULL;
    unsigned long long v;
    if (!s || !*s)
        goto bad;
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno != 0 || *end != '\0' || v > 0xFFFFFFFFull)
        goto bad;
    *out = (uint32_t)v;
    return 0;
bad:
    fprintf(stderr, "wayrt: %s expects a non-negative integer, got \"%s\"\n",
            flag, s ? s : "");
    return -1;
}

static int parse_u64(const char *flag, const char *s, uint64_t *out)
{
    char              *end = NULL;
    unsigned long long v;
    if (!s || !*s)
        goto bad;
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno != 0 || *end != '\0')
        goto bad;
    *out = (uint64_t)v;
    return 0;
bad:
    fprintf(stderr, "wayrt: %s expects a non-negative integer, got \"%s\"\n",
            flag, s ? s : "");
    return -1;
}

static int parse_f32(const char *flag, const char *s, float *out)
{
    char *end = NULL;
    float v;
    if (!s || !*s)
        goto bad;
    errno = 0;
    v = strtof(s, &end);
    if (errno != 0 || *end != '\0' || v != v) /* v != v: reject NaN */
        goto bad;
    *out = v;
    return 0;
bad:
    fprintf(stderr, "wayrt: %s expects a number, got \"%s\"\n",
            flag, s ? s : "");
    return -1;
}

/* Returns the value following argv[*i] (advancing *i), or NULL with a
 * message when the command line ends first. */
static const char *flag_value(int argc, char **argv, int *i)
{
    if (*i + 1 >= argc) {
        fprintf(stderr, "wayrt: %s requires a value\n", argv[*i]);
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

/* Try argv[*i] as one of the global flags.
 * Returns 1 = consumed, 0 = not a global flag, -1 = malformed. */
static int match_global(cli_opts *o, int argc, char **argv, int *i)
{
    if (strcmp(argv[*i], "--threads") == 0) {
        const char *v = flag_value(argc, argv, i);
        if (!v || parse_u32("--threads", v, &o->threads))
            return -1;
        return 1;
    }
    if (strcmp(argv[*i], "--simd") == 0) {
        const char *v = flag_value(argc, argv, i);
        if (!v)
            return -1;
        if (strcmp(v, "auto") == 0)
            o->simd = SIMD_AUTO;
        else if (strcmp(v, "scalar") == 0)
            o->simd = SIMD_SCALAR;
        else if (strcmp(v, "avx2") == 0)
            o->simd = SIMD_AVX2;
        else if (strcmp(v, "avx512") == 0)
            o->simd = SIMD_AVX512;
        else {
            fprintf(stderr,
                    "wayrt: --simd expects auto|scalar|avx2|avx512, "
                    "got \"%s\"\n", v);
            return -1;
        }
        return 1;
    }
    return 0;
}

/* Shared parser for the commands that take only global flags plus a
 * model path (verify, chat).  Returns 0 ok, -1 usage error. */
static int parse_model_only(cli_opts *opt, int argc, char **argv,
                            const char *cmd, const char **path)
{
    *path = NULL;
    for (int i = 0; i < argc; i++) {
        int used = match_global(opt, argc, argv, &i);
        if (used < 0)
            return -1;
        if (used == 1)
            continue;
        if (argv[i][0] == '-') {
            fprintf(stderr, "wayrt: unknown %s option \"%s\"\n", cmd, argv[i]);
            return -1;
        }
        if (*path) {
            fprintf(stderr, "wayrt: unexpected extra argument \"%s\"\n",
                    argv[i]);
            return -1;
        }
        *path = argv[i];
    }
    if (!*path) {
        fprintf(stderr, "wayrt: %s requires a model path "
                        "(see wayrt --help)\n", cmd);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Engine construction from the global flags                          */
/* ------------------------------------------------------------------ */

static wr_engine *make_engine(const cli_opts *o)
{
    wr_engine_config cfg = {0};
    wr_engine       *e   = NULL;
    wr_status        st;
    int              bound;

    cfg.n_threads = o->threads;
    if (o->simd == SIMD_SCALAR)
        cfg.force_scalar = 1;
    if (o->simd == SIMD_AVX2)
        cfg.prefer_avx2 = 1;

    st = wr_engine_create(&cfg, &e);
    if (st != WR_OK) {
        fail_status("engine create", st);
        return NULL;
    }

    /* An explicit --simd avx2/avx512 request must actually bind; no
     * silent fallback to a narrower variant. */
    bound = wr_engine_simd_variant(e);
    if ((o->simd == SIMD_AVX2 && bound != WR_SIMD_AVX2) ||
        (o->simd == SIMD_AVX512 && bound != WR_SIMD_AVX512)) {
        fprintf(stderr,
                "wayrt: --simd %s requested but this host bound %s kernels\n",
                o->simd == SIMD_AVX2 ? "avx2" : "avx512", simd_name(bound));
        wr_engine_destroy(e);
        return NULL;
    }
    return e;
}

/* ------------------------------------------------------------------ */
/* Counters dump (bench)                                              */
/* ------------------------------------------------------------------ */

static const char *const k_counter_names[WR_COUNTER_COUNT] = {
    "matmul_calls",
    "matmul_elems",
    "softmax_calls",
    "attention_calls",
    "fused_calls",
    "matmul_variant",
    "softmax_variant",
    "matmul_ggml_simd",
    "matmul_ggml_variant",
    "matmul_ggml_quant",
    "matmul_perelem",
    "matmul_par_n",
};

static void print_counters(const wr_engine *e)
{
    uint64_t c[WR_COUNTER_COUNT];
    if (wr_engine_counters(e, c) != WR_OK) {
        fprintf(stderr, "wayrt: counters snapshot failed\n");
        return;
    }
    printf("counters:\n");
    for (int i = 0; i < WR_COUNTER_COUNT; i++) {
        if (i == WR_CTR_MATMUL_VARIANT || i == WR_CTR_SOFTMAX_VARIANT ||
            i == WR_CTR_MATMUL_GGML_VARIANT)
            printf("  %-22s %" PRIu64 " (%s)\n",
                   k_counter_names[i], c[i], simd_name((int)c[i]));
        else
            printf("  %-22s %" PRIu64 "\n", k_counter_names[i], c[i]);
    }
}

/* ------------------------------------------------------------------ */
/* verify                                                             */
/* ------------------------------------------------------------------ */

static int cmd_verify(cli_opts *opt, int argc, char **argv)
{
    const char   *path = NULL;
    wr_engine    *engine;
    wr_model     *model = NULL;
    wr_tokenizer *tok   = NULL;
    wr_model_info info;
    wr_status     st;
    int           rc = WRT_EXIT_RUNTIME;

    if (parse_model_only(opt, argc, argv, "verify", &path))
        return WRT_EXIT_USAGE;

    engine = make_engine(opt);
    if (!engine)
        return WRT_EXIT_RUNTIME;

    st = wr_model_load(engine, path, NULL, &model);
    if (st != WR_OK) {
        fail_status("model load", st);
        rc = WRT_EXIT_LOAD;
        goto out;
    }
    st = wr_model_get_info(model, &info);
    if (st != WR_OK) {
        fail_status("model info", st);
        goto out;
    }

    printf("file:            %s\n", path);
    printf("runtime:         wayruntime %s\n", wr_version());
    printf("arch:            %s\n", info.arch);
    printf("quant:           %s\n", info.quant);
    printf("layers:          %u\n", info.n_layers);
    printf("q_heads:         %u\n", info.n_q_heads);
    printf("kv_heads:        %u\n", info.n_kv_heads);
    printf("head_dim:        %u\n", info.head_dim);
    if (info.head_dim_swa)
        printf("head_dim_swa:    %u\n", info.head_dim_swa);
    printf("hidden_dim:      %u\n", info.hidden_dim);
    printf("ffn_dim:         %u\n", info.ffn_dim);
    printf("vocab:           %u\n", info.vocab_size);
    printf("train_context:   %u\n", info.train_context);
    printf("max_context:     %u\n", info.max_context);
    printf("rope_freq_base:  %.1f\n", (double)info.rope_freq_base);
    printf("rms_eps:         %g\n", (double)info.rms_eps);
    printf("logit_softcap:   %g\n", (double)info.logit_softcap);
    printf("sliding_window:  %u\n", info.sliding_window);
    printf("gemma_class:     %s\n", info.is_gemma ? "yes" : "no");
    printf("qk_norm:         %s\n", info.has_qk_norm ? "yes" : "no");
    printf("tied_lm_head:    %s\n", info.tied_lm_head ? "yes" : "no");
    printf("swa_pattern:     %s\n", info.has_swa ? "yes" : "no");
    printf("simd:            %s\n",
           simd_name(wr_engine_simd_variant(engine)));

    /* Tokenizer sanity: build it from the model, round-trip a plain
     * ASCII string.  Failure to build is a load-class error; failure
     * to encode/decode ASCII is a runtime error. */
    st = wr_tokenizer_from_model(model, &tok);
    if (st != WR_OK) {
        fail_status("tokenizer build", st);
        rc = WRT_EXIT_LOAD;
        goto out;
    }
    {
        static const char SANITY[] = "Hello, world!";
        uint32_t          ids[64];
        char              back[256];
        const char       *cmp;
        int               n_ids, n_bytes;

        n_ids = wr_tokenize(tok, SANITY, ids, 64);
        if (n_ids <= 0) {
            fprintf(stderr, "wayrt: tokenizer sanity encode failed: %s (%d)\n",
                    n_ids < 0 ? wr_status_str((wr_status)n_ids) : "0 tokens",
                    n_ids);
            goto out;
        }
        n_bytes = wr_detokenize(tok, ids, n_ids, back, (int)sizeof back);
        if (n_bytes < 0) {
            fail_status("tokenizer sanity decode", (wr_status)n_bytes);
            goto out;
        }
        printf("tokenizer:       vocab=%u bos=%d eos=%d\n",
               info.vocab_size, (int)wr_token_bos(tok), (int)wr_token_eos(tok));
        printf("tokenize:        \"%s\" -> %d ids -> \"%s\"\n",
               SANITY, n_ids, back);
        /* SentencePiece-style vocabularies legitimately re-add a
         * leading space marker; skip it before comparing. */
        cmp = (back[0] == ' ' && SANITY[0] != ' ') ? back + 1 : back;
        printf("roundtrip:       %s\n",
               strcmp(cmp, SANITY) == 0 ? "exact" : "differs (see above)");
    }

    rc = WRT_EXIT_OK;
out:
    if (tok)
        wr_tokenizer_free(tok);
    if (model)
        wr_model_free(model);
    wr_engine_destroy(engine);
    if (rc == WRT_EXIT_OK)
        printf("unload:          clean\n");
    return rc;
}

/* ------------------------------------------------------------------ */
/* generate                                                           */
/* ------------------------------------------------------------------ */

static int cmd_generate(cli_opts *opt, int argc, char **argv)
{
    const char *path = NULL, *prompt = NULL;
    uint32_t    max_tokens = 128;
    uint32_t    top_k = 0;
    uint64_t    seed = 0;
    float       temperature = 0.0f, top_p = 0.0f;
    int         greedy = 0, raw = 0;
    int         has_temp = 0, has_topk = 0, has_topp = 0, has_seed = 0;

    wr_engine  *engine;
    wr_model   *model   = NULL;
    wr_session *session = NULL;
    wr_sampler *sampler = NULL;
    wr_status   st;
    int         rc = WRT_EXIT_RUNTIME;

    for (int i = 0; i < argc; i++) {
        int used = match_global(opt, argc, argv, &i);
        if (used < 0)
            return WRT_EXIT_USAGE;
        if (used == 1)
            continue;
        if (strcmp(argv[i], "--greedy") == 0) {
            greedy = 1;
        } else if (strcmp(argv[i], "--raw") == 0) {
            raw = 1;
        } else if (strcmp(argv[i], "--temp") == 0) {
            const char *v = flag_value(argc, argv, &i);
            if (!v || parse_f32("--temp", v, &temperature))
                return WRT_EXIT_USAGE;
            if (temperature < 0.0f) {
                fprintf(stderr, "wayrt: --temp must be >= 0\n");
                return WRT_EXIT_USAGE;
            }
            has_temp = 1;
        } else if (strcmp(argv[i], "--top-k") == 0) {
            const char *v = flag_value(argc, argv, &i);
            if (!v || parse_u32("--top-k", v, &top_k))
                return WRT_EXIT_USAGE;
            if (top_k > 0x7FFFFFFFu) {
                fprintf(stderr, "wayrt: --top-k out of range\n");
                return WRT_EXIT_USAGE;
            }
            has_topk = 1;
        } else if (strcmp(argv[i], "--top-p") == 0) {
            const char *v = flag_value(argc, argv, &i);
            if (!v || parse_f32("--top-p", v, &top_p))
                return WRT_EXIT_USAGE;
            has_topp = 1;
        } else if (strcmp(argv[i], "--seed") == 0) {
            const char *v = flag_value(argc, argv, &i);
            if (!v || parse_u64("--seed", v, &seed))
                return WRT_EXIT_USAGE;
            has_seed = 1;
        } else if (strcmp(argv[i], "--max-tokens") == 0) {
            const char *v = flag_value(argc, argv, &i);
            if (!v || parse_u32("--max-tokens", v, &max_tokens))
                return WRT_EXIT_USAGE;
        } else if (strcmp(argv[i], "--prompt") == 0) {
            const char *v = flag_value(argc, argv, &i);
            if (!v)
                return WRT_EXIT_USAGE;
            prompt = v;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "wayrt: unknown generate option \"%s\"\n",
                    argv[i]);
            return WRT_EXIT_USAGE;
        } else if (path) {
            fprintf(stderr, "wayrt: unexpected extra argument \"%s\"\n",
                    argv[i]);
            return WRT_EXIT_USAGE;
        } else {
            path = argv[i];
        }
    }
    if (!prompt) {
        fprintf(stderr, "wayrt: generate requires --prompt TEXT\n");
        return WRT_EXIT_USAGE;
    }
    if (!path) {
        fprintf(stderr, "wayrt: generate requires a model path\n");
        return WRT_EXIT_USAGE;
    }
    if (greedy && (has_temp || has_topk || has_topp || has_seed)) {
        fprintf(stderr, "wayrt: --greedy excludes --temp/--top-k/"
                        "--top-p/--seed\n");
        return WRT_EXIT_USAGE;
    }

    engine = make_engine(opt);
    if (!engine)
        return WRT_EXIT_RUNTIME;

    st = wr_model_load(engine, path, NULL, &model);
    if (st != WR_OK) {
        fail_status("model load", st);
        rc = WRT_EXIT_LOAD;
        goto out;
    }
    st = wr_session_create(model, NULL, &session);
    if (st != WR_OK) {
        fail_status("session create", st);
        goto out;
    }

    /* A NULL sampler means greedy argmax (the parity-tested default);
     * only build one when a sampling flag was actually given. */
    if (has_temp || has_topk || has_topp || has_seed) {
        wr_sample_params sp = wr_sample_params_default();
        if (has_temp)
            sp.temperature = temperature;
        if (has_topk)
            sp.top_k = (int32_t)top_k;
        if (has_topp)
            sp.top_p = top_p;
        if (has_seed)
            sp.seed = seed;
        st = wr_sampler_create(&sp, &sampler);
        if (st != WR_OK) {
            fail_status("sampler create", st);
            goto out;
        }
    }

    {
        wr_generate_params gp  = {0};
        wr_generate_result res;
        size_t             text_len;

        gp.prompt     = prompt;
        gp.max_tokens = max_tokens;
        gp.sampler    = sampler;
        /* Default: frame the prompt with the model's chat template (an
         * instruct model answers; a base model's detection just finds no
         * markers and passes the prompt through).  --raw bypasses both
         * the template and BOS injection for reproducible completions. */
        gp.flags      = raw ? WR_GEN_RAW : WR_GEN_CHAT_TEMPLATE;
        gp.stop_token = -1;

        st = wr_generate(session, &gp, &res);
        if (st != WR_OK) {
            fail_status("generate", st);
            goto out;
        }

        /* Generated text is the ONLY stdout output. */
        fputs(res.text, stdout);
        text_len = strlen(res.text);
        if (text_len == 0 || res.text[text_len - 1] != '\n')
            fputc('\n', stdout);

        fprintf(stderr, "wayrt: %u prompt tokens, %u generated, stop=%s\n",
                res.tokens_in, res.tokens_out,
                stop_reason_name(res.stop_reason));
        wr_free(res.text);
    }

    rc = WRT_EXIT_OK;
out:
    if (sampler)
        wr_sampler_free(sampler);
    if (session)
        wr_session_destroy(session);
    if (model)
        wr_model_free(model);
    wr_engine_destroy(engine);
    return rc;
}

/* ------------------------------------------------------------------ */
/* bench                                                              */
/* ------------------------------------------------------------------ */

static int cmd_bench(cli_opts *opt, int argc, char **argv)
{
    static const char BENCH_PROMPT[] = "Once upon a time";

    const char   *path     = NULL;
    uint32_t      n_tokens = 32;
    wr_engine    *engine;
    wr_model     *model   = NULL;
    wr_tokenizer *tok     = NULL;
    wr_session   *session = NULL;
    wr_model_info info;
    wr_status     st;
    int           rc = WRT_EXIT_RUNTIME;

    for (int i = 0; i < argc; i++) {
        int used = match_global(opt, argc, argv, &i);
        if (used < 0)
            return WRT_EXIT_USAGE;
        if (used == 1)
            continue;
        if (strcmp(argv[i], "--tokens") == 0) {
            const char *v = flag_value(argc, argv, &i);
            if (!v || parse_u32("--tokens", v, &n_tokens))
                return WRT_EXIT_USAGE;
            if (n_tokens == 0) {
                fprintf(stderr, "wayrt: --tokens must be >= 1\n");
                return WRT_EXIT_USAGE;
            }
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "wayrt: unknown bench option \"%s\"\n", argv[i]);
            return WRT_EXIT_USAGE;
        } else if (path) {
            fprintf(stderr, "wayrt: unexpected extra argument \"%s\"\n",
                    argv[i]);
            return WRT_EXIT_USAGE;
        } else {
            path = argv[i];
        }
    }
    if (!path) {
        fprintf(stderr, "wayrt: bench requires a model path\n");
        return WRT_EXIT_USAGE;
    }

    engine = make_engine(opt);
    if (!engine)
        return WRT_EXIT_RUNTIME;

    st = wr_model_load(engine, path, NULL, &model);
    if (st != WR_OK) {
        fail_status("model load", st);
        rc = WRT_EXIT_LOAD;
        goto out;
    }
    st = wr_model_get_info(model, &info);
    if (st != WR_OK) {
        fail_status("model info", st);
        goto out;
    }
    st = wr_tokenizer_from_model(model, &tok);
    if (st != WR_OK) {
        fail_status("tokenizer build", st);
        goto out;
    }
    st = wr_session_create(model, NULL, &session);
    if (st != WR_OK) {
        fail_status("session create", st);
        goto out;
    }

    {
        uint32_t ids[64];
        uint32_t n_prefill, produced = 0, prev;
        double   t0, prefill_s = 0.0, decode_s;
        int      n_in, ctx_hit = 0;
        char     tdesc[16];

        n_in = wr_tokenize(tok, BENCH_PROMPT, ids, 64);
        if (n_in <= 0) {
            fprintf(stderr, "wayrt: bench prompt tokenization failed: %s\n",
                    n_in < 0 ? wr_status_str((wr_status)n_in) : "0 tokens");
            goto out;
        }

        /* Prefill everything but the last prompt token (the LM head is
         * skipped during prefill, which isolates prompt-time cost),
         * then greedy-decode from the last token.  The decode loop
         * deliberately runs THROUGH any EOS the model emits so the
         * measured workload is a fixed token count. */
        n_prefill = (uint32_t)n_in - 1;
        if (n_prefill > 0) {
            int pf;
            t0        = now_sec();
            pf        = wr_prefill(session, ids, n_prefill);
            prefill_s = now_sec() - t0;
            if (pf < 0) {
                fail_status("prefill", (wr_status)pf);
                goto out;
            }
        }

        prev = ids[n_in - 1];
        t0   = now_sec();
        for (uint32_t i = 0; i < n_tokens; i++) {
            const float *logits = wr_step(session, prev);
            if (!logits) {
                wr_status ss = wr_session_status(session);
                if (ss == WR_ERR_CTX_FULL) {
                    ctx_hit = 1;
                    break;
                }
                fail_status("decode step", ss);
                goto out;
            }
            prev = argmax_f32(logits, info.vocab_size);
            produced++;
        }
        decode_s = now_sec() - t0;

        if (produced == 0) {
            fprintf(stderr, "wayrt: context full before any decode step "
                            "(max_context %u, prompt %d tokens)\n",
                    info.max_context, n_in);
            goto out;
        }

        if (opt->threads)
            snprintf(tdesc, sizeof tdesc, "%u", opt->threads);
        else
            snprintf(tdesc, sizeof tdesc, "auto");

        printf("model:    %s\n", path);
        printf("arch:     %s  quant: %s  layers: %u  hidden: %u  "
               "vocab: %u  ctx: %u\n",
               info.arch, info.quant, info.n_layers, info.hidden_dim,
               info.vocab_size, info.max_context);
        printf("simd:     %s  threads: %s\n",
               simd_name(wr_engine_simd_variant(engine)), tdesc);
        if (n_prefill > 0)
            printf("prefill:  %u tokens  %.2f ms  %.1f tok/s\n",
                   n_prefill, prefill_s * 1e3,
                   rate_tok_s(n_prefill, prefill_s));
        else
            printf("prefill:  0 tokens (single-token prompt)\n");
        printf("decode:   %u tokens  %.2f ms  %.1f tok/s%s\n",
               produced, decode_s * 1e3, rate_tok_s(produced, decode_s),
               ctx_hit ? "  (stopped: context full)" : "");
        print_counters(engine);
    }

    rc = WRT_EXIT_OK;
out:
    if (session)
        wr_session_destroy(session);
    if (tok)
        wr_tokenizer_free(tok);
    if (model)
        wr_model_free(model);
    wr_engine_destroy(engine);
    return rc;
}

/* ------------------------------------------------------------------ */
/* chat                                                               */
/* ------------------------------------------------------------------ */

static int chat_on_token(uint32_t token_id, const char *piece, void *user)
{
    (void)token_id;
    (void)user;
    fputs(piece, stdout);
    fflush(stdout);
    return 0;
}

static int cmd_chat(cli_opts *opt, int argc, char **argv)
{
    const char   *path = NULL;
    wr_engine    *engine;
    wr_model     *model   = NULL;
    wr_session   *session = NULL;
    wr_sampler   *sampler = NULL;
    wr_model_info info;
    wr_status     st;
    int           rc = WRT_EXIT_RUNTIME;

    if (parse_model_only(opt, argc, argv, "chat", &path))
        return WRT_EXIT_USAGE;

    engine = make_engine(opt);
    if (!engine)
        return WRT_EXIT_RUNTIME;

    st = wr_model_load(engine, path, NULL, &model);
    if (st != WR_OK) {
        fail_status("model load", st);
        rc = WRT_EXIT_LOAD;
        goto out;
    }
    st = wr_model_get_info(model, &info);
    if (st != WR_OK) {
        fail_status("model info", st);
        goto out;
    }
    st = wr_session_create(model, NULL, &session);
    if (st != WR_OK) {
        fail_status("session create", st);
        goto out;
    }

    /* Greedy plus a mild repetition penalty: deterministic, and the
     * penalty breaks the token loops small quantized models fall into
     * under pure argmax during open-ended chat. */
    {
        wr_sample_params sp = wr_sample_params_default();
        sp.repeat_penalty = 1.3f;
        sp.repeat_last_n  = 64;
        st = wr_sampler_create(&sp, &sampler);
        if (st != WR_OK) {
            fail_status("sampler create", st);
            goto out;
        }
    }

    fprintf(stderr,
            "wayrt chat: %s (%s), context %u tokens.  Type a message and "
            "press Enter; EOF (Ctrl-D / Ctrl-Z) ends the chat.\n",
            info.arch, info.quant, info.max_context);

    for (;;) {
        char   line[16384];
        size_t len;

        fputs("> ", stderr);
        fflush(stderr);
        if (!fgets(line, (int)sizeof line, stdin))
            break; /* EOF: clean exit */
        len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        if (len == 0)
            continue;

        /* Fresh penalty history per turn; the conversation itself
         * persists in the session's KV cache across wr_generate calls. */
        wr_sampler_reset(sampler, 0);

        {
            wr_generate_params gp  = {0};
            wr_generate_result res;

            gp.prompt     = line;
            gp.max_tokens = 256;
            gp.sampler    = sampler;
            gp.flags      = WR_GEN_CHAT_TEMPLATE;
            gp.stop_token = -1;
            gp.on_token   = chat_on_token;

            st = wr_generate(session, &gp, &res);
            if (st != WR_OK) {
                if (st == WR_ERR_CTX_FULL)
                    fprintf(stderr,
                            "wayrt: context window is full (%u tokens) — "
                            "restart wayrt chat for a new conversation\n",
                            wr_session_max_context(session));
                else
                    fail_status("generate", st);
                goto out;
            }
            fputc('\n', stdout);
            fflush(stdout);
            fprintf(stderr, "[%u tokens, stop=%s, ctx %u/%u]\n",
                    res.tokens_out, stop_reason_name(res.stop_reason),
                    wr_session_pos(session),
                    wr_session_max_context(session));
            wr_free(res.text);
        }
    }

    rc = WRT_EXIT_OK;
out:
    if (sampler)
        wr_sampler_free(sampler);
    if (session)
        wr_session_destroy(session);
    if (model)
        wr_model_free(model);
    wr_engine_destroy(engine);
    return rc;
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    cli_opts    opt = {0};
    const char *cmd;
    int         i = 1;

#if defined(_WIN32)
    /* The Windows CRT defaults stdout to text mode, which rewrites LF
     * as CRLF even when stdout is redirected.  Generated text is a
     * byte-oriented interface, so preserve the exact bytes on every
     * platform (and make the Windows/POSIX parity test literal). */
    if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
        fputs("wayrt: cannot set stdout to binary mode\n", stderr);
        return WRT_EXIT_RUNTIME;
    }
#endif

    /* Global flags may precede the command name. */
    while (i < argc && argv[i][0] == '-') {
        int used;
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(stdout);
            return WRT_EXIT_OK;
        }
        used = match_global(&opt, argc, argv, &i);
        if (used < 0)
            return WRT_EXIT_USAGE;
        if (used == 0) {
            fprintf(stderr, "wayrt: unknown option \"%s\"\n", argv[i]);
            usage(stderr);
            return WRT_EXIT_USAGE;
        }
        i++;
    }
    if (i >= argc) {
        usage(stderr);
        return WRT_EXIT_USAGE;
    }

    cmd = argv[i++];
    if (strcmp(cmd, "verify") == 0)
        return cmd_verify(&opt, argc - i, argv + i);
    if (strcmp(cmd, "generate") == 0)
        return cmd_generate(&opt, argc - i, argv + i);
    if (strcmp(cmd, "bench") == 0)
        return cmd_bench(&opt, argc - i, argv + i);
    if (strcmp(cmd, "chat") == 0)
        return cmd_chat(&opt, argc - i, argv + i);

    fprintf(stderr, "wayrt: unknown command \"%s\"\n", cmd);
    usage(stderr);
    return WRT_EXIT_USAGE;
}
