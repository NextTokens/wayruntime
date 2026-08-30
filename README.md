# wayruntime

![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue)
![API surface: Apache-2.0](https://img.shields.io/badge/API_surface-Apache--2.0-green)
![Apache contributions: DCO 1.1](https://img.shields.io/badge/Apache_contributions-DCO_1.1-lightgrey)

The public API surface is open source under Apache-2.0. The runtime
core is source-available under BUSL-1.1 and becomes Apache-2.0 on
2030-08-30. The whole repository is therefore not open source before
that date.

**A self-contained CPU inference runtime for GGUF language models.**
One static library, one public header, one CLI. It loads a GGUF file
with a hostile-input posture, tokenizes, runs the transformer decode
with a per-session KV cache, samples, and streams tokens back — with
zero third-party code and zero network code in the tree.

> **Lineage.** wayruntime is extracted from the AI stack of an
> unreleased, from-scratch operating system, where local inference is
> a built-in system service rather than an application dependency.
> This is that engine, matured and ported to Linux and Windows. The
> OS ships later; this runs today.

## Why

- **No dependency surface.** The library is 100% first-party C11:
  no BLAS, no vendored parsers, no HTTP client — the tree contains
  no network code at all (`grep -ri socket src include` comes back
  empty). What you audit is what runs.
- **Determinism as a feature.** Parallel matmul is bit-exact: output
  is identical for any worker-thread count. Batched decode is
  bit-exact versus serial stepping. Samplers are seeded and replay
  deterministically. Greedy output is byte-identical between the
  Linux and Windows builds of the same model.
- **Verified against the reference.** Greedy decode agrees with
  llama.cpp 20/20 on teacher-forced argmax over the same GGUF
  (Qwen3-0.6B); byte-level tokenization is canonical BPE with the
  Qwen2-family pretokenizer, scoring 78/78 exact id-sequence matches
  against llama.cpp's tokenizer on a mixed corpus.
- **Errors, never guesses.** Unsupported architectures and tensor
  dtypes are refused explicitly. Context overflow is an error, not a
  silent truncation. A requested SIMD level the host cannot bind is
  an error, never a silent fallback.

## What's inside

- **GGUF loader**, hardened: header version gate, tensor bounds and
  overlap validation against the real file size, hostile-input
  posture throughout (see [`docs/SECURITY.md`](docs/SECURITY.md))
- **BPE tokenizer** with byte-level and SentencePiece modes
- **Transformer decode** for llama-family, Qwen3, and Gemma-class
  (experimental) models: GQA, flash-attention tiling, RoPE, F16 KV
  cache
- **Quantized weights**: F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K,
  Q6_K compute paths; Q4_1/Q5_0/Q5_1 are refused explicitly rather
  than mis-executed
- **Runtime SIMD dispatch**: scalar, AVX2, AVX-512, NEON — probed at
  engine creation, switchable at runtime
- **Persistent worker pool** with bit-exact parallel matmul
- **Batched decode**: up to 16 sessions stepped together, their
  LM-head projections coalesced into one GEMM, bit-exact versus
  serial
- **Sampler**: greedy, temperature, top-k, top-p, repetition
  penalty; seeded, deterministic replay
- **Streaming token callback** and **grammar/token-mask constrained
  decoding** (see [`examples/grammar_mask.c`](examples/grammar_mask.c))
- **Chat-template autodetect**: ChatML variants including Qwen3
  `/no_think` handling, Gemma turn markers

## Quick start

```sh
make && make test          # needs gcc, make, python3; zero warnings
./build/posix/wayrt verify  model.gguf     # load, print metadata, unload
./build/posix/wayrt generate --prompt "The capital of France is" model.gguf
./build/posix/wayrt chat    model.gguf     # interactive, streaming
./build/posix/wayrt bench   model.gguf     # tokens/s + engine counters
```

Windows: `make WIN=1` cross-compiles `build/win/wayrt.exe`
(mingw-w64). Same commands, same output — greedy generation is
byte-identical to the Linux build.

## Using the library

Everything is behind one header, `include/wayruntime/wayruntime.h`
(Apache-2.0, freestanding, C11): engine → model → session →
generate.

```c
#include <stdio.h>
#include <wayruntime/wayruntime.h>

int main(int argc, char **argv)
{
    (void)argc;
    wr_engine *eng; wr_model *mdl; wr_session *ses;
    if (wr_engine_create(NULL, &eng) != WR_OK) return 1;
    if (wr_model_load(eng, argv[1], NULL, &mdl) != WR_OK) return 1;
    if (wr_session_create(mdl, NULL, &ses) != WR_OK) return 1;
    wr_generate_params p = { .prompt = "The capital of France is",
                             .max_tokens = 32, .stop_token = -1 };
    wr_generate_result out;
    if (wr_generate(ses, &p, &out) == WR_OK) { printf("%s\n", out.text); wr_free(out.text); }
    wr_session_destroy(ses); wr_model_free(mdl); wr_engine_destroy(eng);
    return 0;
}
```

```sh
gcc -std=c11 -O2 -Wall -Wextra -Iinclude hello.c build/posix/libwayruntime.a -lpthread -lm
./a.out model.gguf
```

This example compiles warning-free and runs, exactly as shown,
against the built library. Lower-level control (prefill/step loops,
raw logits, batched stepping, token masks) is the same header; the
`wr_generate` facade is just the short path.

## Status — what works, what doesn't

Works today, and is covered by layered tests (unit + golden numeric
self-tests, an offline integration suite, real-model CLI runs, a
differential suite against llama.cpp, and native Windows runs — see
[`docs/TESTING.md`](docs/TESTING.md) for how to run each layer and
what it proves):

- zero-warning C11 build on gcc and mingw-w64
  (`-Wall -Wextra -Wshadow -Wvla`)
- 55/55 unit checks, including 9 golden numeric self-tests, plus a
  69-check integration battery (hostile-GGUF refusals included)
- greedy decode verified 20/20 teacher-forced argmax agreement with
  llama.cpp on the same GGUF (Qwen3-0.6B)
- tokenization verified against llama.cpp's tokenizer on the same
  GGUF: 78/78 exact id-sequence matches on a mixed corpus (prose,
  code, multi-space runs, CJK, contractions, special tokens)
- byte-identical greedy output, Linux vs Windows
- no libm dependency in the default build's hot path (first-party
  polynomial math; `make MATH_APPROX=0` selects libm instead)

Not yet (deliberately, v0.1):

- CPU only — no GPU backends
- library + CLI only — no server, no streaming HTTP endpoint
- little-endian hosts only (big-endian is refused at engine
  creation, not mis-run)
- gcc / mingw-w64 only: the kernels use GCC vector extensions;
  MSVC is out of scope by design
- the byte-level pretokenizer classifies Unicode with compact range
  tables, not the full Unicode database: scripts outside the tables
  can split at different boundaries than upstream (bytes are never
  altered); the SentencePiece mode has no pretokenizer, matching the
  origin engine it was validated against
- Gemma-class model support is experimental and not yet
  real-model-tested
- default context cap is 4096 tokens (raise per model via
  `wr_model_params.max_context`, up to the compiled attention cap)
- no external security audit — see
  [`docs/SECURITY.md`](docs/SECURITY.md)

## Layout

```
include/wayruntime/     the public C API — one freestanding header
src/core/               engine, SIMD kernels, quant codecs, GGUF loader,
                        tokenizer, model graph, sessions, batch, sampler
src/platform/posix,win/ threads, file mapping, CPU feature probes
src/cli/                the wayrt command-line tool
examples/               Apache-2.0 SDK examples (grammar-masked decoding)
test/                   unit + golden, integration, real-model suites
docs/                   SECURITY.md, TESTING.md
```

## Security

The trust boundary is the GGUF file: it is treated as hostile input
until validated. [`docs/SECURITY.md`](docs/SECURITY.md) describes the
threat model and how to report vulnerabilities privately — do not
open public issues for security reports.

## License

This is a mixed-license repository. Embedded SPDX identifiers and
[`REUSE.toml`](REUSE.toml) define the exact boundary; the complete
explanation is in [`LICENSING.md`](LICENSING.md).

- **Core** (`src/`: engine, kernels, loader, tokenizer, sessions,
  sampler, CLI): Business Source License 1.1. Non-production use and
  the limited production uses in `LICENSE` are free. Other production
  use needs a separate commercial license until the fixed Change
  Date, 2030-08-30.
- **API surface** (the public header, the examples, the Makefile,
  and the repository's test scripts and metadata as mapped in
  `REUSE.toml`): Apache-2.0. Code you write against the header is
  yours under Apache-2.0; the library you link (`libwayruntime.a`)
  is built from BUSL sources, so BUSL terms govern binaries that
  contain the core until the Change Date.

BUSL permits redistribution and restricts production use; it does
not promise payment for every form of resale or support. See
[`COMMERCIAL-LICENSING.md`](COMMERCIAL-LICENSING.md) for the
commercial-production route.

External code contributions are currently accepted only for the
Apache-2.0 surface and require DCO 1.1 sign-off. The BUSL core does
not accept outside code, and no CLA is currently required. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`DCO`](DCO). What the
binaries link against is inventoried in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).
