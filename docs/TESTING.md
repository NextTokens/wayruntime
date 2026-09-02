# Testing wayruntime

Five layers, each answering a different question. All of them are in
the repo and rerunnable; none of them requires the others.

| layer | entry point | needs | proves |
|---|---|---|---|
| 1. Unit + golden | `make unit` | nothing | the numeric primitives are correct |
| 2. Integration | `make integration` | one small llama-family GGUF, offline | the library and CLI wire them together |
| 3. Real model | `make realtest MODEL_GGUF=...` | a real GGUF | the shipped path works end to end |
| 4. Differential | `test/realtest_diff.sh` | a llama.cpp server + the same GGUF | decode agrees with the reference, token for token |
| 5. Windows native | `test\realtest.ps1` | both builds + WSL | the same product on Windows, byte-identical output |

`make test` is the offline gate and runs from a bare clone with no
model files: `license-check` (SPDX/REUSE boundary,
`test/check_license_metadata.py`) + `check-api` (the public header
must compile freestanding under `-std=c11 -pedantic -Werror` with no
`src/` include path) + `unit`.  `make check` runs all of that plus
`integration`, which needs the one small model described in layer 2.

Performance measurement is deliberately separate from these correctness
gates. `make benchmark` is the repeatable local baseline described below;
it never runs as part of `make test`.

Current standing (2026-09-01): 57/57 unit checks; 77/77 integration
(CLI + hostile-GGUF + SDK harness + sampler statistics + concurrent
sessions vs sequential); real-model
realtest PASS; differential vs llama.cpp on Qwen3-0.6B Q4_K_M:
tokenizer corpus 20/20 exact, the compiled-in 78-string fixture corpus
78/78 exact (re-proven on every run), and 20/20 teacher-forced argmax
agreement;
Windows-native battery 7/7 including byte-identical greedy output
Linux vs Windows.

---

## 1. Unit + golden self-tests

```sh
make unit                            # builds and runs build/posix/wayrt_tests
make WIN=1 build/win/wayrt_tests.exe # same suite, Windows build
```

57 checks. The suite pins the contracts a numerics reviewer pokes
first — dtype enum values (the GGUF mapping and the goldens depend on
them), refusal semantics, and API lifetime rules — and embeds **10
golden numeric self-tests** carried over from the engine this library
was extracted from:

- `quant_dequant` — quantization codec round-trips
- `matmul_simd` — every bound SIMD variant against the scalar kernel
- `softmax`, `flash_attn`, `fused`, `ops` — kernel-level numerics
- `qmm_bit_equal` — quantized matmul bit-equality across paths
- `f32_ggml_nsplit` — output-column parallelism bit-equal to the serial head
- `llm_step` — a full transformer step against known-good values
- `batch_step` — batched decode bit-exact versus serial stepping

The goldens run with the engine live, so SIMD dispatch and the worker
pool are exercised, not simulated.

## 2. Integration suite (offline)

```sh
make integration                     # = bash test/integration.sh
```

Drives the built `wayrt` CLI and the library against small local
test models entirely offline: load/verify metadata, deterministic
generation, sampling flags, exit codes, and the refusal paths (a
wrong-architecture GGUF and unsupported dtypes must fail with the
documented errors, not run). It also observes the default mapping attempt
(successful mapping or the documented streamed fallback) and proves that
`--no-mmap` produces identical model metadata without attempting a mapping.
Its concurrency gate drives eight threads of sessions on the one model
(plus a batched step alongside a single step) and requires every
step's logits to be byte-identical to the sequential run.
See `test/integration.sh` for the current check list.

This layer needs no network and exactly one small real model, named
by `WR_TEST_LLAMA` (default `../models/stories15M-q8_0.gguf`, or set
`MODELS_DIR`).  Any llama-architecture GGUF works.  The suite was
validated with **stories15M**: the 15M-parameter TinyStories
checkpoint published by the llama2.c project, converted with
llama.cpp's `convert-llama2c-to-ggml` tool and quantized to Q8_0 with
`llama-quantize`.  Three optional 1-layer K-quant files
(`WR_TEST_Q4K` / `WR_TEST_Q5K` / `WR_TEST_Q6K`, defaults
`../models/tiny-q{4,5,6}k-test.gguf`) additionally exercise the
Q4_K / Q5_K / Q6_K load paths; when one is absent the script prints a
SKIP line for that step rather than counting it.  The
unsupported-architecture refusal case and the eight hostile GGUFs are
synthesized by the script itself.  `make check` = `make test` + this
layer.

## 3. Real-model end-to-end

```sh
make realtest MODEL_GGUF=path/to/model.gguf
# equivalently: bash test/realtest.sh build/posix path/to/model.gguf
```

No mocks: loads a real GGUF through the real CLI and asserts that
`verify` reports sane metadata, that greedy generation is non-empty
and **deterministic across two runs**, and that the engine counters
show SIMD fast-path activity (a nonzero per-element-fallback delta
means a fast path was missed).

Any llama-family or Qwen3 GGUF works; a small model (a 15M-parameter
storyteller, a 0.6B chat model) keeps the run to seconds.

## 4. Differential against llama.cpp

```sh
bash test/realtest_diff.sh
```

The strongest correctness claim in the repo: wayruntime and llama.cpp
load the **same GGUF** and must agree on teacher-forced argmax,
token for token, and on tokenization over a text corpus. The recorded
run on Qwen3-0.6B Q4_K_M agrees 20/20 on forced-decode argmax.  The
compiled-in 78-string tokenizer fixture corpus
(`test/tokenizer_fixtures.h`: whitespace runs, code, CJK, contractions,
control markers with the parse-special flag) is replayed against the
reference on every run and must match exactly.

The checked-in acceptance bars are 19/20 exact tokenizer sequences and
20/20 teacher-forced argmax agreement. The direct script and Make target
use the same defaults. For a deliberately relaxed diagnostic run, pass
explicit thresholds either positionally or through Make:

```sh
bash test/realtest_diff.sh build/posix 14 19
make realtest-diff DIFF_TOKENIZER_MIN=14 DIFF_TEACHER_MIN=19
```

Those overrides are diagnostic controls; release evidence uses the
checked-in defaults.

The reference side is a stock `llama-server` (endpoints `/health`,
`/tokenize`, `/completion`):

```sh
./llama-server -m Qwen3-0.6B-Q4_K_M.gguf --host 127.0.0.1 --port 18080 -c 4096
```

Point the script at that server and the matching GGUF path; see the
header of `test/realtest_diff.sh` for its parameters. The byte-level
pretokenizer classifies Unicode with compact range tables rather than
the full Unicode database, so scripts outside those tables could split
at different boundaries than the reference; every script in the
fixture corpus matches, and any new divergence surfaces here first.

## 5. Windows native

```powershell
make WIN=1                           # cross-compile with mingw-w64
powershell -File test\realtest.ps1
```

Runs the real-model battery natively on Windows against
`build\win\wayrt.exe`. The bar is not "it runs" but **byte-identical
greedy output** to the Linux build for the same model and prompt —
the cross-platform determinism claim in the README is this test.
The parity step is required by default, so the command also requires
WSL and `build/posix/wayrt`; missing parity prerequisites are a setup
error (exit 2), not a passing skip. For an intentional Windows-only
diagnostic run, pass `-SkipCrossPlatform`; its summary explicitly says
that cross-platform parity was not run and therefore does not validate
the byte-identical-output claim.

## Performance baseline (non-gating)

Run this harness on a POSIX host build. `WIN=1` is a cross-compile mode,
so `make WIN=1 benchmark` exits with a clear diagnostic instead of trying
to execute the resulting Windows binary on the build host.

```sh
make clean
make -s benchmark MODEL_GGUF=path/to/model.gguf
make -s benchmark MODEL_GGUF=path/to/model.gguf BENCH_THREADS=1,2,4,8
```

The harness runs one discarded warmup and three measured 32-token decode
samples at each requested thread count, then emits comma-delimited sample
and summary report sections with means and sample standard deviations.
Rates are recomputed from the CLI's elapsed milliseconds rather than its
rounded human-readable rate. Defaults are fixed rather than derived from
the current machine: `BENCH_THREADS=1,2,4`, `BENCH_TOKENS=32`,
`BENCH_REPETITIONS=3`, and `BENCH_WARMUPS=1`. Override them explicitly to
define another workload. The model has no default; every report names the
GGUF path and SHA-256, repository worktree commit/state, exact wayruntime
executable path and SHA-256, and host CPU/OS. Use `make -s` when redirecting
so build command echo does not contaminate the report.

Each wayruntime sample is a fresh process and model load. The discarded run
warms the operating-system file cache and CPU, not a persistent model
instance; with mmap, the measured fixed-prompt prefill can still include
first-touch page faults. Treat it as first-request prompt latency rather
than a steady-state long-prompt throughput test.

An installed llama.cpp `llama-bench` can be included in the same run without
putting a developer-specific path in the repository:

```sh
make -s benchmark MODEL_GGUF=path/to/model.gguf \
  BENCH_THREADS=1,2,4,8 LLAMA_BENCH=/path/to/llama-bench \
  > benchmark-report.txt
```

The optional command pins llama.cpp to CPU execution (`-ngl 0`), the same
thread list and decode-token count, and prints its native CSV including its
build identifier. Treat the two decode rates as directional rather than a
perfectly controlled microbenchmark: `wayrt bench` includes its greedy
argmax scan and starts after the fixed `Once upon a time` prompt, whereas
`llama-bench`'s generation test excludes sampling and controls its own
synthetic context. Always compare the same model file, build flags, power
state, and otherwise idle host.

The command plan itself can be checked without a build or model:

```sh
BENCH_DRY_RUN=1 BENCH_THREADS=1,4 LLAMA_BENCH=/path/to/llama-bench \
  bash test/benchmark.sh build/posix placeholder.gguf
```

## What is deliberately NOT covered yet

- Gemma-class models: the forward path and goldens exist, but no
  real Gemma GGUF run is recorded — treat the support as
  experimental until layer 3/4 evidence lands.
- fuzzing of the GGUF and tokenizer parsers (planned pre-release
  hardening; see `docs/SECURITY.md`)
- macOS and big-endian hosts (the latter are refused at engine
  creation by design)
