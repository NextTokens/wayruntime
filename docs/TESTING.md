# Testing wayruntime

Five layers, each answering a different question. All of them are in
the repo and rerunnable; none of them requires the others.

| layer | entry point | needs | proves |
|---|---|---|---|
| 1. Unit + golden | `make unit` | nothing | the numeric primitives are correct |
| 2. Integration | `make integration` | tiny test models, offline | the library and CLI wire them together |
| 3. Real model | `make realtest MODEL_GGUF=...` | a real GGUF | the shipped path works end to end |
| 4. Differential | `test/realtest_diff.sh` | a llama.cpp server + the same GGUF | decode agrees with the reference, token for token |
| 5. Windows native | `test\realtest.ps1` | both builds + WSL | the same product on Windows, byte-identical output |

`make test` is the full local gate: `license-check` (SPDX/REUSE
boundary, `test/check_license_metadata.py`) + `check-api` (the public
header must compile freestanding under `-std=c11 -pedantic -Werror`
with no `src/` include path) + `unit` + `integration`.

Current standing (2026-08-30): 55/55 unit checks; 69/69 integration
(CLI + hostile-GGUF + SDK harness + sampler statistics); real-model
realtest PASS; differential vs llama.cpp on Qwen3-0.6B Q4_K_M:
tokenizer corpus 20/20 exact and 20/20 teacher-forced argmax
agreement (a 78-string corpus scored 78/78 during development);
Windows-native battery 7/7 including byte-identical greedy output
Linux vs Windows.

---

## 1. Unit + golden self-tests

```sh
make unit                            # builds and runs build/posix/wayrt_tests
make WIN=1 build/win/wayrt_tests.exe # same suite, Windows build
```

55 checks. The suite pins the contracts a numerics reviewer pokes
first — dtype enum values (the GGUF mapping and the goldens depend on
them), refusal semantics, and API lifetime rules — and embeds **9
golden numeric self-tests** carried over from the engine this library
was extracted from:

- `quant_dequant` — quantization codec round-trips
- `matmul_simd` — every bound SIMD variant against the scalar kernel
- `softmax`, `flash_attn`, `fused`, `ops` — kernel-level numerics
- `qmm_bit_equal` — quantized matmul bit-equality across paths
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
documented errors, not run). See `test/integration.sh` for the
current check list.

This layer needs no real model and no network; `make test` includes it
as the full local/CI gate.

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
run on Qwen3-0.6B Q4_K_M agrees 20/20 on forced-decode argmax.

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
header of `test/realtest_diff.sh` for its parameters. Residual
tokenizer divergences on exotic whitespace are documented in code and
tracked against the corpus this layer runs.

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

## What is deliberately NOT covered yet

- Gemma-class models: the forward path and goldens exist, but no
  real Gemma GGUF run is recorded — treat the support as
  experimental until layer 3/4 evidence lands.
- fuzzing of the GGUF and tokenizer parsers (planned pre-release
  hardening; see `docs/SECURITY.md`)
- macOS and big-endian hosts (the latter are refused at engine
  creation by design)
