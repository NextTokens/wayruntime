# Contributing to wayruntime

Thanks for your interest. wayruntime is an inference engine whose
selling points are determinism, refusal semantics, and a
provenance-auditable tree; the contribution bar is deliberately high.

## Accepted contribution scope

Contributions are accepted for the whole repository — the public
header, the runtime core, the platform layers, the CLI, the examples,
the tests, and the documentation. There are no off-limits paths.

The bar is evidence, not permission: read the engineering rules
below before opening a pull request, because they are what a change
is measured against.

There is no CLA. Outside code is accepted under Apache-2.0 with a DCO
1.1 sign-off, and contributors retain copyright in their own
contributions.

## Contributions and DCO

By intentionally submitting a contribution for inclusion, you submit
it under Apache License 2.0 and retain copyright in your
contribution.

Every commit in an external pull request must carry a **DCO 1.1
sign-off**: `Signed-off-by: Full Name <email@example.com>`. The
sign-off certifies the unmodified [Developer Certificate of
Origin](DCO).

New files must carry an Apache-2.0 SPDX identifier and appropriate
SPDX copyright information; `make license-check` enforces both.

**Third-party code:** do not import any without prior maintainer
approval, a row in `THIRD-PARTY-NOTICES.md`, and the full license
text under `LICENSES/`. The default answer is no — the tree is
deliberately kept 100% first-party and provenance-auditable.

## Engineering rules

1. **No scaffolding.** A change is complete only when every consumer
   it implies is wired, error paths are production-grade, tests
   exist at the right layer (see `docs/TESTING.md`), and the docs
   match reality. No `TODO` stubs in shipped paths.
2. **Numeric claims need receipts.** The bit-exactness invariants
   are load-bearing: parallel matmul output is identical for every
   worker-thread count, and batched decode is bit-exact versus
   serial stepping. The golden self-tests must stay green, and a
   claim of numerical equivalence comes with the test that proves
   it — differential runs against the reference implementation
   (`docs/TESTING.md`) for anything touching the decode path.
3. **Fail loudly; never clamp.** There is no silent truncation or
   silent fallback anywhere in the API: context overflow is
   `WR_ERR_CTX_FULL`, an unsupported dtype or architecture is
   `WR_ERR_UNSUPPORTED`, an explicitly requested SIMD level the host
   cannot bind is an error. New surfaces follow the same rule, and
   documented refusals keep their tests.
4. **The GGUF file is hostile input.** Every offset, length, and
   count read from a model file is validated against the real file
   before use; malformed input produces a structured error, never a
   partial load. Model tensors are immutable after load — no code
   path may write to loaded weights.
5. **Both platforms.** `make` and `make WIN=1` build with zero
   warnings under `-Wall -Wextra -Wshadow -Wvla`; `make test`
   passes. Real-model and differential suites (`docs/TESTING.md`
   §3–5) are required for changes touching load, tokenize, decode,
   or sampling behavior.
6. **AI-assisted is welcome; unevidenced is not.** Disclose heavy
   tool assistance in the PR; every line is held to the same
   evidence bar regardless of who or what wrote it.

## Workflow

1. Branch from `main`; keep commits per-slice and buildable.
2. Run the relevant suites; say in the PR exactly which you ran.
3. Sign off every allowed Apache-2.0 contribution.
4. Never commit model files (`*.gguf`), build artifacts, or secrets.

## Security issues

Do not open public issues for vulnerabilities — see
`docs/SECURITY.md` for private reporting.
