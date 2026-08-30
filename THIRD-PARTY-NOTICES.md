# Third-party notices

No third-party implementation source is vendored. The project records
the GGUF loader, BPE tokenizer, quantization codecs, SIMD kernels,
transformer forward pass, worker pool, sampler, and CLI as
first-party implementations. This file inventories what wayruntime
*links against*, which matters for binary distributions.

| Component | Use | License | Linkage |
|---|---|---|---|
| libc, libpthread | C runtime and the worker pool (POSIX builds) | system | dynamic, system-provided |
| libm | transcendental math **only** in `MATH_APPROX=0` builds; the default build's hot path uses first-party polynomial approximations | system | dynamic, system-provided |
| Win32 (kernel32 et al.) | Windows threads, file mapping, console | Windows OS components | OS API, not distributed |
| mingw-w64 runtime | Windows builds' CRT startup | permissive w/ linking exceptions | static per its exception terms |

There is no networking dependency of any kind: no libcurl, no
OpenSSL, no sockets — the tree contains no network code.

Every Windows binary package must include the exact
`COPYING.MinGW-w64-runtime.txt` shipped with the mingw-w64 toolchain
used for that release; release packaging sources it from the
toolchain installation and must confirm it against the actual
toolchain version.

Rules for keeping this true (mirrored in CONTRIBUTING.md):

- Do not import third-party source without recording it here with
  its license, adding the full license text under `LICENSES/`, and
  getting maintainer approval first — the BUSL-1.1 license on
  first-party code cannot absorb copyleft or incompatible code.
- If a binary release ever bundles anything beyond the static
  library and CLI built from this tree, add the corresponding
  license texts to the release artifacts and update this table.
