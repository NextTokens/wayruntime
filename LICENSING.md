# wayruntime licensing

wayruntime is a mixed-license repository. The public API surface is
open source under Apache License 2.0. The runtime core is
source-available under Business Source License 1.1 and is not open
source before its Change Date.

An embedded `SPDX-License-Identifier` is authoritative when present.
`REUSE.toml` supplies or confirms licensing for files without an
embedded identifier. If this guide conflicts with a license or SPDX
metadata, the license and SPDX metadata control.

## BUSL-1.1 core

The following are BUSL-1.1:

- `src/core/**` — engine, SIMD kernels, quantization codecs, GGUF
  loader, tokenizer, model graph, sessions, batched decode, sampler,
  generate facade
- `src/platform/**` and `src/cli/**`
- the C test sources that include internal headers
  (`test/unit_tests.c`)
- core-specific documentation mapped as BUSL-1.1 in `REUSE.toml`

BUSL always permits copying, modification, redistribution, derivative
works, and non-production use. The Additional Use Grant in `LICENSE`
also permits the specified limited production uses. Other production
use requires a separate commercial license from WayOS Project until
the Change Date.

The Change Date for wayruntime 0.1.0 is **2030-08-30**. On that date,
or earlier if required by BUSL's fourth-anniversary rule, the covered
0.1.0 core becomes available under Apache-2.0. Future versions must
publish their own unambiguous version and Change Date.

BUSL restricts production use; it does not prohibit redistribution.
Charging for copies or support does not, by itself, alter the rights
BUSL grants or the restrictions imposed on production use.

## Apache-2.0 API surface

The following are Apache-2.0:

- `include/wayruntime/**` — the complete public C API
- `examples/**`
- the root `Makefile`
- the standalone test scripts and gates (`test/check_api.c`,
  `test/check_license_metadata.py`, `test/realtest.sh`)
- project documentation and metadata mapped as Apache-2.0 in
  `REUSE.toml`

The public header is deliberately freestanding (`make check-api`
enforces it): applications, bindings, and headers-only tooling can be
written against `include/wayruntime/wayruntime.h` under Apache-2.0
alone.

Unlike a repository with a separable client SDK, the *implementation*
behind that header is the BUSL core: `libwayruntime.a` and `wayrt`
are built from BUSL-1.1 sources. Programs that link the library
therefore contain BUSL-licensed code, and BUSL's production-use terms
apply to them until the Change Date. The root Makefile's Apache
license does not change the licenses of the files or programs it
builds.

## Contributions

External contributions are currently accepted only for the
Apache-2.0 surface, under Apache-2.0 with DCO 1.1 sign-off. Outside
code is not accepted for the BUSL core. There is no CLA under this
policy. See `CONTRIBUTING.md`.

## Separate distributions by the Licensor

Owner-authored portions may also appear in other WayOS Project
distributions under other terms, including GPL-3.0-or-later. A
separate distribution does not change the terms of this repository.
Only material for which WayOS Project holds sufficient rights is
offered here under BUSL or under a separate commercial agreement.
