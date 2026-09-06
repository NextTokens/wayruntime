# wayruntime licensing

wayruntime is open source under the Apache License 2.0. Every
first-party file in this repository — the public header, the runtime
core, the platform layers, the CLI, the examples, the tests, and the
documentation — is Apache-2.0. There is no separate core license, no
production-use restriction, and no commercial license to buy.

An embedded `SPDX-License-Identifier` is authoritative when present.
`REUSE.toml` supplies licensing for files without an embedded
identifier. If this guide conflicts with a license or SPDX metadata,
the license and SPDX metadata control.

## What that means in practice

- You may use wayruntime in production, commercially, and in
  closed-source software, subject to Apache-2.0's notice and
  attribution requirements.
- Programs that link `libwayruntime.a` carry no license obligation
  beyond Apache-2.0's own terms.
- Apache-2.0 includes an express patent grant and a trademark
  limitation; read `LICENSE` for the operative text.

`LICENSE` is the verbatim Apache License 2.0. `NOTICE` carries the
attribution notice that Apache-2.0 section 4(d) propagates to
redistributions.

## The one exception

`DCO` is the Developer Certificate of Origin 1.1, © 2004, 2006 The
Linux Foundation, reproduced verbatim under its own terms
(`LICENSES/LicenseRef-DCO-1.1.txt`). It is a contributor
certification, not a license covering wayruntime's code.

## Third-party code

None is vendored. The tree is 100% first-party C11; what the binaries
link against is inventoried in `THIRD-PARTY-NOTICES.md`. New
third-party imports require maintainer approval, an entry in that
file, and the full license text under `LICENSES/` — see
`CONTRIBUTING.md`.

## Contributions

Contributions are accepted for the whole repository under Apache-2.0
with DCO 1.1 sign-off. There is no CLA. See `CONTRIBUTING.md`.

## Previous licensing

wayruntime 0.1.0 was first published with its core under Business
Source License 1.1 and only its public API surface under Apache-2.0.
The repository was relicensed in full to Apache-2.0 on 2026-09-02 by
the copyright holder. The BUSL-1.1 terms, the Additional Use Grant,
and the 2030-08-30 Change Date no longer apply to anything in this
repository; Apache-2.0 grants strictly more, so no permission granted
under the earlier terms is withdrawn.

## Separate distributions by the copyright holder

Owner-authored portions may also appear in other WayOS Project
distributions under other terms. A separate distribution does not
change the terms of this repository.
