# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 WayOS Project
"""Validate the repository's declared licensing boundary without Git.

Layout under check:
  Apache-2.0  include/wayruntime/**, examples/**, Makefile, test scripts,
              repo metadata files (mapped in REUSE.toml)
  BUSL-1.1    everything under src/, plus the C test sources that include
              internal headers
"""

from __future__ import annotations

import fnmatch
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", "build", "LICENSES"}
EXCLUDED_FILES = {
    ".env",
    "COPYING.MinGW-w64-runtime.txt",
    "LICENSE",
    "REUSE.toml",
    "TRY-IT.md",
    # Internal porting notes for the initial port; gitignored, not shipped.
    "docs/PORTING-CONTRACTS.md",
}
SPDX_TAG = "SPDX-License-" "Identifier:"
SPDX_RE = re.compile(
    r"(?m)^[^\r\n]*" + re.escape(SPDX_TAG) + r"\s*"
    r"([A-Za-z0-9][A-Za-z0-9.+-]*)"
    r"(?=\s*(?:\*/|-->|$))"
)
EXPECTED_LICENSE_FILES = {
    "Apache-2.0": ROOT / "LICENSES/Apache-2.0.txt",
    "BUSL-1.1": ROOT / "LICENSES/BUSL-1.1.txt",
    "LicenseRef-DCO-1.1": ROOT / "LICENSES/LicenseRef-DCO-1.1.txt",
}


def repository_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if relative.as_posix() in EXCLUDED_FILES:
            continue
        files.append(path)
    return sorted(files)


def path_matches(relative: str, declared: str | list[str]) -> bool:
    patterns = [declared] if isinstance(declared, str) else declared
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns)


def main() -> None:
    reuse = tomllib.loads((ROOT / "REUSE.toml").read_text(encoding="utf-8"))
    if reuse.get("version") != 1:
        raise SystemExit("license metadata: REUSE.toml must use version 1")

    annotations = reuse.get("annotations", [])
    failures: list[str] = []
    used_licenses: set[str] = set()

    for annotation in annotations:
        if annotation.get("precedence") not in {"aggregate", "closest", "override"}:
            failures.append("REUSE.toml: invalid annotation precedence")
        license_id = annotation.get("SPDX-License-Identifier")
        copyright_text = annotation.get("SPDX-FileCopyrightText")
        if not license_id or not copyright_text:
            failures.append("REUSE.toml: every annotation needs license and copyright")
        else:
            used_licenses.add(license_id)

    for path in repository_files():
        relative = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"{relative}: unexpected binary file outside excluded paths")
            continue

        inline = SPDX_RE.findall(text)
        used_licenses.update(inline)

        # The licensing line the repo promises: src/ is BUSL, the public
        # headers and examples are Apache.  Enforce it, not just presence.
        if relative.startswith("src/") and "BUSL-1.1" not in inline:
            failures.append(f"{relative}: src/ files must declare BUSL-1.1")
        if (relative.startswith("include/") or relative.startswith("examples/")) \
                and "Apache-2.0" not in inline:
            failures.append(f"{relative}: include/ and examples/ must declare Apache-2.0")

        mapped = [a for a in annotations if path_matches(relative, a.get("path", []))]
        if not inline and not mapped:
            failures.append(f"{relative}: no SPDX identifier or REUSE mapping")

    for license_id in sorted(used_licenses):
        license_file = EXPECTED_LICENSE_FILES.get(license_id)
        if license_file is None:
            failures.append(f"{license_id}: no expected canonical license-file mapping")
        elif not license_file.is_file():
            failures.append(f"{license_id}: missing {license_file.relative_to(ROOT)}")

    apache = (ROOT / "LICENSES/Apache-2.0.txt").read_text(encoding="utf-8")
    if "APPENDIX: How to apply the Apache License to your work." not in apache:
        failures.append("LICENSES/Apache-2.0.txt: canonical appendix is missing")

    dco = (ROOT / "DCO").read_text(encoding="utf-8").strip()
    dco_license = (ROOT / "LICENSES/LicenseRef-DCO-1.1.txt").read_text(
        encoding="utf-8"
    ).strip()
    if dco != dco_license:
        failures.append("DCO and its LicenseRef text differ")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    for required in ("Licensor:             WayOS Project",
                     "Licensed Work:        The wayruntime 0.1.0 files",
                     "Change Date:          2030-08-30"):
        if required not in license_text:
            failures.append(f"LICENSE: missing {required.strip()!r}")

    if failures:
        raise SystemExit("License metadata failed:\n  " + "\n  ".join(failures))

    print(f"License metadata: {len(repository_files())} files covered; "
          f"{len(used_licenses)} license identifiers resolved")


if __name__ == "__main__":
    main()
