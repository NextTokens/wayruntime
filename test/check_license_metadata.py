# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 WayOS Project
"""Validate the repository's declared licensing without Git.

wayruntime is Apache-2.0 throughout. The invariants enforced here:

  * every file carries an Apache-2.0 SPDX identifier or is mapped to one
    by REUSE.toml (the DCO is the single documented exception);
  * no file declares any other license identifier;
  * no code or test file mentions the superseded source-available terms
    at all, and only the two documents that record the relicensing
    history may mention them in prose;
  * LICENSE is the verbatim Apache-2.0 text, DCO matches its LicenseRef
    copy, and NOTICE names the copyright holder the sources declare.
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
}

# The repository is single-licensed; DCO is a contributor certification
# reproduced verbatim under its own terms, not a license over this code.
PROJECT_LICENSE = "Apache-2.0"
ALLOWED_LICENSES = {PROJECT_LICENSE, "LicenseRef-DCO-1.1"}
EXPECTED_LICENSE_FILES = {
    "Apache-2.0": ROOT / "LICENSES/Apache-2.0.txt",
    "LicenseRef-DCO-1.1": ROOT / "LICENSES/LicenseRef-DCO-1.1.txt",
}

# wayruntime 0.1.0 shipped a source-available core before the 2026-09-02
# relicensing.  These two documents record that history on purpose;
# anywhere else, a surviving mention means a file was missed.  The
# needles are split so this checker never matches itself, the same way
# the SPDX tags below are.
SUPERSEDED_TERMS = ("BU" "SL", "Business Source" " License")
HISTORY_DOCS = {"LICENSING.md", "README.md"}

SPDX_TAG = "SPDX-License-" "Identifier:"
SPDX_RE = re.compile(
    r"(?m)^[^\r\n]*" + re.escape(SPDX_TAG) + r"\s*"
    r"([A-Za-z0-9][A-Za-z0-9.+-]*)"
    r"(?=\s*(?:\*/|-->|$))"
)
COPYRIGHT_TAG = "SPDX-FileCopyright" "Text:"


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
    holders: set[str] = set()

    for annotation in annotations:
        if annotation.get("precedence") not in {"aggregate", "closest", "override"}:
            failures.append("REUSE.toml: invalid annotation precedence")
        license_id = annotation.get("SPDX-License-Identifier")
        copyright_text = annotation.get("SPDX-FileCopyrightText")
        if not license_id or not copyright_text:
            failures.append("REUSE.toml: every annotation needs license and copyright")
            continue
        used_licenses.add(license_id)
        if license_id == PROJECT_LICENSE:
            holders.add(copyright_text)

    for path in repository_files():
        relative = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"{relative}: unexpected binary file outside excluded paths")
            continue

        inline = SPDX_RE.findall(text)
        used_licenses.update(inline)

        # Single-license repository: nothing may declare anything else.
        for license_id in inline:
            if license_id != PROJECT_LICENSE:
                failures.append(
                    f"{relative}: declares {license_id}, expected {PROJECT_LICENSE}"
                )

        # Every source and test file states it inline, not just by mapping.
        if relative.split("/")[0] in {"src", "include", "examples"} \
                and PROJECT_LICENSE not in inline:
            failures.append(f"{relative}: must declare {PROJECT_LICENSE} inline")

        # No surviving residue of the old terms outside the history docs.
        if relative not in HISTORY_DOCS:
            for term in SUPERSEDED_TERMS:
                if term in text:
                    failures.append(
                        f"{relative}: mentions superseded {term} terms "
                        f"(relicensed to {PROJECT_LICENSE})"
                    )
                    break

        mapped = [a for a in annotations if path_matches(relative, a.get("path", []))]
        if not inline and not mapped:
            failures.append(f"{relative}: no SPDX identifier or REUSE mapping")

    for license_id in sorted(used_licenses):
        if license_id not in ALLOWED_LICENSES:
            failures.append(f"{license_id}: not an allowed license for this repository")
            continue
        license_file = EXPECTED_LICENSE_FILES[license_id]
        if not license_file.is_file():
            failures.append(f"{license_id}: missing {license_file.relative_to(ROOT)}")

    # LICENSE is the operative text and must be the canonical Apache-2.0.
    apache = (ROOT / "LICENSES/Apache-2.0.txt").read_text(encoding="utf-8")
    if "APPENDIX: How to apply the Apache License to your work." not in apache:
        failures.append("LICENSES/Apache-2.0.txt: canonical appendix is missing")
    if (ROOT / "LICENSE").read_text(encoding="utf-8") != apache:
        failures.append("LICENSE: must be the verbatim LICENSES/Apache-2.0.txt text")

    dco = (ROOT / "DCO").read_text(encoding="utf-8").strip()
    dco_license = (ROOT / "LICENSES/LicenseRef-DCO-1.1.txt").read_text(
        encoding="utf-8"
    ).strip()
    if dco != dco_license:
        failures.append("DCO and its LicenseRef text differ")

    # The copyright holder is read from REUSE.toml (never duplicated here)
    # and must be the name NOTICE and the source headers actually use.
    if len(holders) != 1:
        failures.append(f"REUSE.toml: expected one {PROJECT_LICENSE} copyright holder")
    else:
        holder = next(iter(holders))
        name = holder.split(" ", 1)[1] if holder[0].isdigit() else holder
        notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
        if name not in notice:
            failures.append(f"NOTICE: does not name the copyright holder {name!r}")
        engine = (ROOT / "src/core/engine.c").read_text(encoding="utf-8")
        if f"{COPYRIGHT_TAG} {holder}" not in engine:
            failures.append(f"src/core/engine.c: copyright header is not {holder!r}")

    if failures:
        raise SystemExit("License metadata failed:\n  " + "\n  ".join(failures))

    print(f"License metadata: {len(repository_files())} files covered; "
          f"{PROJECT_LICENSE} throughout")


if __name__ == "__main__":
    main()
