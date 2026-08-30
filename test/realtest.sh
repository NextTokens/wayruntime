# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 WayOS Project
#
# realtest.sh — end-to-end against a REAL GGUF model, no mocks.
# Always invoked as `bash test/realtest.sh` (no shebang by design: the
# SPDX header owns line 1).
#
#   bash test/realtest.sh <builddir> <model.gguf>
#   (usually: make realtest MODEL_GGUF=path/to/model.gguf)
#
# Asserts: model loads and reports sane metadata, deterministic greedy
# generation produces non-empty output twice identically, and the SIMD
# counters show fast-path activity.
set -euo pipefail

BUILDDIR="${1:?usage: realtest.sh <builddir> <model.gguf>}"
MODEL="${2:-}"

if [ -z "$MODEL" ]; then
    echo "realtest: pass MODEL_GGUF=path/to/model.gguf" >&2
    exit 2
fi
if [ ! -f "$MODEL" ]; then
    echo "realtest: model not found: $MODEL" >&2
    exit 2
fi

WAYRT="$BUILDDIR/wayrt"
[ -x "$WAYRT" ] || WAYRT="$BUILDDIR/wayrt.exe"
if [ ! -x "$WAYRT" ]; then
    echo "realtest: $BUILDDIR/wayrt not built (run make first)" >&2
    exit 2
fi

fail() { echo "realtest FAIL: $*" >&2; exit 1; }

echo "== load/verify =="
INFO="$("$WAYRT" verify "$MODEL")" || fail "verify returned nonzero"
echo "$INFO"
echo "$INFO" | grep -qi 'arch' || fail "verify printed no architecture"

echo "== deterministic greedy generation (2 runs) =="
PROMPT="The capital of France is"
OUT1="$("$WAYRT" generate --greedy --max-tokens 16 --prompt "$PROMPT" "$MODEL")" \
    || fail "generate run 1 returned nonzero"
OUT2="$("$WAYRT" generate --greedy --max-tokens 16 --prompt "$PROMPT" "$MODEL")" \
    || fail "generate run 2 returned nonzero"
[ -n "$OUT1" ] || fail "empty generation"
[ "$OUT1" = "$OUT2" ] || fail "greedy generation is not deterministic"

echo "== counters =="
BENCH="$("$WAYRT" bench --tokens 4 "$MODEL")" \
    || fail "bench returned nonzero"
printf '%s\n' "$BENCH"

counter_value() {
    printf '%s\n' "$BENCH" | awk -v name="$1" '
        $1 == name {
            count++
            value = $2
            sub(/\r$/, "", value)
            if (NF != 2 || value !~ /^[0-9]+$/)
                invalid = 1
        }
        END {
            if (count != 1 || invalid)
                exit 1
            print value
        }
    '
}

SIMD_HITS="$(counter_value matmul_ggml_simd)" \
    || fail "bench missing a unique numeric matmul_ggml_simd counter"
QUANT_HITS="$(counter_value matmul_ggml_quant)" \
    || fail "bench missing a unique numeric matmul_ggml_quant counter"
PERELEM_HITS="$(counter_value matmul_perelem)" \
    || fail "bench missing a unique numeric matmul_perelem counter"

case "$SIMD_HITS$QUANT_HITS" in
    *[1-9]*) ;;
    *) fail "no optimized GGML matmul activity (simd=$SIMD_HITS quant=$QUANT_HITS)" ;;
esac
case "$PERELEM_HITS" in
    *[1-9]*) fail "per-element matmul fallback used ($PERELEM_HITS hit(s))" ;;
esac

echo "realtest PASS"
