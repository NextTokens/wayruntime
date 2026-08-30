# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 WayOS Project
#
# usecase_grammar.sh -- USE CASE test: "schema-shaped output from a
# local model, guaranteed by construction" (docs/USECASE.md).
# Always invoked as `bash test/usecase_grammar.sh` (no shebang by
# design: the SPDX header owns line 1).
#
#   bash test/usecase_grammar.sh [builddir] [model.gguf]
#     builddir    default: build/posix
#     model.gguf  default: $MODELS_DIR/Qwen3-0.6B-Q4_K_M.gguf
#   MODELS_DIR=/path/to/models overrides the model directory
#   (default: ../models next to the repository, as in integration.sh).
#   Usually invoked as `make usecase [MODEL_GGUF=path]`.
#
# Executes the exact scenario documented in docs/USECASE.md against a
# REAL model (no mocks) and asserts every claim the document makes:
#
#   1. examples/grammar_mask.c compiles warning-free -- full warning
#      set plus -Werror, public header + static library only;
#   2. the constrained run exits 0 and stderr reports the grammar
#      reached its accept state ("grammar complete");
#   3. its stdout parses as JSON via python3 json.load -- THE claim;
#   4. the parsed value is a flat object of >= 2 string:string pairs
#      (the example demands >= 2 pairs through the mask alone);
#   5. greedy + masked decoding is deterministic: a second run is
#      byte-identical on stdout;
#   6. the temp workspace is removed.
#
# Between 5 and 6 an INFORMATIONAL contrast (never scored): the same
# prompt, same model, unconstrained through `wayrt generate --raw
# --greedy`, piped into the same JSON parse -- printed as PARSES or
# DOES-NOT-PARSE so the transcript shows what the mask buys.
#
# Output: one numbered line per step, summary "N/N passed", exit
# nonzero on any failure.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

BUILDDIR="${1:-build/posix}"
MODELS_DIR="${MODELS_DIR:-$(cd "$ROOT/.." && pwd)/models}"
MODEL="${2:-}"
[ -n "$MODEL" ] || MODEL="$MODELS_DIR/Qwen3-0.6B-Q4_K_M.gguf"

LIB="$BUILDDIR/libwayruntime.a"
if [ ! -f "$LIB" ]; then
    echo "usecase: $LIB not built (run make first)" >&2
    exit 2
fi
WAYRT="$BUILDDIR/wayrt"
[ -x "$WAYRT" ] || WAYRT="$BUILDDIR/wayrt.exe"
if [ ! -x "$WAYRT" ]; then
    echo "usecase: $BUILDDIR/wayrt not built (run make first)" >&2
    exit 2
fi
if [ ! -f "$MODEL" ]; then
    echo "usecase: model missing: $MODEL" >&2
    echo "usecase: pass a model path or set MODELS_DIR" >&2
    exit 2
fi
command -v python3 >/dev/null 2>&1 || {
    echo "usecase: python3 required (JSON-parse assertions)" >&2
    exit 2
}

T="$(mktemp -d /tmp/wayrt-usecase-XXXXXX)"
trap 'rm -rf "$T"' EXIT

PASS=0
FAIL=0
STEP=0
DESC=""

say()  { printf '%s\n' "$*"; }
step() { STEP=$((STEP+1)); DESC="$1"; }
ok()   { PASS=$((PASS+1)); printf '%2d. ok   %s\n' "$STEP" "$DESC"; }
bad()  {
    FAIL=$((FAIL+1))
    printf '%2d. FAIL %s%s\n' "$STEP" "$DESC" "${1:+ [$1]}"
    sed 's/^/      /' "$T/out" "$T/err" 2>/dev/null | head -8
}

# run <cmd...>: capture stdout/stderr/exit code without aborting.
RC=0
run() { "$@" >"$T/out" 2>"$T/err"; RC=$?; }

CC="${CC:-gcc}"

# The exact prompt baked into examples/grammar_mask.c as its default,
# passed explicitly so the constrained run and the unconstrained
# contrast run see the identical string (trailing newline included).
PROMPT='Here is a small JSON object with string values describing a cat:
'

say "usecase: grammar-masked JSON on $(basename "$MODEL")"

# ------------------------------------------------------------------
# 1. The example compiles warning-free against the SDK surface only:
#    public header + static library, full warning set, -Werror, and
#    an empty compiler stderr (so even non-fatal notes would fail).
# ------------------------------------------------------------------

step "grammar_mask.c compiles warning-free (-Wall -Wextra -Wshadow -Wvla -Werror)"
run $CC -std=c11 -O2 -Wall -Wextra -Wshadow -Wvla -Werror -Iinclude \
    -o "$T/grammar_mask" examples/grammar_mask.c "$LIB" -lpthread -lm
if [ "$RC" -ne 0 ]; then bad "compile failed"
elif [ -s "$T/err" ]; then bad "compiler emitted diagnostics"
else ok; fi
[ -x "$T/grammar_mask" ] || { say "usecase: cannot continue"; exit 1; }

# ------------------------------------------------------------------
# 2. The constrained run succeeds and the automaton reaches its
#    accept state -- an incomplete object exits nonzero by design.
# ------------------------------------------------------------------

step "constrained run exits 0 and reports \"grammar complete\""
run "$T/grammar_mask" "$MODEL" "$PROMPT"
cp "$T/out" "$T/g1"
cp "$T/err" "$T/e1"
if [ "$RC" -ne 0 ]; then bad "exit $RC, wanted 0"
elif ! grep -q 'grammar complete' "$T/e1"; then bad "no 'grammar complete'"
else ok; fi

# ------------------------------------------------------------------
# 3. THE core claim: stdout is machine-consumable as-is.  json.load
#    is the strict reference parser -- no trimming, no repair.
# ------------------------------------------------------------------

step "constrained stdout parses as JSON (python3 json.load, verbatim)"
run python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$T/g1"
if [ "$RC" -ne 0 ]; then bad "json.load rejected the output"; else ok; fi

# ------------------------------------------------------------------
# 4. Shape: a flat object of only string:string pairs, at least two
#    of them (gm_require_pairs(&g, 2) enforces the minimum via the
#    mask -- '}' and EOS are simply illegal before pair two).
# ------------------------------------------------------------------

step "parsed object is flat with >= 2 string:string pairs"
run python3 -c '
import json, sys
o = json.load(open(sys.argv[1]))
assert isinstance(o, dict), "top level is not an object"
assert len(o) >= 2, "fewer than 2 pairs: %d" % len(o)
assert all(isinstance(k, str) and isinstance(v, str)
           for k, v in o.items()), "non-string member"
print("%d string:string pairs" % len(o))
' "$T/g1"
if [ "$RC" -ne 0 ]; then bad; else ok; fi

# ------------------------------------------------------------------
# 5. Determinism: greedy sampling + a pure mask = a reproducible
#    artifact.  Two runs must agree to the byte.
# ------------------------------------------------------------------

step "deterministic: second constrained run byte-identical on stdout"
run "$T/grammar_mask" "$MODEL" "$PROMPT"
if [ "$RC" -ne 0 ]; then bad "exit $RC"
elif ! cmp -s "$T/g1" "$T/out"; then bad "outputs differ"
else ok; fi

# ------------------------------------------------------------------
# Contrast (INFORMATIONAL, never scored): the same prompt and model
# without the mask.  Whatever happens is printed, not asserted --
# an unmasked model is free to emit valid JSON by luck, prose, or
# half an object; the point is that nothing guarantees it.
# ------------------------------------------------------------------

say "-- contrast (informational, not scored): same prompt, no mask --"
run "$WAYRT" generate --raw --greedy --max-tokens 96 \
    --prompt "$PROMPT" "$MODEL"
cp "$T/out" "$T/u1"
say "   unconstrained output (wayrt generate --raw --greedy, exit $RC):"
sed 's/^/   | /' "$T/u1"
if python3 -c 'import json, sys; json.load(open(sys.argv[1]))' \
    "$T/u1" >/dev/null 2>&1
then say "   unconstrained output: PARSES as JSON (this run got lucky)"
else say "   unconstrained output: DOES-NOT-PARSE as JSON"
fi

# ------------------------------------------------------------------
# 6. Cleanup: the workspace and every temp file in it are gone.
# ------------------------------------------------------------------

step "temp workspace removed"
rm -rf "$T"
if [ -e "$T" ]; then bad "still present"; else ok; fi
trap - EXIT

# ------------------------------------------------------------------

TOTAL=$((PASS + FAIL))
say ""
say "usecase: $PASS/$TOTAL passed"
[ "$FAIL" -eq 0 ]
