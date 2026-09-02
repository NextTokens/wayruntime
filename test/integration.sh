# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 WayOS Project
#
# integration.sh -- CLI + hostile-input + SDK integration suite (POSIX).
# Always invoked as `bash test/integration.sh` (no shebang by design:
# the SPDX header owns line 1).
#
#   bash test/integration.sh [builddir]        default: build/posix
#   WR_TEST_LLAMA=/path/to/model.gguf  the llama-family model the suite
#       runs on (REQUIRED; default $MODELS_DIR/stories15M-q8_0.gguf,
#       MODELS_DIR default ../models next to the repository).  Any small
#       llama-architecture GGUF works; docs/TESTING.md lists the files
#       the suite was validated with and where they come from.
#   WR_TEST_Q4K / WR_TEST_Q5K / WR_TEST_Q6K  optional K-quant models for
#       the extra load checks (default $MODELS_DIR/tiny-q?k-test.gguf);
#       an absent optional file prints a SKIP line, never a silent pass.
#
# Real models, no mocks.  Layers:
#   1. `wayrt verify` accepts the test model (and the optional K-quant
#      files) and refuses a synthesized unsupported-architecture file
#      with exit 2.
#   2. HOSTILE-GGUF suite: eight structurally corrupt files generated
#      on the fly; every one must be refused with exit 2 and must
#      never crash.
#   3. Generation invariants: greedy determinism, seeded
#      reproducibility, exact --max-tokens accounting, bench counters,
#      threads/SIMD numeric-path invariance, chat-template fallback.
#   4. SDK harness (test/sdk_gen.c against libwayruntime.a): session
#      continuation, batch-vs-serial bit-exactness, honest
#      WR_ERR_CTX_FULL, logits view rules, sampler determinism, mask
#      callback, and sampler statistics; its pass/fail counts fold
#      into this suite's totals.
#   5. Concurrency: eight threads decoding their own sessions of the
#      one model, plus a batched step running alongside a single step,
#      must reproduce the sequential logits byte for byte.
#
# Output: one numbered line per step, summary "N/N passed", exit
# nonzero on any failure.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

BUILDDIR="${1:-build/posix}"
MODELS_DIR="${MODELS_DIR:-$(cd "$ROOT/.." && pwd)/models}"

WAYRT="$BUILDDIR/wayrt"
[ -x "$WAYRT" ] || WAYRT="$BUILDDIR/wayrt.exe"
if [ ! -x "$WAYRT" ]; then
    echo "integration: $BUILDDIR/wayrt not built (run make first)" >&2
    exit 2
fi
LIB="$BUILDDIR/libwayruntime.a"
if [ ! -f "$LIB" ]; then
    echo "integration: $LIB not built (run make first)" >&2
    exit 2
fi

LLAMA="${WR_TEST_LLAMA:-$MODELS_DIR/stories15M-q8_0.gguf}"
Q4K="${WR_TEST_Q4K:-$MODELS_DIR/tiny-q4k-test.gguf}"
Q5K="${WR_TEST_Q5K:-$MODELS_DIR/tiny-q5k-test.gguf}"
Q6K="${WR_TEST_Q6K:-$MODELS_DIR/tiny-q6k-test.gguf}"
if [ ! -f "$LLAMA" ]; then
    echo "integration: llama-family test model missing: $LLAMA" >&2
    echo "integration: set WR_TEST_LLAMA=/path/to/model.gguf (any small" >&2
    echo "integration: llama-architecture GGUF; docs/TESTING.md names the" >&2
    echo "integration: validated files and their provenance)" >&2
    exit 2
fi

T="$(mktemp -d /tmp/wayrt-int-XXXXXX)"
trap 'rm -rf "$T"' EXIT

PASS=0
FAIL=0
SKIP=0
STEP=0
DESC=""

say()  { printf '%s\n' "$*"; }
step() { STEP=$((STEP+1)); DESC="$1"; }
ok()   { PASS=$((PASS+1)); printf '%2d. ok   %s\n' "$STEP" "$DESC"; }
# Optional-input steps only: an absent optional model is reported, never
# counted as a pass.
skip() { SKIP=$((SKIP+1)); printf '%2d. SKIP %s [%s]\n' "$STEP" "$DESC" "$1"; }
bad()  {
    FAIL=$((FAIL+1))
    printf '%2d. FAIL %s%s\n' "$STEP" "$DESC" "${1:+ [$1]}"
    sed 's/^/      /' "$T/out" "$T/err" 2>/dev/null | head -8
}

# run <cmd...>: capture stdout/stderr/exit code without aborting.
RC=0
run() { "$@" >"$T/out" 2>"$T/err"; RC=$?; }

# ------------------------------------------------------------------
# 1. verify: the test model loads (arch line present); the optional
#    K-quant files exercise the Q4_K / Q5_K / Q6_K load paths.
# ------------------------------------------------------------------

step "verify $(basename "$LLAMA") (exit 0, arch line)"
run "$WAYRT" verify "$LLAMA"
if [ "$RC" -ne 0 ]; then bad "exit $RC, wanted 0"
elif ! grep -q '^arch:' "$T/out"; then bad "no arch line"
else ok; fi

for kq in "Q4_K:$Q4K" "Q5_K:$Q5K" "Q6_K:$Q6K"; do
    kind="${kq%%:*}"; m="${kq#*:}"
    step "verify optional $kind model $(basename "$m") (exit 0, arch line)"
    if [ ! -f "$m" ]; then skip "optional model not present"; continue; fi
    run "$WAYRT" verify "$m"
    if [ "$RC" -ne 0 ]; then bad "exit $RC, wanted 0"
    elif ! grep -q '^arch:' "$T/out"; then bad "no arch line"
    else ok; fi
done

step "default model load maps read-only or reports streamed fallback"
run "$WAYRT" verify "$LLAMA"
cp "$T/out" "$T/verify-mmap"
if [ "$RC" -ne 0 ]; then bad "exit $RC"
elif grep -q 'loader: .* mmap$' "$T/err"; then ok
elif grep -q 'loader: mapping .* failed .* streaming instead' "$T/err"; then ok
else bad "loader reported neither mmap success nor streamed fallback"; fi

step "--no-mmap streams weights with identical model metadata"
run "$WAYRT" verify --no-mmap "$LLAMA"
if [ "$RC" -ne 0 ]; then bad "exit $RC"
elif grep -q 'loader: .* mmap$' "$T/err"; then
    bad "loader still reported mmap"
elif grep -q 'loader: mapping .* failed' "$T/err"; then
    bad "loader attempted mmap despite explicit opt-out"
elif ! cmp -s "$T/verify-mmap" "$T/out"; then
    bad "streamed load changed verify output"
else ok; fi

# ------------------------------------------------------------------
# 2. refusals: missing file, garbage file (the unsupported-architecture
#    refusal follows the generator in section 3)
# ------------------------------------------------------------------

step "nonexistent file refused (exit 2)"
run "$WAYRT" verify "$T/does-not-exist.gguf"
if [ "$RC" -ne 2 ]; then bad "exit $RC, wanted 2"; else ok; fi

step "malformed file refused (exit 2)"
echo "garbage, not a gguf at all" > "$T/garbage.gguf"
run "$WAYRT" verify "$T/garbage.gguf"
if [ "$RC" -ne 2 ]; then bad "exit $RC, wanted 2"; else ok; fi

# ------------------------------------------------------------------
# 3. HOSTILE-GGUF suite: structurally corrupt files.  Every case must
#    be REFUSED with exit 2 and must never crash (a segfault surfaces
#    as exit 139, caught by the exact-2 assertion; the grep guards
#    against a caught-and-printed crash).
# ------------------------------------------------------------------

python3 - "$T" <<'PYEOF'
import os, struct, sys

out = sys.argv[1]

def u32(v): return struct.pack("<I", v)
def u64(v): return struct.pack("<Q", v)
def s(txt):
    b = txt.encode()
    return u64(len(b)) + b
def kv_str(k, v): return s(k) + u32(8) + s(v)   # value type 8 = string
def kv_u32(k, v): return s(k) + u32(4) + u32(v) # value type 4 = uint32

MAGIC = b"GGUF"
ARCH = kv_str("general.architecture", "llama")

def header(version, n_tensors, kvs):
    blob = MAGIC + u32(version) + u64(n_tensors) + u64(len(kvs))
    for kv in kvs:
        blob += kv
    return blob

def tensor(name, dims, ttype, offset):
    b = s(name) + u32(len(dims))
    for d in dims:
        b += u64(d)
    return b + u32(ttype) + u64(offset)

def write(name, data):
    with open(os.path.join(out, name), "wb") as f:
        f.write(data)

# 1: not a GGUF file at all (wrong magic, otherwise plausible header)
write("h1-bad-magic.gguf", b"FUGG" + u32(3) + u64(0) + u64(0))
# 2: header version 1 (pre-v2 layout, unsupported by contract)
write("h2-version-1.gguf", MAGIC + u32(1) + u64(0) + u64(0))
# 3: header version 99 (future/unknown)
write("h3-version-99.gguf", MAGIC + u32(99) + u64(0) + u64(0))
# 4: tensor data range far past EOF (offset 2^40 in a <1KB file)
write("h4-offset-past-eof.gguf",
      header(3, 1, [ARCH]) + tensor("t0", [4], 0, 1 << 40) + b"\0" * 64)
# 5: two tensors with overlapping byte ranges
#    (F32[16] = 64 bytes each, at data offsets 0 and 32; the padding
#    keeps both ranges inside the file so only the overlap can trip)
write("h5-overlap.gguf",
      header(3, 2, [ARCH]) + tensor("t0", [16], 0, 0)
      + tensor("t1", [16], 0, 32) + b"\0" * 256)
# 6: general.alignment = 24 (not a power of two)
write("h6-bad-alignment.gguf",
      header(3, 0, [ARCH, kv_u32("general.alignment", 24)]) + b"\0" * 64)
# 7: kv key string length 0xFFFFFFFFFFFFFFFF
write("h7-huge-string.gguf",
      MAGIC + u32(3) + u64(0) + u64(1)
      + u64(0xFFFFFFFFFFFFFFFF) + b"A" * 32)
# 8: tensor count 100000 (over the compiled hard limit)
write("h8-tensor-count.gguf",
      MAGIC + u32(3) + u64(100000) + u64(0) + b"\0" * 64)
# well-formed but unsupported architecture: one F32 tensor inside the
# file so every structural check passes and only the arch gate trips
write("unsupported-arch.gguf",
      header(3, 1, [kv_str("general.architecture", "gpt2")])
      + tensor("t0", [16], 0, 0) + b"\0" * 128)
PYEOF
if [ $? -ne 0 ]; then
    echo "integration: hostile-GGUF generator failed (python3 required)" >&2
    exit 2
fi

for h in \
    "h1-bad-magic.gguf:bad magic" \
    "h2-version-1.gguf:header version 1" \
    "h3-version-99.gguf:header version 99" \
    "h4-offset-past-eof.gguf:tensor offset past EOF" \
    "h5-overlap.gguf:overlapping tensors" \
    "h6-bad-alignment.gguf:non-pow2 alignment" \
    "h7-huge-string.gguf:string length 0xFFFFFFFFFFFFFFFF" \
    "h8-tensor-count.gguf:tensor count 100000"
do
    file="${h%%:*}"
    desc="${h#*:}"
    step "hostile gguf refused: $desc (exit 2, no crash)"
    run "$WAYRT" verify "$T/$file"
    if [ "$RC" -ne 2 ]; then bad "exit $RC, wanted 2"
    elif grep -qi 'segmentation' "$T/out" "$T/err"; then bad "crash text"
    else ok; fi
done

step "unsupported architecture refused (exit exactly 2, stderr names gpt2)"
run "$WAYRT" verify "$T/unsupported-arch.gguf"
if [ "$RC" -ne 2 ]; then bad "exit $RC, wanted 2"
elif ! grep -q 'gpt2' "$T/err"; then bad "stderr does not name gpt2"
else ok; fi

# ------------------------------------------------------------------
# 4. generation invariants on the test model
# ------------------------------------------------------------------

PROMPT="Once upon a time"

step "generate --raw --greedy deterministic (2 runs byte-identical)"
run "$WAYRT" generate --raw --greedy --max-tokens 24 \
    --prompt "$PROMPT" "$LLAMA"
cp "$T/out" "$T/g1"; RC1=$RC
run "$WAYRT" generate --raw --greedy --max-tokens 24 \
    --prompt "$PROMPT" "$LLAMA"
if [ "$RC1" -ne 0 ] || [ "$RC" -ne 0 ]; then bad "exit $RC1/$RC"
elif [ ! -s "$T/g1" ]; then bad "empty generation"
elif ! cmp -s "$T/g1" "$T/out"; then bad "outputs differ"
else ok; fi

step "--seed 42 reproducible at --temp 0.8 (2 runs identical)"
run "$WAYRT" generate --raw --temp 0.8 --seed 42 --max-tokens 16 \
    --prompt "$PROMPT" "$LLAMA"
cp "$T/out" "$T/s42"; RC1=$RC
run "$WAYRT" generate --raw --temp 0.8 --seed 42 --max-tokens 16 \
    --prompt "$PROMPT" "$LLAMA"
if [ "$RC1" -ne 0 ] || [ "$RC" -ne 0 ]; then bad "exit $RC1/$RC"
elif ! cmp -s "$T/s42" "$T/out"; then bad "same seed diverged"
else ok; fi

# THEORETICAL FLAKE: seeds 42 and 43 are independent RNG streams, so a
# 16-token collision is possible in principle; over stories15M's ~32000
# token distribution the probability is far below 2^-60.  If this ever
# fires, re-run once; a repeat is a real sampler/RNG defect.
step "--seed 43 differs from --seed 42 at --temp 0.8"
run "$WAYRT" generate --raw --temp 0.8 --seed 43 --max-tokens 16 \
    --prompt "$PROMPT" "$LLAMA"
if [ "$RC" -ne 0 ]; then bad "exit $RC"
elif cmp -s "$T/s42" "$T/out"; then bad "different seeds produced identical output"
else ok; fi

step "--max-tokens honored exactly (stderr reports 7 generated)"
run "$WAYRT" generate --raw --greedy --max-tokens 7 \
    --prompt "$PROMPT" "$LLAMA"
NGEN="$(sed -n 's/.*[^0-9]\([0-9][0-9]*\) generated.*/\1/p' "$T/err")"
if [ "$RC" -ne 0 ]; then bad "exit $RC"
elif [ "$NGEN" != "7" ]; then bad "stderr reports '$NGEN' generated, wanted 7"
elif ! grep -q 'stop=max_tokens' "$T/err"; then bad "stop reason not max_tokens"
else ok; fi

step "bench --tokens 8 (exit 0, 12 counters, nonzero decode tok/s)"
run "$WAYRT" bench --tokens 8 "$LLAMA"
NCTR="$(awk '/^counters:/{f=1;next} f && /^  /{n++} END{print n+0}' "$T/out")"
TOKS="$(sed -n 's/^decode:.* \([0-9][0-9.]*\) tok\/s.*/\1/p' "$T/out")"
if [ "$RC" -ne 0 ]; then bad "exit $RC"
elif [ "$NCTR" != "12" ]; then bad "$NCTR counters printed, wanted 12"
elif [ -z "$TOKS" ]; then bad "no decode tok/s line"
elif ! awk "BEGIN{exit !($TOKS > 0)}"; then bad "decode rate $TOKS tok/s"
else ok; fi

step "--threads 1 matches default threads (greedy bit-exactness)"
run "$WAYRT" --threads 1 generate --raw --greedy --max-tokens 24 \
    --prompt "$PROMPT" "$LLAMA"
if [ "$RC" -ne 0 ]; then bad "exit $RC"
elif ! cmp -s "$T/g1" "$T/out"; then
    bad "REAL FINDING: thread-count changed greedy output — the parallel reduction order is leaking into the numerics; investigate the worker-pool split, do not delete this test"
else ok; fi

step "--simd scalar matches --simd auto (greedy bit-exactness)"
run "$WAYRT" --simd scalar generate --raw --greedy --max-tokens 24 \
    --prompt "$PROMPT" "$LLAMA"
if [ "$RC" -ne 0 ]; then bad "exit $RC"
elif ! cmp -s "$T/g1" "$T/out"; then
    bad "REAL FINDING: scalar and SIMD kernels disagree on the F32/Q8_0 greedy path — numeric-path invariance is broken; investigate the kernel variants, do not delete this test"
else ok; fi

step "chat-template smoke: template autodetect on the test model, generate works"
run "$WAYRT" generate --greedy --max-tokens 8 --prompt "$PROMPT" "$LLAMA"
if [ "$RC" -ne 0 ]; then bad "exit $RC"
elif [ ! -s "$T/out" ]; then bad "empty output"
else ok; fi

# ------------------------------------------------------------------
# 5. SDK harness: compile test/sdk_gen.c against the public header +
#    static library only, then fold its checks into the totals.
# ------------------------------------------------------------------

CC="${CC:-gcc}"
step "sdk_gen.c compiles warning-free against the SDK surface"
if $CC -std=c11 -O2 -Wall -Wextra -Wshadow -Wvla -Werror -Iinclude \
       -o "$T/sdk_gen" test/sdk_gen.c "$LIB" -lpthread -lm \
       >"$T/out" 2>"$T/err"; then
    ok
else
    bad "compile failed"
fi

fold_harness() {  # fold_harness <label> <args...>
    local label="$1"; shift
    if [ ! -x "$T/sdk_gen" ]; then
        step "$label"; bad "harness not built"; return
    fi
    "$T/sdk_gen" "$@" >"$T/out" 2>"$T/err"
    local rc=$?
    sed 's/^/      /' "$T/out"
    local hp hf
    hp="$(sed -n 's/^sdk_gen .*: \([0-9][0-9]*\) passed.*/\1/p' "$T/out")"
    hf="$(sed -n 's/.* \([0-9][0-9]*\) failed$/\1/p' "$T/out")"
    if [ -z "$hp" ] || [ -z "$hf" ]; then
        step "$label"; bad "harness exit $rc, no summary"
        return
    fi
    PASS=$((PASS + hp))
    FAIL=$((FAIL + hf))
    STEP=$((STEP + hp + hf))
    if [ "$hf" -ne 0 ] || [ "$rc" -ne 0 ]; then
        say "      ($label: $hf failed, exit $rc)"
    fi
}

say "-- sdk harness on $(basename "$LLAMA") --"
fold_harness "sdk harness" sdk "$LLAMA"
say "-- sampler statistics (synthetic logits) --"
fold_harness "stats harness" stats
say "-- concurrent sessions vs sequential reference (8 threads + batch) --"
fold_harness "conc harness" conc "$LLAMA"

# ------------------------------------------------------------------

TOTAL=$((PASS + FAIL))
say ""
if [ "$SKIP" -ne 0 ]; then
    say "integration: $PASS/$TOTAL passed ($SKIP optional step(s) skipped: model not present)"
else
    say "integration: $PASS/$TOTAL passed"
fi
[ "$FAIL" -eq 0 ]
