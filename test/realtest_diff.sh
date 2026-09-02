# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 WayOS Project
#
# realtest_diff.sh -- differential test against a LIVE llama.cpp
# reference server.  Always invoked as `bash test/realtest_diff.sh`
# (no shebang by design: the SPDX header owns line 1).
#
#   bash test/realtest_diff.sh [builddir] [tokenizer-threshold] [tf-threshold]
#     builddir            default build/posix
#     tokenizer-threshold minimum exact-match count over the 20-string
#                         corpus, default 19
#     tf-threshold        minimum teacher-forced argmax matches over the
#                         20-token continuation, default 20 (exact)
#
# Both thresholds are parameters so deliberately relaxed diagnostic runs
# can be made without weakening the checked-in release defaults.
#
#   env: WR_LLAMA_URL   reference server (default http://127.0.0.1:18080)
#        MODELS_DIR     model directory (default ../models next to repo)
#        WR_DIFF_MODEL  GGUF path; MUST be the same model the server
#                       serves (default $MODELS_DIR/Qwen3-0.6B-Q4_K_M.gguf)
#
# Three comparisons:
#   (a) tokenizer corpus: 20 strings through wr_tokenize vs POST
#       /tokenize; reports the exact-match count, asserts >= threshold.
#   (b) 20-step teacher-forced argmax agreement: llama.cpp's greedy
#       continuation of prompt ids 12522,5193,264,882 is replayed
#       token-by-token through wr_step; each step's argmax is compared
#       with the reference token; asserts 20/20 by default.
#   (c) the compiled-in fixture corpus (test/tokenizer_fixtures.h,
#       plain + parse-special sets) replayed through the SDK harness's
#       `tokfix` mode; EVERY fixture must match the reference exactly
#       (no threshold: this is the README's 78/78 claim, re-proven).
#
# Exit codes: 0 pass, 1 fail, 2 setup error, 3 SKIP (server down —
# deliberately distinct from failure).
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

BUILDDIR="${1:-build/posix}"
TOK_THRESH="${2:-19}"
TF_THRESH="${3:-20}"
URL="${WR_LLAMA_URL:-http://127.0.0.1:18080}"
MODELS_DIR="${MODELS_DIR:-$(cd "$ROOT/.." && pwd)/models}"
MODEL="${WR_DIFF_MODEL:-$MODELS_DIR/Qwen3-0.6B-Q4_K_M.gguf}"

case "$TOK_THRESH" in
    ''|*[!0-9]*)
        echo "realtest-diff: tokenizer threshold must be a non-negative integer" >&2
        exit 2
        ;;
esac
case "$TF_THRESH" in
    ''|*[!0-9]*)
        echo "realtest-diff: teacher-forced threshold must be a non-negative integer" >&2
        exit 2
        ;;
esac
if [ "$TOK_THRESH" -gt 20 ] || [ "$TF_THRESH" -gt 20 ]; then
    echo "realtest-diff: thresholds cannot exceed the 20-case corpus" >&2
    exit 2
fi

LIB="$BUILDDIR/libwayruntime.a"
if [ ! -f "$LIB" ]; then
    echo "realtest-diff: $LIB not built (run make first)" >&2
    exit 2
fi
if [ ! -f "$MODEL" ]; then
    echo "realtest-diff: model not found: $MODEL" >&2
    exit 2
fi

# ------------------------------------------------------------------
# Reference server health.  Unreachable = SKIP (exit 3), NOT failure:
# the differential is only meaningful against the live reference.
# ------------------------------------------------------------------

if ! curl -fsS -m 5 "$URL/health" >/dev/null 2>&1; then
    cat >&2 <<EOF
realtest-diff: llama.cpp reference server unreachable at $URL
  restart it (inside WSL):
    cd ~/llamasrv/llama-b10621 && \\
    LD_LIBRARY_PATH=\$PWD nohup ./llama-server \\
      -m ~/llamasrv/Qwen3-0.6B-Q4_K_M.gguf \\
      --host 127.0.0.1 --port 18080 -c 4096 --threads 8 &
  then re-run: bash test/realtest_diff.sh $BUILDDIR $TOK_THRESH $TF_THRESH
realtest-diff: SKIP (exit 3)
EOF
    exit 3
fi

# ------------------------------------------------------------------
# Build the SDK harness (tok + tf modes of test/sdk_gen.c).
# ------------------------------------------------------------------

T="$(mktemp -d /tmp/wayrt-diff-XXXXXX)"
trap 'rm -rf "$T"' EXIT

CC="${CC:-gcc}"
if ! $CC -std=c11 -O2 -Wall -Wextra -Wshadow -Wvla -Werror -Iinclude \
        -o "$T/sdk_gen" test/sdk_gen.c "$LIB" -lpthread -lm; then
    echo "realtest-diff: harness compile failed" >&2
    exit 2
fi

# ------------------------------------------------------------------
# The differential proper (python3 owns the JSON plumbing).
# ------------------------------------------------------------------

WRD_URL="$URL" WRD_SDK="$T/sdk_gen" WRD_MODEL="$MODEL" \
WRD_TOK_THRESH="$TOK_THRESH" WRD_TF_THRESH="$TF_THRESH" python3 - <<'PYEOF'
import json
import os
import subprocess
import sys
import urllib.request

url = os.environ["WRD_URL"]
sdk = os.environ["WRD_SDK"]
model = os.environ["WRD_MODEL"]
tok_thresh = int(os.environ["WRD_TOK_THRESH"])
tf_thresh = int(os.environ["WRD_TF_THRESH"])


def post(path, obj):
    req = urllib.request.Request(
        url + path,
        json.dumps(obj).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


# ---- (a) tokenizer corpus diff -----------------------------------

corpus = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces",
    "trailing spaces  ",
    "1234567890 3.14159 -42",
    "CamelCaseIdentifier snake_case_name SCREAMING_CASE",
    "def main(argv): return 0  # python",
    "int main(void) { return 0; }",
    "https://example.com/path?q=1&x=2#frag",
    r"C:\Users\test\file.txt",
    "don't stop can't won't it's",
    "na\u00efve fa\u00e7ade \u00fcber r\u00e9sum\u00e9",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059",
    "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440!",
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
    "\U0001f600 emoji \U0001f680\U0001f525 test \U0001f389",
    "MixedNumbers123AndText456End",
    "    indented    with     runs of spaces",
    "\u03a9\u2248\u00e7\u221a\u222b\u02dc\u00b5\u2264\u2265\u00f7 unicode math",
    "repeat repeat repeat repeat repeat repeat repeat repeat",
]

p = subprocess.run(
    [sdk, "tok", model],
    input=("\n".join(corpus) + "\n").encode("utf-8"),
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
if p.returncode != 0:
    print("realtest-diff: tok harness failed (exit %d)" % p.returncode)
    sys.exit(2)
ours_lines = p.stdout.decode("utf-8").splitlines()
if len(ours_lines) != len(corpus):
    print(
        "realtest-diff: tok harness returned %d lines for %d strings"
        % (len(ours_lines), len(corpus))
    )
    sys.exit(2)

tok_match = 0
for i, text in enumerate(corpus):
    ref = post("/tokenize", {"content": text})["tokens"]
    ours = [int(x) for x in ours_lines[i].split()]
    same = ours == ref
    tok_match += same
    print("tok %2d  %s  %r" % (i + 1, "match" if same else "DIFF ", text))
    if not same:
        print("        ours: %s" % ours)
        print("        ref:  %s" % ref)

# ---- (c) compiled-in fixture corpus ------------------------------

p = subprocess.run(
    [sdk, "tokfix", model],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
if p.returncode != 0:
    print("realtest-diff: tokfix harness failed (exit %d)" % p.returncode)
    sys.exit(2)
fix_total = 0
fix_match = 0
for line in p.stdout.decode("utf-8").split("\n"):
    if not line:
        continue
    flags, jtext, idstr = line.split("\t", 2)
    text = json.loads(jtext)
    ours = [int(x) for x in idstr.split(",")] if idstr else []
    body = {"content": text}
    if int(flags) & 1:
        body["parse_special"] = True
    ref = post("/tokenize", body)["tokens"]
    fix_total += 1
    same = ours == ref
    fix_match += same
    if not same:
        print("fixture %3d  DIFF  %r" % (fix_total, text))
        print("        ours: %s" % ours)
        print("        ref:  %s" % ref)
print("fixtures: %d/%d exact" % (fix_match, fix_total))

# ---- (b) teacher-forced argmax agreement -------------------------

prompt_ids = [12522, 5193, 264, 882]
resp = post(
    "/completion",
    {
        "prompt": prompt_ids,
        "n_predict": 20,
        "temperature": 0,
        "return_tokens": True,
        "cache_prompt": False,
    },
)
teacher = list(resp.get("tokens", []))[:20]
if len(teacher) < 2:
    print("realtest-diff: reference /completion returned %d tokens"
          % len(teacher))
    sys.exit(2)

p = subprocess.run(
    [
        sdk,
        "tf",
        model,
        ",".join(str(t) for t in prompt_ids),
        ",".join(str(t) for t in teacher),
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
out = p.stdout.decode("utf-8")
sys.stdout.write(out)
if p.returncode != 0:
    print("realtest-diff: tf harness failed (exit %d)" % p.returncode)
    sys.exit(2)

agree = n_steps = None
for line in out.splitlines():
    if line.startswith("tf: agree="):
        agree, n_steps = map(int, line.split("=")[1].split("/"))
if agree is None:
    print("realtest-diff: tf harness printed no agreement summary")
    sys.exit(2)

# ---- summary -----------------------------------------------------

tok_ok = tok_match >= tok_thresh
tf_ok = agree >= tf_thresh
fix_ok = fix_total > 0 and fix_match == fix_total
print(
    "realtest-diff: tokenizer %d/%d exact (threshold %d) %s, "
    "fixtures %d/%d exact %s, "
    "teacher-forced %d/%d (threshold %d) %s"
    % (
        tok_match, len(corpus), tok_thresh, "OK" if tok_ok else "FAIL",
        fix_match, fix_total, "OK" if fix_ok else "FAIL",
        agree, n_steps, tf_thresh, "OK" if tf_ok else "FAIL",
    )
)
all_ok = tok_ok and tf_ok and fix_ok
print("realtest-diff: %s" % ("PASS" if all_ok else "FAIL"))
sys.exit(0 if all_ok else 1)
PYEOF
