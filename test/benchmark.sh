#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 WayOS Project
#
# Repeatable local performance sweep for wayruntime.  This is evidence,
# not a correctness gate: it deliberately requires the caller to name the
# model and makes every workload parameter visible in the output.

set -eu

usage()
{
    cat >&2 <<'EOF'
usage: benchmark.sh <builddir> <model.gguf>

Environment:
  BENCH_THREADS       comma-separated thread counts in 1..32 (default: 1,2,4)
  BENCH_TOKENS        decode tokens per sample (default: 32)
  BENCH_REPETITIONS   measured samples per thread count (default: 3)
  BENCH_WARMUPS       discarded warmups per thread count (default: 1)
  LLAMA_BENCH         optional path to a llama.cpp llama-bench executable
  WAYRT               optional wayrt executable (default: <builddir>/wayrt)
  BENCH_DRY_RUN       1 prints commands without checking paths or running them

Comma-delimited sample, summary, and optional llama.cpp report sections are
written to stdout. Progress and errors go to stderr. Redirect stdout to keep
a result artifact.
EOF
}

die()
{
    echo "benchmark: $*" >&2
    exit 2
}

print_command()
{
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
}

is_positive_integer()
{
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if [ "$#" -ne 2 ]; then
    usage
    exit 2
fi

BUILDDIR=$1
MODEL=$2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
WAYRT_BIN=${WAYRT:-"$BUILDDIR/wayrt"}
THREADS_TEXT=${BENCH_THREADS:-1,2,4}
TOKENS=${BENCH_TOKENS:-32}
REPETITIONS=${BENCH_REPETITIONS:-3}
WARMUPS=${BENCH_WARMUPS:-1}
LLAMA_BENCH_BIN=${LLAMA_BENCH:-}
DRY_RUN=${BENCH_DRY_RUN:-0}

[[ "$THREADS_TEXT" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] ||
    die "BENCH_THREADS must be a comma-separated list of positive integers"
is_positive_integer "$TOKENS" || die "BENCH_TOKENS must be a positive integer"
is_positive_integer "$REPETITIONS" ||
    die "BENCH_REPETITIONS must be a positive integer"
[[ "$WARMUPS" =~ ^[0-9]+$ ]] ||
    die "BENCH_WARMUPS must be a non-negative integer"
[ "$DRY_RUN" = 0 ] || [ "$DRY_RUN" = 1 ] ||
    die "BENCH_DRY_RUN must be 0 or 1"

IFS=',' read -r -a THREADS <<<"$THREADS_TEXT"
for i in "${!THREADS[@]}"; do
    thread_value=$((10#${THREADS[$i]}))
    [ "$thread_value" -le 32 ] ||
        die "BENCH_THREADS exceeds wayruntime's compiled worker cap (32)"
    for ((j = 0; j < i; j++)); do
        [ "${THREADS[$j]}" != "$thread_value" ] ||
            die "BENCH_THREADS contains duplicate count: $thread_value"
    done
    THREADS[$i]=$thread_value
done
THREADS_TEXT=$(IFS=,; printf '%s' "${THREADS[*]}")

if [ "$DRY_RUN" = 1 ]; then
    for thread in "${THREADS[@]}"; do
        for ((run = 1; run <= WARMUPS; run++)); do
            printf '# warmup threads=%s sample=%s\n' "$thread" "$run"
            print_command "$WAYRT_BIN" --threads "$thread" bench \
                --tokens "$TOKENS" "$MODEL"
        done
        for ((run = 1; run <= REPETITIONS; run++)); do
            printf '# measured threads=%s sample=%s\n' "$thread" "$run"
            print_command "$WAYRT_BIN" --threads "$thread" bench \
                --tokens "$TOKENS" "$MODEL"
        done
    done
    if [ -n "$LLAMA_BENCH_BIN" ]; then
        printf '# optional llama.cpp CPU comparison\n'
        print_command "$LLAMA_BENCH_BIN" -m "$MODEL" -p 0 -n "$TOKENS" \
            -t "$THREADS_TEXT" -r "$REPETITIONS" -ngl 0 -o csv
    fi
    exit 0
fi

[ -x "$WAYRT_BIN" ] || die "wayrt is not executable: $WAYRT_BIN"
[ -f "$MODEL" ] || die "model is not a regular file: $MODEL"
if [ -n "$LLAMA_BENCH_BIN" ]; then
    [ -x "$LLAMA_BENCH_BIN" ] ||
        die "LLAMA_BENCH is not executable: $LLAMA_BENCH_BIN"
fi

# The CLI prints rates with a decimal point. Pin parsing and awk output to
# the C locale so result files do not depend on the host locale.
LC_ALL=C
export LC_ALL

GIT_COMMIT=unavailable
GIT_STATE=unavailable
if command -v git >/dev/null 2>&1 &&
        GIT_COMMIT=$(git -C "$REPO_ROOT" rev-parse --verify HEAD 2>/dev/null); then
    if [ -z "$(git -C "$REPO_ROOT" status --porcelain=v1 \
            --untracked-files=normal 2>/dev/null)" ]; then
        GIT_STATE=clean
    else
        GIT_STATE=dirty
    fi
fi

WAYRT_PATH=$WAYRT_BIN
if command -v realpath >/dev/null 2>&1; then
    WAYRT_PATH=$(realpath -- "$WAYRT_BIN")
fi

sha256_file()
{
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -- "$1" | awk '{ print $1 }'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 -- "$1" | awk '{ print $1 }'
    else
        printf 'unavailable\n'
    fi
}

WAYRT_SHA256=$(sha256_file "$WAYRT_BIN")
MODEL_SHA256=$(sha256_file "$MODEL")
HOST_OS=$(uname -srm 2>/dev/null || printf 'unavailable\n')
HOST_CPU=unavailable
if [ -r /proc/cpuinfo ]; then
    HOST_CPU=$(awk -F: '
        /^[[:space:]]*model name[[:space:]]*:/ {
            sub(/^[[:space:]]+/, "", $2); print $2; exit
        }
    ' /proc/cpuinfo)
elif command -v sysctl >/dev/null 2>&1; then
    HOST_CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null ||
        printf 'unavailable\n')
fi
[ -n "$HOST_CPU" ] || HOST_CPU=unavailable

SAMPLES_FILE=$(mktemp)
cleanup()
{
    rm -f -- "$SAMPLES_FILE"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

printf '# wayruntime benchmark v2\n'
printf '# repo_worktree_git=%s state=%s\n' "$GIT_COMMIT" "$GIT_STATE"
printf '# wayruntime_executable='; printf '%q\n' "$WAYRT_PATH"
printf '# wayruntime_sha256=%s\n' "$WAYRT_SHA256"
printf '# model='; printf '%q\n' "$MODEL"
printf '# model_sha256=%s\n' "$MODEL_SHA256"
printf '# host_os='; printf '%q\n' "$HOST_OS"
printf '# host_cpu='; printf '%q\n' "$HOST_CPU"
printf '# tokens=%s repetitions=%s warmups=%s threads=%s\n' \
    "$TOKENS" "$REPETITIONS" "$WARMUPS" "$THREADS_TEXT"
printf 'runtime,requested_threads,threads,sample,simd,prefill_tokens,prefill_ms,prefill_tok_s,decode_tokens,decode_ms,decode_tok_s\n'

for thread in "${THREADS[@]}"; do
    for ((run = 1; run <= WARMUPS; run++)); do
        echo "benchmark: wayruntime warmup threads=$thread sample=$run/$WARMUPS" >&2
        if output=$("$WAYRT_BIN" --threads "$thread" bench \
                --tokens "$TOKENS" "$MODEL" 2>&1); then
            :
        else
            status=$?
            printf '%s\n' "$output" >&2
            exit "$status"
        fi
    done

    for ((run = 1; run <= REPETITIONS; run++)); do
        echo "benchmark: wayruntime measured threads=$thread sample=$run/$REPETITIONS" >&2
        if output=$("$WAYRT_BIN" --threads "$thread" bench \
                --tokens "$TOKENS" "$MODEL" 2>&1); then
            :
        else
            status=$?
            printf '%s\n' "$output" >&2
            exit "$status"
        fi

        simd_line=$(printf '%s\n' "$output" | awk '
            $1 == "simd:" && $3 == "threads:" { print $2 "," $4; exit }
        ')
        prefill=$(printf '%s\n' "$output" | awk '
            $1 == "prefill:" {
                tokens = $2
                ms = ""
                for (i = 1; i < NF; i++)
                    if ($(i + 1) == "ms") ms = $i
                # A one-token prompt has no prefill operation, so the CLI
                # deliberately reports no elapsed time.  Represent that
                # valid case as zero work in the machine-readable report.
                if (tokens == 0 && ms == "") ms = 0
                if (ms != "") print tokens "," ms
                exit
            }
        ')
        decode=$(printf '%s\n' "$output" | awk '
            $1 == "decode:" {
                tokens = $2
                ms = ""
                for (i = 1; i < NF; i++)
                    if ($(i + 1) == "ms") ms = $i
                if (ms != "") print tokens "," ms
                exit
            }
        ')

        [ -n "$simd_line" ] ||
            die "could not parse SIMD/thread data from wayrt output"
        [ -n "$prefill" ] || die "could not parse prefill time from wayrt output"
        [ -n "$decode" ] || die "could not parse decode time from wayrt output"
        IFS=',' read -r simd actual_threads <<<"$simd_line"
        IFS=',' read -r prefill_tokens prefill_ms <<<"$prefill"
        IFS=',' read -r decode_tokens decode_ms <<<"$decode"
        [ "$actual_threads" = "$thread" ] ||
            die "wayrt resolved $thread requested threads to $actual_threads"
        [ "$decode_tokens" = "$TOKENS" ] ||
            die "wayrt produced $decode_tokens of $TOKENS requested decode tokens"
        [[ "$prefill_ms" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
            die "invalid prefill time from wayrt: $prefill_ms"
        [[ "$decode_ms" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
            die "invalid decode time from wayrt: $decode_ms"
        prefill_rate=$(awk -v n="$prefill_tokens" -v ms="$prefill_ms" '
            BEGIN {
                if (n == 0 && ms == 0) { printf "0.000"; exit }
                if (n <= 0 || ms <= 0) exit 1
                printf "%.3f", n * 1000 / ms
            }') || die "prefill time must be positive for nonzero work"
        decode_rate=$(awk -v n="$decode_tokens" -v ms="$decode_ms" \
            'BEGIN { if (ms <= 0) exit 1; printf "%.3f", n * 1000 / ms }') ||
            die "decode time must be positive"

        printf 'wayruntime,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$thread" "$actual_threads" "$run" "$simd" "$prefill_tokens" \
            "$prefill_ms" "$prefill_rate" "$decode_tokens" "$decode_ms" \
            "$decode_rate"
        printf '%s,%s,%s,%s\n' "$thread" "$actual_threads" \
            "$prefill_rate" "$decode_rate" \
            >>"$SAMPLES_FILE"
    done
done

printf '# wayruntime summary (sample standard deviation)\n'
printf 'runtime,requested_threads,threads,repetitions,prefill_mean_tok_s,prefill_stddev_tok_s,decode_mean_tok_s,decode_stddev_tok_s\n'
for thread in "${THREADS[@]}"; do
    awk -F, -v thread="$thread" '
        $1 == thread {
            actual = $2
            p_sum += $3; p_sq += $3 * $3
            d_sum += $4; d_sq += $4 * $4
            n++
        }
        END {
            p_mean = p_sum / n
            d_mean = d_sum / n
            if (n > 1) {
                p_var = (p_sq - n * p_mean * p_mean) / (n - 1)
                d_var = (d_sq - n * d_mean * d_mean) / (n - 1)
                if (p_var < 0) p_var = 0
                if (d_var < 0) d_var = 0
                p_stddev = sqrt(p_var)
                d_stddev = sqrt(d_var)
            } else {
                p_stddev = 0
                d_stddev = 0
            }
            printf "wayruntime,%s,%s,%d,%.3f,%.3f,%.3f,%.3f\n", \
                   thread, actual, n, p_mean, p_stddev, d_mean, d_stddev
        }
    ' "$SAMPLES_FILE"
done

if [ -n "$LLAMA_BENCH_BIN" ]; then
    echo "benchmark: running optional llama.cpp CPU comparison" >&2
    printf '# llama.cpp llama-bench raw CSV (CPU-only, generation workload)\n'
    "$LLAMA_BENCH_BIN" -m "$MODEL" -p 0 -n "$TOKENS" \
        -t "$THREADS_TEXT" -r "$REPETITIONS" -ngl 0 -o csv
fi
