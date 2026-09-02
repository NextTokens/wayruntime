# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 WayOS Project
# wayruntime — build
#
#   make                 build libwayruntime.a + wayrt (host POSIX, gcc)
#   make WIN=1           cross-compile Windows binaries (mingw-w64)
#   make unit            build + run the unit tests
#   make test            license-check + check-api + unit (offline; runs from a clone)
#   make check           test + integration (needs a small llama-family GGUF,
#                        WR_TEST_LLAMA=path; see docs/TESTING.md)
#   make realtest MODEL_GGUF=path/to/model.gguf   end-to-end on a real model
#   make benchmark MODEL_GGUF=path/to/model.gguf  POSIX-host thread sweep
#   make MATH_APPROX=0   build the libm-backed math path (see mathx.h)
#   make clean

VERSION := 0.1.0

# Toolchains are pinned: gcc (POSIX) and x86_64-w64-mingw32-gcc (Windows).
# The kernels use GCC vector extensions, per-function target attributes
# and __atomic builtins; MSVC is out of scope by design.  -O3 is intentional:
# the block-quant row decoders are on every token's hot path and benefit
# materially from the additional inlining and loop optimization.
CFLAGS ?= -std=c11 -O3 -Wall -Wextra -Wshadow -Wvla -g
CFLAGS += -Iinclude -Isrc
# Header dependency tracking: every object records the headers it
# included, so a header edit rebuilds exactly the objects that use it.
CFLAGS += -MMD -MP

MATH_APPROX ?= 1
CFLAGS += -DWR_MATH_APPROX=$(MATH_APPROX)

CORE_SRC := src/core/engine.c src/core/kernels.c src/core/mathx.c \
            src/core/quant.c src/core/tensor.c src/core/pool.c \
            src/core/gguf.c src/core/bpe.c src/core/tokenizer.c \
            src/core/model.c src/core/loader.c src/core/session.c \
            src/core/batch.c src/core/sample.c src/core/generate.c
CLI_SRC  := src/cli/wayrt.c
TEST_SRC := test/unit_tests.c

# WIN=1 selects the Windows cross build; any other value (or unset) is the
# host POSIX build (`ifeq`, not `ifdef`: WIN=0 must mean off).
ifeq ($(WIN),1)
CC       := x86_64-w64-mingw32-gcc
AR       := x86_64-w64-mingw32-ar
PLAT_SRC := src/platform/win/os_win.c src/platform/win/cpu_features.c
LIBS     :=
EXE      := .exe
BUILDDIR := build/win
else
# GNU make predefines CC=cc, so `?=` would never pin gcc; only an explicit
# user override (command line / environment) is honored.
ifeq ($(origin CC),default)
CC       := gcc
endif
ifeq ($(origin AR),default)
AR       := ar
endif
PLAT_SRC := src/platform/posix/os_posix.c src/platform/posix/cpu_features.c
LIBS     := -lpthread -lm
EXE      :=
BUILDDIR := build/posix
endif

LIB_SRC  := $(CORE_SRC) $(PLAT_SRC)
LIB_OBJS := $(patsubst src/%.c,$(BUILDDIR)/%.o,$(LIB_SRC))
LIB      := $(BUILDDIR)/libwayruntime.a

all: $(LIB) $(BUILDDIR)/wayrt$(EXE)

$(LIB): $(LIB_OBJS) Makefile
	@mkdir -p $(dir $@)
	$(RM) $@
	$(AR) rcs $@ $(LIB_OBJS)

$(BUILDDIR)/%.o: src/%.c Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/test/%.o: test/%.c Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/wayrt$(EXE): $(BUILDDIR)/cli/wayrt.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $(BUILDDIR)/cli/wayrt.o $(LIB) $(LIBS)

$(BUILDDIR)/wayrt_tests$(EXE): $(BUILDDIR)/test/unit_tests.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $(BUILDDIR)/test/unit_tests.o $(LIB) $(LIBS)

-include $(LIB_OBJS:.o=.d) $(BUILDDIR)/cli/wayrt.d $(BUILDDIR)/test/unit_tests.d

.PHONY: all unit test check realtest benchmark check-api license-check \
        integration realtest-diff usecase clean

unit: $(BUILDDIR)/wayrt_tests$(EXE)
	$(BUILDDIR)/wayrt_tests$(EXE)

# SDK-boundary gate: the public header must stand alone — a strict
# translation unit that includes ONLY <wayruntime/wayruntime.h> compiles
# with no src/ include path, -pedantic, warnings as errors.
check-api:
	@mkdir -p $(BUILDDIR)/test
	$(CC) -std=c11 -pedantic -Wall -Wextra -Werror -Iinclude \
	      -c test/check_api.c -o $(BUILDDIR)/test/check_api.o
	@echo "check-api: public header is freestanding"

license-check:
	python3 test/check_license_metadata.py

test: license-check check-api unit

# Everything `test` runs plus the real-model integration suite.
check: test integration

# CLI + hostile-GGUF + SDK integration suite against a real model (see
# test/integration.sh).  WR_TEST_LLAMA=path names the llama-family GGUF
# it runs on (default ../models/stories15M-q8_0.gguf next to the
# repository; MODELS_DIR=... moves the directory); WR_TEST_Q4K/Q5K/Q6K
# name optional K-quant files for the extra load checks.
integration: all
	bash test/integration.sh $(BUILDDIR)

# Documented use case: schema-shaped JSON by construction via the
# token-mask API (docs/USECASE.md <-> test/usecase_grammar.sh; needs a
# real model, default ../models/Qwen3-0.6B-Q4_K_M.gguf, override with
# MODEL_GGUF=path or MODELS_DIR=dir).
usecase: all
	bash test/usecase_grammar.sh $(BUILDDIR) "$(MODEL_GGUF)"

# Differential against a live llama.cpp reference server (tokenizer
# corpus + teacher-forced argmax agreement).  Exits 3 = SKIP when the
# server is down.  Release defaults require 19/20 tokenizer matches and
# exact 20/20 teacher-forced agreement.  Override either minimum only for
# explicit diagnostic runs; the script has the same standalone defaults.
DIFF_TOKENIZER_MIN ?= 19
DIFF_TEACHER_MIN ?= 20
realtest-diff: all
	bash test/realtest_diff.sh $(BUILDDIR) $(DIFF_TOKENIZER_MIN) $(DIFF_TEACHER_MIN)

# realtest-win: test/realtest.ps1 is Windows-native (needs build\win
# binaries from `make WIN=1`); there is no make target on purpose —
# run it from Windows:  powershell -File test\realtest.ps1

# End-to-end against a real GGUF model (no mocks): load, tokenize,
# greedy-generate, verify counters.  Pass MODEL_GGUF=path.
MODEL_GGUF ?=
realtest: all
	bash test/realtest.sh $(BUILDDIR) "$(MODEL_GGUF)"

# Local performance evidence, deliberately separate from the correctness
# gates.  Values are explicit and machine-independent; callers must provide
# the model and may opt into a CPU-only llama.cpp comparison by setting the
# path to its llama-bench executable.
BENCH_THREADS ?= 1,2,4
BENCH_TOKENS ?= 32
BENCH_REPETITIONS ?= 3
BENCH_WARMUPS ?= 1
LLAMA_BENCH ?=
ifeq ($(WIN),1)
benchmark:
	@echo "benchmark: WIN=1 cross-builds binaries that cannot run on the POSIX host" >&2
	@exit 2
else
benchmark: all
	@BENCH_THREADS="$(BENCH_THREADS)" BENCH_TOKENS="$(BENCH_TOKENS)" \
	BENCH_REPETITIONS="$(BENCH_REPETITIONS)" BENCH_WARMUPS="$(BENCH_WARMUPS)" \
	LLAMA_BENCH="$(LLAMA_BENCH)" \
	bash test/benchmark.sh $(BUILDDIR) "$(MODEL_GGUF)"
endif

clean:
	rm -rf build
