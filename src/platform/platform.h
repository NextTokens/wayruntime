/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * platform.h — the OS shim contract.
 *
 * Exactly two implementations exist:
 *   src/platform/posix/os_posix.c + cpu_features.c   (Linux, gcc)
 *   src/platform/win/os_win.c     + cpu_features.c   (Windows, mingw-w64)
 *
 * Everything above this line of the library is OS-agnostic C11 and calls
 * ONLY this API (plus stdio/stdlib/string).  No #ifdef _WIN32 outside
 * src/platform/ — except the two seek macros below, which exist so GGUF
 * loading can stay on plain buffered stdio.
 *
 * All functions are callable from any thread unless noted.  None of them
 * log; they return errors and the caller decides.
 */
#ifndef WR_PLATFORM_H
#define WR_PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* --------------------------------------------------------------------------
 * 64-bit file offsets over stdio
 *
 * GGUF files exceed 2 GiB; plain fseek/ftell take a long, which is 32-bit
 * on Windows (LLP64).  All file positioning in the library goes through
 * these macros.  Never call fseek/ftell directly.
 * -------------------------------------------------------------------------- */
#if defined(_WIN32)
#  define wr_fseek64(f, off, whence) _fseeki64((f), (off), (whence))
#  define wr_ftell64(f)              _ftelli64(f)
#else
#  define wr_fseek64(f, off, whence) fseeko((f), (off_t)(off), (whence))
#  define wr_ftell64(f)              ((int64_t)ftello(f))
#endif

/* Size of an open file in bytes via fstat/GetFileSizeEx — the anchor for
 * GGUF bounds validation (every tensor offset+size is checked against
 * this BEFORE any tensor read).  Returns 0 on success, -1 on failure. */
int wr_file_size(FILE *f, uint64_t *out_bytes);

/* --------------------------------------------------------------------------
 * Threads
 * -------------------------------------------------------------------------- */

typedef struct wr_thread wr_thread;   /* opaque; heap-allocated */

/* Start a thread running fn(arg).  Returns NULL on failure.  Every
 * created thread must be joined exactly once. */
wr_thread *wr_thread_create(void (*fn)(void *arg), void *arg);

/* Join and free the thread object.  Returns 0 on success. */
int wr_thread_join(wr_thread *t);

/* --------------------------------------------------------------------------
 * Mutex + condition variable
 *
 * Plain blocking primitives (pthread_mutex/pthread_cond on POSIX,
 * SRWLOCK/CONDITION_VARIABLE on Windows).  No spinning, no affinity, no
 * priority games — the worker pool's correctness must not depend on any
 * scheduler behavior beyond "condvars wake".
 * -------------------------------------------------------------------------- */

typedef struct wr_mutex wr_mutex;     /* opaque; heap-allocated */
typedef struct wr_cond  wr_cond;      /* opaque; heap-allocated */

wr_mutex *wr_mutex_create(void);
void      wr_mutex_destroy(wr_mutex *m);
void      wr_mutex_lock(wr_mutex *m);
void      wr_mutex_unlock(wr_mutex *m);

wr_cond *wr_cond_create(void);
void     wr_cond_destroy(wr_cond *c);
/* Atomically unlock m, wait, relock m before returning.  Spurious wakeups
 * are permitted — callers loop on their predicate. */
void     wr_cond_wait(wr_cond *c, wr_mutex *m);
void     wr_cond_signal(wr_cond *c);
void     wr_cond_broadcast(wr_cond *c);

/* --------------------------------------------------------------------------
 * Time
 * -------------------------------------------------------------------------- */

/* Monotonic microseconds (CLOCK_MONOTONIC / QueryPerformanceCounter).
 * The zero point is arbitrary; only differences are meaningful.  This is
 * the ONLY time source in the library — benches included (no rdtsc/cntvct
 * inline asm anywhere). */
uint64_t wr_monotonic_us(void);

/* --------------------------------------------------------------------------
 * CPU
 * -------------------------------------------------------------------------- */

/* Online logical CPU count (>= 1). */
uint32_t wr_cpu_count(void);

/* CPU feature probe — computed once, cached. */
#define WR_CPU_FEAT_AVX2    (1u << 0)  /* AVX2 + FMA usable */
#define WR_CPU_FEAT_AVX512  (1u << 1)  /* AVX-512F + DQ usable */
#define WR_CPU_FEAT_NEON    (1u << 2)  /* arm64 ASIMD usable */
uint32_t wr_cpu_features(void);

/* Spin-wait hint for the rare short waits (_mm_pause on x86, yield/isb on
 * arm64, no-op elsewhere). */
void wr_plat_pause(void);

/* --------------------------------------------------------------------------
 * Memory
 * -------------------------------------------------------------------------- */

/* Aligned heap allocation for small tensors and op scratch.  `align` must
 * be a power of two (the library uses 64).  Free ONLY with
 * wr_aligned_free — the two are a matched pair on Windows
 * (_aligned_malloc/_aligned_free). */
void *wr_aligned_alloc(size_t align, size_t size);
void  wr_aligned_free(void *p);

/* Large-arena allocation for the weight slab and KV caches: one mapping of
 * `size` bytes, read/write, zero-filled, page-aligned (mmap ANONYMOUS on
 * POSIX, VirtualAlloc RESERVE|COMMIT on Windows).  Returns NULL on
 * failure.  Release with wr_map_free(ptr, size) — `size` must be the
 * value passed at allocation. */
void *wr_map_alloc(size_t size);
void  wr_map_free(void *p, size_t size);

/* Read-only file mapping for the optional zero-copy GGUF weight path.
 * Maps `length` bytes at absolute file offset `offset` (which the
 * implementation aligns down internally; the returned pointer addresses
 * exactly byte `offset`).  Returns NULL on failure — the caller then
 * falls back to streamed reads.  Unmap with wr_file_unmap using the SAME
 * pointer and length. */
void *wr_file_map_ro(FILE *f, uint64_t offset, size_t length);
void  wr_file_unmap(void *p, size_t length);

#endif /* WR_PLATFORM_H */
