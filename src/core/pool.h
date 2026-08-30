/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * pool.h — persistent worker pool.
 *
 * One pool per engine, created at engine init, destroyed at engine
 * shutdown.  Replaces BOTH threading mechanisms of the origin OS's
 * compute engine (transient per-call kernel threads for M-split matmul,
 * and the spin/park/steal persistent pool for quantized N-split matmul)
 * with a single plain condvar pool.  The park/steal/rehome apparatus is
 * deliberately NOT ported — it existed to work around origin-scheduler
 * pathologies; only the work-PARTITIONING math survives, in the ops that
 * call wr_pool_run.
 *
 * ------------------------------------------------------------------
 * Contract of wr_pool_run (everything matmul bit-exactness rests on)
 * ------------------------------------------------------------------
 *  1. fn(arg, part, n_parts) is invoked exactly once for every
 *     part in [0, n_parts).
 *  2. Each part executes entirely on ONE thread, start to finish.
 *     It is never split, migrated, restarted, or re-executed.  The
 *     N-split quantized matmul relies on this: every output column's
 *     complete k-reduction stays inside one part, so float summation
 *     order — and therefore the result — is identical for any thread
 *     count, 1..n.  Never "help" a slow part from another thread.
 *  3. Barrier semantics: wr_pool_run returns only after every part has
 *     completed, and all memory writes performed inside parts
 *     happen-before the return (release/acquire via the pool's mutex).
 *  4. The calling thread participates: it executes parts too, so a
 *     1-thread pool (n_threads == 1) degenerates to a plain serial
 *     loop with zero cross-thread traffic.
 *  5. Single dispatcher: at most one wr_pool_run executes at a time.
 *     Concurrent callers (e.g. two sessions decoding on two user
 *     threads) serialize on an internal mutex — the ported equivalent
 *     of the origin's single-dispatcher busy flag.  Ops never assume
 *     they own the pool without being inside wr_pool_run.
 *  6. n_parts may exceed the thread count (threads loop, claiming
 *     parts from a shared counter) and may be less (extra threads
 *     stay parked).  n_parts == 0 returns immediately.
 *  7. Part claim order is unspecified.  Correctness must never depend
 *     on which thread runs which part, only on the partitioning.
 *
 * fn must not call wr_pool_run recursively and must not block on locks
 * held by code that is waiting for the pool.
 */
#ifndef WR_POOL_H
#define WR_POOL_H

#include <stdint.h>

typedef struct wr_pool wr_pool;

/* fn(arg, part, n_parts): execute one partition of the job. */
typedef void (*wr_pool_fn)(void *arg, uint32_t part, uint32_t n_parts);

/* Create a pool with `n_threads` total workers INCLUDING the caller
 * (n_threads == 4 spawns 3 threads).  n_threads is clamped to
 * [1, WR_MAX_WORKERS].  Returns NULL on thread-creation failure (the
 * engine then fails to create — no degraded half-pool). */
wr_pool *wr_pool_create(uint32_t n_threads);

/* Join and free all workers.  Must not be called while a wr_pool_run is
 * in flight. */
void wr_pool_destroy(wr_pool *p);

/* Run one parallel job — see the contract above.  Blocks until done. */
void wr_pool_run(wr_pool *p, wr_pool_fn fn, void *arg, uint32_t n_parts);

/* Total worker count (including the caller), as clamped at create. */
uint32_t wr_pool_size(const wr_pool *p);

#endif /* WR_POOL_H */
