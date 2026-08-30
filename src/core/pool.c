/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * pool.c — the persistent worker pool behind every parallel op.
 *
 * A deliberately plain condvar pool: one mutex guards the whole job
 * state, workers claim parts from a shared counter under that mutex and
 * execute them unlocked, and the dispatcher both participates and waits
 * on a completion count.  The origin OS's compute engine used two other
 * mechanisms (transient per-call threads for the row-split matmul, and a
 * spin/park/steal pool for the quantized column-split path); neither is
 * reproduced here — their spin limits, CPU re-homing, steal recovery and
 * fencing worked around origin-scheduler pathologies that hosted
 * schedulers do not have.  Only the partitioning contract survives, and
 * it lives in the callers (see pool.h rules 1-7).
 *
 * Correctness sketch against the pool.h contract:
 *   Rule 1 (exactly once)  next_part only advances under the mutex; each
 *          successful claim hands out a distinct index in [0, n_parts).
 *   Rule 2 (one thread)    a claimed part runs to completion on the
 *          claiming thread before that thread touches pool state again;
 *          nothing ever splits, migrates or re-runs a part.
 *   Rule 3 (barrier)       the dispatcher returns only once parts_done
 *          == n_parts, read under the same mutex workers increment it
 *          under, so every write inside a part happens-before the
 *          return of wr_pool_run.
 *   Rule 4 (participation) the dispatcher runs the same claim loop as
 *          the workers; a 1-thread pool never touches the condvars.
 *   Rule 5 (one dispatcher) run_lock serializes wr_pool_run callers.
 *   Rule 6 (any n_parts)   the claim counter naturally handles n_parts
 *          above or below the thread count; n_parts == 0 is a no-op.
 *   Rule 7 (claim order)   unspecified by design — whoever gets the
 *          mutex next claims the next index.
 *
 * Stale-claim safety (why a late-waking worker is harmless): a new job
 * cannot be published until the previous dispatcher observed parts_done
 * == n_parts and released run_lock, which requires every CLAIMED part to
 * have been reported; and a worker re-reads fn/arg/n_parts inside the
 * same critical section as each claim.  So a worker that sleeps through
 * an entire job simply finds no parts left when it wakes, and a worker
 * that wakes into a newer job claims from that job's state — an old fn
 * can never run against a new job's counter.
 */

#include <stdlib.h>

#include "core/internal.h"

struct wr_pool {
    /* Immutable after create. */
    uint32_t    n_threads;   /* total workers, INCLUDING the caller     */
    uint32_t    n_spawned;   /* threads[] entries actually running      */
    wr_thread **threads;     /* n_threads - 1 slots (NULL when serial)  */

    /* Rule 5: at most one wr_pool_run at a time. */
    wr_mutex   *run_lock;

    /* Job state — every field below is guarded by m. */
    wr_mutex   *m;
    wr_cond    *work_cv;     /* workers wait here for a job / shutdown  */
    wr_cond    *done_cv;     /* dispatcher waits here for completion    */
    uint64_t    job_seq;     /* bumped once per published job           */
    wr_pool_fn  fn;
    void       *fn_arg;
    uint32_t    n_parts;
    uint32_t    next_part;   /* shared claim counter                    */
    uint32_t    parts_done;  /* completed (not merely claimed) parts    */
    int         shutdown;
};

/* Claim-and-run loop shared by workers and the dispatcher.  Entered and
 * left with p->m HELD.  fn/arg/n_parts are re-read under the mutex at
 * every claim (stale-claim safety, header comment). */
static void wri_pool_drain(wr_pool *p)
{
    while (p->next_part < p->n_parts) {
        uint32_t   part    = p->next_part++;
        wr_pool_fn fn      = p->fn;
        void      *arg     = p->fn_arg;
        uint32_t   n_parts = p->n_parts;

        wr_mutex_unlock(p->m);
        fn(arg, part, n_parts);
        wr_mutex_lock(p->m);

        p->parts_done++;
        if (p->parts_done == p->n_parts)
            wr_cond_signal(p->done_cv);  /* only the dispatcher waits */
    }
}

static void wri_pool_worker(void *arg)
{
    wr_pool *p = arg;
    uint64_t seen_seq = 0;

    wr_mutex_lock(p->m);
    for (;;) {
        while (!p->shutdown && p->job_seq == seen_seq)
            wr_cond_wait(p->work_cv, p->m);
        if (p->shutdown)
            break;
        seen_seq = p->job_seq;
        wri_pool_drain(p);
    }
    wr_mutex_unlock(p->m);
}

/* Join workers (if any) and free everything.  Tolerates a partially
 * constructed pool: threads are only ever spawned after all primitives
 * exist, and only spawned threads are joined. */
static void wri_pool_teardown(wr_pool *p)
{
    if (p->n_spawned > 0) {
        wr_mutex_lock(p->m);
        p->shutdown = 1;
        wr_cond_broadcast(p->work_cv);
        wr_mutex_unlock(p->m);
        for (uint32_t i = 0; i < p->n_spawned; i++)
            (void)wr_thread_join(p->threads[i]);
    }
    if (p->done_cv != NULL)
        wr_cond_destroy(p->done_cv);
    if (p->work_cv != NULL)
        wr_cond_destroy(p->work_cv);
    if (p->m != NULL)
        wr_mutex_destroy(p->m);
    if (p->run_lock != NULL)
        wr_mutex_destroy(p->run_lock);
    free(p->threads);
    free(p);
}

wr_pool *wr_pool_create(uint32_t n_threads)
{
    if (n_threads < 1)
        n_threads = 1;
    if (n_threads > WR_MAX_WORKERS)
        n_threads = WR_MAX_WORKERS;

    wr_pool *p = calloc(1, sizeof *p);
    if (p == NULL)
        return NULL;
    p->n_threads = n_threads;

    p->run_lock = wr_mutex_create();
    p->m        = wr_mutex_create();
    p->work_cv  = wr_cond_create();
    p->done_cv  = wr_cond_create();
    if (p->run_lock == NULL || p->m == NULL ||
        p->work_cv == NULL || p->done_cv == NULL) {
        wri_pool_teardown(p);
        return NULL;
    }

    if (n_threads > 1) {
        p->threads = calloc(n_threads - 1u, sizeof *p->threads);
        if (p->threads == NULL) {
            wri_pool_teardown(p);
            return NULL;
        }
        for (uint32_t i = 0; i + 1u < n_threads; i++) {
            p->threads[i] = wr_thread_create(wri_pool_worker, p);
            if (p->threads[i] == NULL) {
                /* No degraded half-pool (pool.h): tear it all down. */
                wri_pool_teardown(p);
                return NULL;
            }
            p->n_spawned++;
        }
    }
    return p;
}

void wr_pool_destroy(wr_pool *p)
{
    if (p == NULL)
        return;
    wri_pool_teardown(p);
}

void wr_pool_run(wr_pool *p, wr_pool_fn fn, void *arg, uint32_t n_parts)
{
    WRI_ASSERT(p != NULL && fn != NULL);
    if (p == NULL || fn == NULL || n_parts == 0)
        return;

    wr_mutex_lock(p->run_lock);  /* rule 5 — single dispatcher */

    if (p->n_spawned == 0 || n_parts == 1) {
        /* Serial degenerate case (rule 4): no condvar traffic at all.
         * Also taken for 1-part jobs — waking the pool to watch the
         * dispatcher run the only part would be pure overhead. */
        for (uint32_t part = 0; part < n_parts; part++)
            fn(arg, part, n_parts);
        wr_mutex_unlock(p->run_lock);
        return;
    }

    wr_mutex_lock(p->m);
    p->fn         = fn;
    p->fn_arg     = arg;
    p->n_parts    = n_parts;
    p->next_part  = 0;
    p->parts_done = 0;
    p->job_seq++;
    wr_cond_broadcast(p->work_cv);

    wri_pool_drain(p);           /* rule 4 — the caller participates */

    while (p->parts_done < n_parts)
        wr_cond_wait(p->done_cv, p->m);
    wr_mutex_unlock(p->m);       /* rule 3 — all part writes visible */

    wr_mutex_unlock(p->run_lock);
}

uint32_t wr_pool_size(const wr_pool *p)
{
    return (p != NULL) ? p->n_threads : 0;
}
