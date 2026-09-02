/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * engine.c — engine lifecycle: create/destroy, the process-wide anchor,
 * SIMD re-selection, counters snapshot, version/status strings and the
 * library log helper.
 *
 * v1 is single-engine-per-process (public contract).  Everything the
 * compute path needs without a parameter trail — the worker pool, the
 * SIMD dispatch tables, the counters — hangs off wri_g_engine; a second
 * wr_engine_create while one exists is refused with WR_ERR_STATE rather
 * than silently sharing state.
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/internal.h"

/* The process engine anchor (declared in internal.h).  Set on successful
 * create, cleared on destroy.  Create/destroy are externally
 * synchronized (public thread contract); concurrent readers — wri_log
 * from worker threads, query calls — see either NULL or a fully
 * constructed engine because the pointer is published only after
 * construction completes and cleared only after teardown of everything
 * that could log through it. */
wr_engine *wri_g_engine = NULL;

/* --------------------------------------------------------------------------
 * Version + status strings
 * -------------------------------------------------------------------------- */

#define WRI_STR_(x) #x
#define WRI_STR(x)  WRI_STR_(x)

const char *wr_version(void)
{
    return WRI_STR(WR_VERSION_MAJOR) "."
           WRI_STR(WR_VERSION_MINOR) "."
           WRI_STR(WR_VERSION_PATCH);
}

/* Buffers the library hands to the caller (wr_generate_result.text) are
 * plain malloc allocations; wr_free must stay free()-compatible. */
void wr_free(void *ptr)
{
    free(ptr);
}

const char *wr_status_str(wr_status s)
{
    switch (s) {
    case WR_OK:              return "WR_OK";
    case WR_ERR_IO:          return "WR_ERR_IO";
    case WR_ERR_FORMAT:      return "WR_ERR_FORMAT";
    case WR_ERR_UNSUPPORTED: return "WR_ERR_UNSUPPORTED";
    case WR_ERR_INVAL:       return "WR_ERR_INVAL";
    case WR_ERR_NOMEM:       return "WR_ERR_NOMEM";
    case WR_ERR_CTX_FULL:    return "WR_ERR_CTX_FULL";
    case WR_ERR_STATE:       return "WR_ERR_STATE";
    case WR_ERR_LIMIT:       return "WR_ERR_LIMIT";
    case WR_ERR_BUSY:        return "WR_ERR_BUSY";
    case WR_ERR_INTERNAL:    return "WR_ERR_INTERNAL";
    }
    return "WR_ERR_UNKNOWN";
}

/* --------------------------------------------------------------------------
 * Logging
 * -------------------------------------------------------------------------- */

/* Format into a fixed line, then hand it to the engine's sink (or stderr
 * when no engine / no callback).  Callable from any thread, including
 * pool workers: the engine pointer is snapshotted once, and both sinks
 * are line-atomic.  Long messages truncate at the buffer — every caller
 * in this library logs single diagnostic lines. */
void wri_log_msg(int level, const char *fmt, ...)
{
    char    line[512];
    va_list ap;

    va_start(ap, fmt);
    (void)vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);

    wr_engine *e = wri_g_engine;
    if (e != NULL && e->cfg.log_fn != NULL) {
        e->cfg.log_fn(level, line, e->cfg.log_user);
        return;
    }

    static const char *const tag[4] = { "error", "warn", "info", "debug" };
    int lv = level;
    if (lv < 0)
        lv = 0;
    if (lv > 3)
        lv = 3;
    fprintf(stderr, "wayruntime %s: %s\n", tag[lv], line);
}

/* --------------------------------------------------------------------------
 * Engine lifecycle
 * -------------------------------------------------------------------------- */

/* Runtime little-endian probe.  internal.h already refuses big-endian
 * builds at compile time when the compiler reports byte order; this
 * backstop keeps the public promise (WR_ERR_UNSUPPORTED) honest even
 * for a toolchain that slipped past that gate. */
static int wri_host_is_le(void)
{
    const uint16_t probe = 0x0102;
    uint8_t first;
    memcpy(&first, &probe, 1);
    return first == 0x02;
}

wr_status wr_engine_create(const wr_engine_config *cfg, wr_engine **out)
{
    if (out == NULL)
        return WR_ERR_INVAL;
    *out = NULL;

    if (wri_g_engine != NULL) {
        wri_log_msg(0, "engine_create: an engine already exists "
                   "(v1 is single-engine per process)");
        return WR_ERR_STATE;
    }
    if (!wri_host_is_le())
        return WR_ERR_UNSUPPORTED;

    /* Counters are engine-scoped in the public contract; the storage is
     * process-global, so a new engine starts them from zero. */
    wri_counters_reset();

    wr_engine *e = calloc(1, sizeof *e);
    if (e == NULL)
        return WR_ERR_NOMEM;
    if (cfg != NULL)
        e->cfg = *cfg;

    /* Resolve the thread count: 0 means "online CPU count"; every value
     * lands in [1, WR_MAX_WORKERS] (the pool clamps identically —
     * bit-exactness never depends on worker count, see pool.h). */
    const uint32_t requested = e->cfg.n_threads;
    uint32_t n = requested;
    if (n == 0)
        n = wr_cpu_count();
    if (n < 1)
        n = 1;
    if (n > WR_MAX_WORKERS)
        n = WR_MAX_WORKERS;

    e->pool = wr_pool_create(n);
    if (e->pool == NULL) {
        free(e);
        wri_log_msg(0, "engine_create: worker pool creation failed "
                   "(%u thread(s))", n);
        return WR_ERR_NOMEM;
    }
    e->cfg.n_threads = wr_pool_size(e->pool);

    /* Publish before dispatch init so any line it logs reaches the
     * caller's sink.  Nothing can fail past this point. */
    wri_g_engine = e;
    wri_simd_init(e->cfg.force_scalar, e->cfg.prefer_avx2);

    if (requested > (uint32_t)WR_MAX_WORKERS)
        wri_log_msg(1, "engine_create: n_threads %u clamped to the compiled "
                   "cap %u", requested, (unsigned)WR_MAX_WORKERS);
    wri_log_msg(2, "engine ready: version %s, %u worker thread(s), "
               "simd variant %d",
            wr_version(), e->cfg.n_threads, wri_simd_matmul_variant());

    *out = e;
    return WR_OK;
}

void wr_engine_destroy(wr_engine *e)
{
    if (e == NULL)
        return;
    if (e != wri_g_engine) {
        /* Not the live engine: refusing is the only safe move (freeing
         * an unknown pointer could tear down a stranger's memory). */
        wri_log_msg(0, "engine_destroy: not the live engine (refused)");
        return;
    }
    if (e->models_live != 0) {
        wri_log_msg(0, "engine_destroy refused: %u model(s) still live "
                   "(engine leaked; free the models first)",
                e->models_live);
        return;
    }
    wr_pool_destroy(e->pool);
    wri_g_engine = NULL;
    free(e);
}

int wr_engine_simd_variant(const wr_engine *e)
{
    if (e == NULL || e != wri_g_engine)
        return WR_ERR_INVAL;
    return wri_simd_matmul_variant();
}

int wr_engine_thread_count(const wr_engine *e)
{
    if (e == NULL)
        return WR_ERR_INVAL;
    if (e != wri_g_engine)
        return WR_ERR_STATE;
    return (int)e->cfg.n_threads;
}

int wr_engine_set_simd(wr_engine *e, int force_scalar, int prefer_avx2)
{
    if (e == NULL)
        return WR_ERR_INVAL;
    if (e != wri_g_engine)
        return WR_ERR_STATE;

    /* Republishers are serialized against each other (a spin gate: the
     * critical section is a handful of stores); decoders are NOT held
     * up — kernels.c publishes each table entry with a release store and
     * every op snapshots exactly one kernel per call, so an in-flight op
     * finishes on the variant it started with.  Without the gate two
     * concurrent callers could interleave their stores and leave the
     * table half from each. */
    static int gate = 0;
    while (__atomic_exchange_n(&gate, 1, __ATOMIC_ACQUIRE))
        wr_plat_pause();
    e->cfg.force_scalar = (force_scalar != 0);
    e->cfg.prefer_avx2  = (prefer_avx2 != 0);
    wri_simd_init(force_scalar, prefer_avx2);
    int bound = wri_simd_matmul_variant();
    __atomic_store_n(&gate, 0, __ATOMIC_RELEASE);
    return bound;
}

wr_status wr_engine_counters(const wr_engine *e,
                             uint64_t out[WR_COUNTER_COUNT])
{
    if (e == NULL || out == NULL)
        return WR_ERR_INVAL;
    if (e != wri_g_engine)
        return WR_ERR_STATE;
    wri_counters_snapshot(out);
    return WR_OK;
}
