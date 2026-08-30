/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * os_posix.c — POSIX implementation of the platform shim (platform.h).
 *
 * Linux/gcc side of the exactly-two-implementations contract.  Everything
 * here is a thin, honest wrapper: pthreads for threads/mutex/cond,
 * CLOCK_MONOTONIC for the microsecond clock, aligned_alloc for small
 * aligned buffers, anonymous private mmap for the large arenas, and a
 * read-only file mmap for the optional zero-copy GGUF weight path.
 *
 * Design rules carried over from the porting decisions:
 *   - plain blocking primitives only — no spinning, no affinity, no
 *     priority games (the origin OS's park/steal/rehome machinery is
 *     deliberately not reproduced here);
 *   - no logging from this layer; failures surface as NULL/-1 and the
 *     caller decides;
 *   - the monotonic clock is the ONLY time source in the library (no
 *     rdtsc/cntvct inline asm anywhere).
 */
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "platform/platform.h"

#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

/* --------------------------------------------------------------------------
 * File size
 * -------------------------------------------------------------------------- */

int wr_file_size(FILE *f, uint64_t *out_bytes)
{
    struct stat st;
    int fd;

    if (f == NULL || out_bytes == NULL)
        return -1;
    fd = fileno(f);
    if (fd < 0)
        return -1;
    if (fstat(fd, &st) != 0)
        return -1;
    if (st.st_size < 0)
        return -1;
    *out_bytes = (uint64_t)st.st_size;
    return 0;
}

/* --------------------------------------------------------------------------
 * Threads
 * -------------------------------------------------------------------------- */

struct wr_thread {
    pthread_t handle;
    void    (*fn)(void *arg);
    void     *arg;
};

/* pthread start routines return void*; the shim contract takes a void
 * function, so a trampoline bridges the signatures. */
static void *wri_thread_trampoline(void *param)
{
    struct wr_thread *t = (struct wr_thread *)param;
    t->fn(t->arg);
    return NULL;
}

wr_thread *wr_thread_create(void (*fn)(void *arg), void *arg)
{
    struct wr_thread *t;

    if (fn == NULL)
        return NULL;
    t = (struct wr_thread *)malloc(sizeof(*t));
    if (t == NULL)
        return NULL;
    t->fn  = fn;
    t->arg = arg;
    if (pthread_create(&t->handle, NULL, wri_thread_trampoline, t) != 0) {
        free(t);
        return NULL;
    }
    return t;
}

int wr_thread_join(wr_thread *t)
{
    int rc;

    if (t == NULL)
        return -1;
    rc = pthread_join(t->handle, NULL);
    free(t);
    return (rc == 0) ? 0 : -1;
}

/* --------------------------------------------------------------------------
 * Mutex + condition variable
 * -------------------------------------------------------------------------- */

struct wr_mutex {
    pthread_mutex_t m;
};

struct wr_cond {
    pthread_cond_t c;
};

wr_mutex *wr_mutex_create(void)
{
    struct wr_mutex *m = (struct wr_mutex *)malloc(sizeof(*m));

    if (m == NULL)
        return NULL;
    if (pthread_mutex_init(&m->m, NULL) != 0) {
        free(m);
        return NULL;
    }
    return m;
}

void wr_mutex_destroy(wr_mutex *m)
{
    if (m == NULL)
        return;
    (void)pthread_mutex_destroy(&m->m);
    free(m);
}

void wr_mutex_lock(wr_mutex *m)
{
    /* A default (non-recursive, non-robust) mutex lock fails only on
     * caller bugs the contract already forbids; the API is void. */
    (void)pthread_mutex_lock(&m->m);
}

void wr_mutex_unlock(wr_mutex *m)
{
    (void)pthread_mutex_unlock(&m->m);
}

wr_cond *wr_cond_create(void)
{
    struct wr_cond *c = (struct wr_cond *)malloc(sizeof(*c));

    if (c == NULL)
        return NULL;
    if (pthread_cond_init(&c->c, NULL) != 0) {
        free(c);
        return NULL;
    }
    return c;
}

void wr_cond_destroy(wr_cond *c)
{
    if (c == NULL)
        return;
    (void)pthread_cond_destroy(&c->c);
    free(c);
}

void wr_cond_wait(wr_cond *c, wr_mutex *m)
{
    /* Callers loop on their predicate (spurious wakeups permitted), so a
     * failed wait degenerates to a spurious wakeup. */
    (void)pthread_cond_wait(&c->c, &m->m);
}

void wr_cond_signal(wr_cond *c)
{
    (void)pthread_cond_signal(&c->c);
}

void wr_cond_broadcast(wr_cond *c)
{
    (void)pthread_cond_broadcast(&c->c);
}

/* --------------------------------------------------------------------------
 * Time
 * -------------------------------------------------------------------------- */

uint64_t wr_monotonic_us(void)
{
    struct timespec ts;

    /* CLOCK_MONOTONIC cannot fail on any supported host; the guard only
     * keeps the impossible branch defined. */
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000u;
}

/* --------------------------------------------------------------------------
 * CPU
 * -------------------------------------------------------------------------- */

uint32_t wr_cpu_count(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);

    return (n > 0) ? (uint32_t)n : 1u;
}

void wr_plat_pause(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    /* No spin hint on this architecture; a compiler barrier keeps the
     * surrounding spin loop honest. */
    __asm__ __volatile__("" ::: "memory");
#endif
}

/* --------------------------------------------------------------------------
 * Memory
 * -------------------------------------------------------------------------- */

void *wr_aligned_alloc(size_t align, size_t size)
{
    size_t rounded;

    if (align == 0 || (align & (align - 1)) != 0 || size == 0)
        return NULL;
    /* C11 aligned_alloc wants size to be a multiple of the alignment;
     * round up so callers do not have to care. */
    rounded = (size + align - 1) & ~(align - 1);
    if (rounded < size)
        return NULL;  /* size_t overflow */
    return aligned_alloc(align, rounded);
}

void wr_aligned_free(void *p)
{
    free(p);
}

void *wr_map_alloc(size_t size)
{
    void *p;

    if (size == 0)
        return NULL;
    p = mmap(NULL, size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? NULL : p;
}

void wr_map_free(void *p, size_t size)
{
    if (p == NULL || size == 0)
        return;
    (void)munmap(p, size);
}

/* --------------------------------------------------------------------------
 * Read-only file mapping
 *
 * mmap requires a page-aligned file offset; the shim aligns down
 * internally and hands the caller a pointer to exactly byte `offset`.
 * wr_file_unmap recovers the alignment slack from the pointer itself
 * (the mapping base is page-aligned, so ptr mod pagesize == slack).
 * -------------------------------------------------------------------------- */

void *wr_file_map_ro(FILE *f, uint64_t offset, size_t length)
{
    uint64_t aligned_off, delta;
    size_t   map_len;
    void    *base;
    long     page;
    int      fd;

    if (f == NULL || length == 0)
        return NULL;
    if (offset > (uint64_t)INT64_MAX)
        return NULL;
    fd = fileno(f);
    if (fd < 0)
        return NULL;

    page = sysconf(_SC_PAGESIZE);
    if (page <= 0 || ((uint64_t)page & ((uint64_t)page - 1u)) != 0)
        return NULL;

    aligned_off = offset & ~((uint64_t)page - 1u);
    delta       = offset - aligned_off;
    if (length > SIZE_MAX - (size_t)delta)
        return NULL;
    map_len = length + (size_t)delta;

    base = mmap(NULL, map_len, PROT_READ, MAP_PRIVATE, fd,
                (off_t)aligned_off);
    if (base == MAP_FAILED)
        return NULL;

    /* Weight bytes are read soon and mostly sequentially; the hint is
     * best-effort and its result deliberately ignored. */
    (void)posix_madvise(base, map_len, POSIX_MADV_WILLNEED);

    return (char *)base + delta;
}

void wr_file_unmap(void *p, size_t length)
{
    uintptr_t delta;
    long      page;

    if (p == NULL)
        return;
    page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
        return;
    delta = (uintptr_t)p & ((uintptr_t)page - 1u);
    if (length > SIZE_MAX - (size_t)delta)
        return;
    (void)munmap((char *)p - delta, length + (size_t)delta);
}
