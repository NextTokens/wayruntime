/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * os_win.c — Windows (mingw-w64) implementation of the platform shim
 * (platform.h).
 *
 * Windows side of the exactly-two-implementations contract: CreateThread
 * for threads, SRWLOCK + CONDITION_VARIABLE for the blocking primitives,
 * QueryPerformanceCounter for the microsecond clock, _aligned_malloc for
 * small aligned buffers, VirtualAlloc for the large arenas, and
 * CreateFileMapping/MapViewOfFile for the optional zero-copy GGUF weight
 * path.
 *
 * Design rules carried over from the porting decisions:
 *   - plain blocking primitives only — no spinning, no affinity, no
 *     priority games (the origin OS's park/steal/rehome machinery is
 *     deliberately not reproduced here);
 *   - no logging from this layer; failures surface as NULL/-1 and the
 *     caller decides;
 *   - the monotonic clock is the ONLY time source in the library (no
 *     rdtsc/cntvct inline asm anywhere).
 *
 * LLP64 note: unsigned long is 32-bit here; every 64-bit quantity in
 * this file is an explicit (u)int64_t.
 */
#if !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0601)
#  undef  _WIN32_WINNT
#  define _WIN32_WINNT 0x0601  /* SRWLOCK/CONDITION_VARIABLE need >= Vista */
#endif
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <io.h>
#include <malloc.h>
#include <stdlib.h>

#include "platform/platform.h"

/* --------------------------------------------------------------------------
 * File size
 * -------------------------------------------------------------------------- */

static HANDLE wri_file_handle(FILE *f)
{
    int fd;

    if (f == NULL)
        return INVALID_HANDLE_VALUE;
    fd = _fileno(f);
    if (fd < 0)
        return INVALID_HANDLE_VALUE;
    return (HANDLE)_get_osfhandle(fd);
}

int wr_file_size(FILE *f, uint64_t *out_bytes)
{
    LARGE_INTEGER sz;
    HANDLE h;

    if (out_bytes == NULL)
        return -1;
    h = wri_file_handle(f);
    if (h == INVALID_HANDLE_VALUE)
        return -1;
    if (!GetFileSizeEx(h, &sz))
        return -1;
    if (sz.QuadPart < 0)
        return -1;
    *out_bytes = (uint64_t)sz.QuadPart;
    return 0;
}

/* --------------------------------------------------------------------------
 * Threads
 * -------------------------------------------------------------------------- */

struct wr_thread {
    HANDLE handle;
    void (*fn)(void *arg);
    void  *arg;
};

/* Win32 thread procs return DWORD; the shim contract takes a void
 * function, so a trampoline bridges the signatures. */
static DWORD WINAPI wri_thread_trampoline(LPVOID param)
{
    struct wr_thread *t = (struct wr_thread *)param;

    t->fn(t->arg);
    return 0;
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
    t->handle = CreateThread(NULL, 0, wri_thread_trampoline, t, 0, NULL);
    if (t->handle == NULL) {
        free(t);
        return NULL;
    }
    return t;
}

int wr_thread_join(wr_thread *t)
{
    DWORD rc;

    if (t == NULL)
        return -1;
    rc = WaitForSingleObject(t->handle, INFINITE);
    (void)CloseHandle(t->handle);
    free(t);
    return (rc == WAIT_OBJECT_0) ? 0 : -1;
}

/* --------------------------------------------------------------------------
 * Mutex + condition variable
 *
 * SRW locks and condition variables need no teardown call — destroy is
 * just the heap release.  All lock acquisitions are exclusive (the
 * condvar sleep below passes flags = 0 to match).
 * -------------------------------------------------------------------------- */

struct wr_mutex {
    SRWLOCK lock;
};

struct wr_cond {
    CONDITION_VARIABLE cv;
};

wr_mutex *wr_mutex_create(void)
{
    struct wr_mutex *m = (struct wr_mutex *)malloc(sizeof(*m));

    if (m == NULL)
        return NULL;
    InitializeSRWLock(&m->lock);
    return m;
}

void wr_mutex_destroy(wr_mutex *m)
{
    free(m);
}

void wr_mutex_lock(wr_mutex *m)
{
    AcquireSRWLockExclusive(&m->lock);
}

void wr_mutex_unlock(wr_mutex *m)
{
    ReleaseSRWLockExclusive(&m->lock);
}

wr_cond *wr_cond_create(void)
{
    struct wr_cond *c = (struct wr_cond *)malloc(sizeof(*c));

    if (c == NULL)
        return NULL;
    InitializeConditionVariable(&c->cv);
    return c;
}

void wr_cond_destroy(wr_cond *c)
{
    free(c);
}

void wr_cond_wait(wr_cond *c, wr_mutex *m)
{
    /* INFINITE timeout: the only documented failure is a timeout, which
     * cannot occur.  Callers loop on their predicate anyway (spurious
     * wakeups permitted), so a failed sleep degenerates to one. */
    (void)SleepConditionVariableSRW(&c->cv, &m->lock, INFINITE, 0);
}

void wr_cond_signal(wr_cond *c)
{
    WakeConditionVariable(&c->cv);
}

void wr_cond_broadcast(wr_cond *c)
{
    WakeAllConditionVariable(&c->cv);
}

/* --------------------------------------------------------------------------
 * Time
 * -------------------------------------------------------------------------- */

/* QPC frequency is fixed at boot; probe once and cache.  Concurrent
 * first calls may both probe — same value, benign race. */
static uint64_t wri_g_qpc_freq;

uint64_t wr_monotonic_us(void)
{
    LARGE_INTEGER now;
    uint64_t freq, ticks;

    freq = __atomic_load_n(&wri_g_qpc_freq, __ATOMIC_ACQUIRE);
    if (freq == 0) {
        LARGE_INTEGER li;

        /* Cannot fail on any supported Windows; the <= 0 guard only
         * keeps the arithmetic below defined. */
        (void)QueryPerformanceFrequency(&li);
        freq = (li.QuadPart > 0) ? (uint64_t)li.QuadPart : 1u;
        __atomic_store_n(&wri_g_qpc_freq, freq, __ATOMIC_RELEASE);
    }

    (void)QueryPerformanceCounter(&now);
    ticks = (uint64_t)now.QuadPart;

    /* Split the conversion so ticks * 1e6 cannot overflow 64 bits over
     * any realistic uptime. */
    return (ticks / freq) * 1000000ull + ((ticks % freq) * 1000000ull) / freq;
}

/* --------------------------------------------------------------------------
 * CPU
 * -------------------------------------------------------------------------- */

uint32_t wr_cpu_count(void)
{
    SYSTEM_INFO si;

    GetSystemInfo(&si);
    return (si.dwNumberOfProcessors > 0) ? (uint32_t)si.dwNumberOfProcessors
                                         : 1u;
}

void wr_plat_pause(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
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
    if (align == 0 || (align & (align - 1)) != 0 || size == 0)
        return NULL;
    return _aligned_malloc(size, align);
}

void wr_aligned_free(void *p)
{
    _aligned_free(p);
}

void *wr_map_alloc(size_t size)
{
    if (size == 0)
        return NULL;
    /* MEM_COMMIT pages are zero-filled on first touch, matching the
     * anonymous-mmap semantics of the POSIX side. */
    return VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

void wr_map_free(void *p, size_t size)
{
    (void)size;  /* MEM_RELEASE requires dwSize == 0 */
    if (p == NULL)
        return;
    (void)VirtualFree(p, 0, MEM_RELEASE);
}

/* --------------------------------------------------------------------------
 * Read-only file mapping
 *
 * MapViewOfFile requires the file offset to be aligned to the system
 * allocation granularity (64 KiB on every shipping Windows); the shim
 * aligns down internally and hands the caller a pointer to exactly byte
 * `offset`.  wr_file_unmap recovers the alignment slack from the pointer
 * itself (view bases are granularity-aligned, so ptr mod granularity ==
 * slack).
 * -------------------------------------------------------------------------- */

/* Allocation granularity is fixed at boot; probe once and cache.  A
 * non-power-of-two report would break the mask math, so it fails the
 * mapping (callers fall back to streamed reads) instead of guessing. */
static uint32_t wri_g_alloc_gran;

static uint32_t wri_alloc_granularity(void)
{
    uint32_t g = __atomic_load_n(&wri_g_alloc_gran, __ATOMIC_ACQUIRE);

    if (g == 0) {
        SYSTEM_INFO si;

        GetSystemInfo(&si);
        g = (uint32_t)si.dwAllocationGranularity;
        if (g == 0 || (g & (g - 1u)) != 0)
            return 0;
        __atomic_store_n(&wri_g_alloc_gran, g, __ATOMIC_RELEASE);
    }
    return g;
}

void *wr_file_map_ro(FILE *f, uint64_t offset, size_t length)
{
    uint64_t aligned_off, delta;
    uint32_t gran;
    size_t   map_len;
    HANDLE   h, mapping;
    void    *view;

    if (length == 0)
        return NULL;
    if (offset > (uint64_t)INT64_MAX)
        return NULL;
    h = wri_file_handle(f);
    if (h == INVALID_HANDLE_VALUE)
        return NULL;

    gran = wri_alloc_granularity();
    if (gran == 0)
        return NULL;

    aligned_off = offset & ~((uint64_t)gran - 1u);
    delta       = offset - aligned_off;
    if (length > SIZE_MAX - (size_t)delta)
        return NULL;
    map_len = length + (size_t)delta;

    /* Max size 0/0 = current file size; PAGE_READONLY fails on an empty
     * file, which is correct — there is nothing to map. */
    mapping = CreateFileMappingA(h, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mapping == NULL)
        return NULL;

    view = MapViewOfFile(mapping, FILE_MAP_READ,
                         (DWORD)(aligned_off >> 32),
                         (DWORD)(aligned_off & 0xFFFFFFFFu),
                         map_len);
    /* The view holds its own reference to the mapping object. */
    (void)CloseHandle(mapping);
    if (view == NULL)
        return NULL;
    return (char *)view + delta;
}

void wr_file_unmap(void *p, size_t length)
{
    uintptr_t delta;
    uint32_t  gran;

    (void)length;  /* Windows unmaps whole views by base address */
    if (p == NULL)
        return;
    gran = wri_alloc_granularity();
    if (gran == 0)
        return;
    delta = (uintptr_t)p & ((uintptr_t)gran - 1u);
    (void)UnmapViewOfFile((char *)p - delta);
}
