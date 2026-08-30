/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * tensor.c — tensor descriptors, byte-plane access, views, and the
 * weight arena.
 *
 * Ported from the origin OS's kernel tensor layer.  What survives is
 * the descriptor lifecycle (block-aware sizing, create/destroy), the
 * bounds-checked byte read/write plane, the GROWABLE/valid_rows KV
 * cursor, and the 64 B-aligned bump arena — re-backed here by a single
 * wr_map_alloc slab instead of physically-contiguous page blocks
 * reserved at boot.  The origin's global tensor table with integer ids,
 * and its ownership / quota / accounting / reclaim machinery, were OS
 * policy and did not port: a wr_tensor is a plain descriptor embedded
 * in its owner (model weight slots, session KV and scratch).
 *
 * Storage ownership is carried by WR_TENSOR_ARENA (see tensor.h): set
 * means `data` references an arena or mapping and is never freed here;
 * clear means the bytes came from wr_aligned_alloc and wri_tensor_free
 * releases them.
 */

#include <stdlib.h>
#include <string.h>

#include "core/tensor.h"
#include "core/quant.h"

/* --------------------------------------------------------------------------
 * Descriptor init + standalone allocation
 * -------------------------------------------------------------------------- */

int wri_tensor_init(wr_tensor *t, wr_dtype dtype, uint32_t ndim,
                    const uint32_t *shape, uint16_t flags, void *backing)
{
    if (t == NULL || shape == NULL)
        return WR_ERR_INVAL;
    if (ndim == 0 || ndim > WR_TENSOR_MAX_DIMS)
        return WR_ERR_INVAL;
    if ((uint32_t)dtype >= (uint32_t)WR_DTYPE_COUNT)
        return WR_ERR_INVAL;

    /* Element count with a per-step overflow check. */
    uint64_t total_elems = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        if (shape[i] == 0)
            return WR_ERR_INVAL;
        if (total_elems > UINT64_MAX / shape[i])
            return WR_ERR_INVAL;
        total_elems *= shape[i];
    }

    /* Conservative guard so the byte sizing below cannot overflow: the
     * widest supported dtype is 4 bytes per element (F32) and every
     * block format is denser than one byte per element plus one block
     * of rounding.  2^61 elements is far beyond anything allocatable. */
    if (total_elems > UINT64_MAX / 8)
        return WR_ERR_INVAL;

    /* Block-aware sizing.  wri_dtype_bytes_for_count rounds partial
     * blocks up for ALL five block-quantized dtypes and multiplies by
     * the fixed stride otherwise.  (The origin engine once routed only
     * Q4_0 through the block path, which made create refuse every other
     * quantized dtype the compute paths could already consume; the
     * K-quant super-blocks — 256 elements each — size through the same
     * calculator here.) */
    uint64_t size_bytes = wri_dtype_bytes_for_count(dtype, total_elems);
    if (size_bytes == 0)
        return WR_ERR_INVAL;

    memset(t, 0, sizeof(*t));
    for (uint32_t i = 0; i < ndim; i++)
        t->shape[i] = shape[i];
    t->ndim       = (uint8_t)ndim;
    t->dtype      = (uint8_t)dtype;
    t->size_bytes = size_bytes;
    t->valid_rows = 0;

    if (backing != NULL) {
        /* Arena / mapping-backed: reference only; never freed here. */
        t->data  = backing;
        t->flags = (uint16_t)(flags | WR_TENSOR_ARENA);
        return WR_OK;
    }

#if SIZE_MAX < UINT64_MAX
    if (size_bytes > (uint64_t)SIZE_MAX)
        return WR_ERR_NOMEM;
#endif

    void *data = wr_aligned_alloc(64, (size_t)size_bytes);
    if (data == NULL)
        return WR_ERR_NOMEM;    /* *t stays zeroed (empty slot) */
    memset(data, 0, (size_t)size_bytes);

    t->data  = data;
    t->flags = (uint16_t)(flags & (uint16_t)~WR_TENSOR_ARENA);
    return WR_OK;
}

void wri_tensor_free(wr_tensor *t)
{
    if (t == NULL)
        return;
    if (t->data != NULL && !(t->flags & WR_TENSOR_ARENA))
        wr_aligned_free(t->data);
    memset(t, 0, sizeof(*t));
}

uint64_t wri_tensor_total_elements(const wr_tensor *t)
{
    if (t == NULL || t->ndim == 0)
        return 0;
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->ndim; i++)
        n *= t->shape[i];
    return n;
}

/* --------------------------------------------------------------------------
 * KV cursor
 * -------------------------------------------------------------------------- */

int wri_tensor_set_valid_rows(wr_tensor *t, uint32_t new_rows)
{
    if (t == NULL)
        return WR_ERR_INVAL;
    if (!(t->flags & WR_TENSOR_GROWABLE))
        return WR_ERR_INVAL;
    if (new_rows > t->shape[0])
        return WR_ERR_INVAL;    /* never clamped */
    t->valid_rows = new_rows;
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Byte-plane access
 *
 * Whole-range bounds check up front — a copy either happens completely
 * or not at all.  `size > size_bytes` is tested first so the subtraction
 * in the offset check cannot wrap.
 * -------------------------------------------------------------------------- */

int wri_tensor_write(wr_tensor *t, uint64_t offset, const void *data,
                     uint64_t size)
{
    if (t == NULL || t->data == NULL || data == NULL || size == 0)
        return WR_ERR_INVAL;
    if (size > t->size_bytes || offset > t->size_bytes - size)
        return WR_ERR_INVAL;
    memcpy((uint8_t *)t->data + offset, data, (size_t)size);
    return WR_OK;
}

int wri_tensor_read(const wr_tensor *t, uint64_t offset, void *buf,
                    uint64_t size)
{
    if (t == NULL || t->data == NULL || buf == NULL || size == 0)
        return WR_ERR_INVAL;
    if (size > t->size_bytes || offset > t->size_bytes - size)
        return WR_ERR_INVAL;
    memcpy(buf, (const uint8_t *)t->data + offset, (size_t)size);
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Views
 * -------------------------------------------------------------------------- */

int wri_tensor_view_2d(const wr_tensor *src, uint32_t rows, uint32_t cols,
                       wr_tensor *out_view)
{
    if (src == NULL || out_view == NULL || src->data == NULL)
        return WR_ERR_INVAL;
    if (rows == 0 || cols == 0)
        return WR_ERR_INVAL;
    if ((uint64_t)rows * (uint64_t)cols != wri_tensor_total_elements(src))
        return WR_ERR_INVAL;

    /* Non-owning reshape of src's storage.  The source descriptor is
     * never touched: the tied-LM-head [vocab, hidden] → [hidden, vocab]
     * flip happens on this per-call copy, not on the shared model slot
     * (the origin reshaped the shared descriptor in place, which raced
     * across sessions). */
    memset(out_view, 0, sizeof(*out_view));
    out_view->data       = src->data;
    out_view->size_bytes = src->size_bytes;
    out_view->shape[0]   = rows;
    out_view->shape[1]   = cols;
    out_view->ndim       = 2;
    out_view->dtype      = src->dtype;
    out_view->flags      = (uint16_t)(src->flags | WR_TENSOR_ARENA);
    /* Carried verbatim; meaningful only under WR_TENSOR_GROWABLE, where
     * a view is only coherent if it keeps shape[0] (nothing in the
     * library reshapes a growable tensor). */
    out_view->valid_rows = src->valid_rows;
    return WR_OK;
}

/* --------------------------------------------------------------------------
 * Weight arena
 *
 * One bump allocator over a single wr_map_alloc slab (zero-filled,
 * page-aligned).  The origin reserved its arena blocks at early boot to
 * beat physical-memory fragmentation; a hosted process has no such
 * constraint, so the slab is sized once at model load instead.  Bumps
 * advance in 64-byte multiples off a page-aligned base, so every
 * returned pointer is 64 B aligned.  No free list: the slab is released
 * wholesale with the model.
 * -------------------------------------------------------------------------- */

#define WRI_ARENA_PAGE 4096u    /* slab-size rounding granularity */

wri_arena *wri_arena_create(size_t bytes)
{
    if (bytes > SIZE_MAX - (WRI_ARENA_PAGE - 1u))
        return NULL;
    size_t size = (bytes + (WRI_ARENA_PAGE - 1u)) &
                  ~(size_t)(WRI_ARENA_PAGE - 1u);
    if (size == 0)
        size = WRI_ARENA_PAGE;  /* a zero request still yields a usable arena */

    wri_arena *a = malloc(sizeof(*a));
    if (a == NULL)
        return NULL;
    a->base = wr_map_alloc(size);
    if (a->base == NULL) {
        free(a);
        return NULL;
    }
    a->size = size;
    a->used = 0;
    return a;
}

void wri_arena_destroy(wri_arena *a)
{
    if (a == NULL)
        return;
    wr_map_free(a->base, a->size);
    free(a);
}

void *wri_arena_alloc(wri_arena *a, size_t bytes)
{
    if (a == NULL)
        return NULL;
    size_t aligned = (bytes + 63u) & ~(size_t)63u;
    if (aligned < bytes)
        return NULL;            /* align-up wrapped */
    if (aligned > a->size - a->used)
        return NULL;            /* slab exhausted — loader sizing bug */
    void *out = a->base + a->used;
    a->used += aligned;
    return out;
}
