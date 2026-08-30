/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * tensor.h — tensor lifecycle, the weight arena, and view descriptors.
 *
 * No global tensor table, no ids: a wr_tensor is a plain descriptor the
 * owner embeds (model weight slots, session KV/scratch).  Ownership of
 * `data` is tracked by WR_TENSOR_ARENA: set = the bytes live in an arena
 * or mapping and wri_tensor_free must not touch them; clear = the bytes
 * were individually allocated and wri_tensor_free releases them.
 */
#ifndef WR_TENSOR_H
#define WR_TENSOR_H

#include "core/internal.h"

/* --------------------------------------------------------------------------
 * Descriptor init + standalone allocation
 * -------------------------------------------------------------------------- */

/* Fill *t for `ndim` dims of `shape`, computing size_bytes with
 * block-aware rounding.  If `backing` is non-NULL the tensor references
 * it (flags gets WR_TENSOR_ARENA); if NULL, allocates 64 B-aligned
 * zeroed storage owned by the tensor.  Errors: WR_ERR_INVAL (ndim 0 or
 * > WR_TENSOR_MAX_DIMS, zero dim, dtype out of range), WR_ERR_NOMEM. */
int wri_tensor_init(wr_tensor *t, wr_dtype dtype, uint32_t ndim,
                    const uint32_t *shape, uint16_t flags, void *backing);

/* Release owned storage (no-op for WR_TENSOR_ARENA data) and zero *t. */
void wri_tensor_free(wr_tensor *t);

/* Product of shape[0..ndim). */
uint64_t wri_tensor_total_elements(const wr_tensor *t);

/* KV cursor.  Requires WR_TENSOR_GROWABLE and new_rows <= shape[0];
 * WR_ERR_INVAL otherwise (never clamped). */
int wri_tensor_set_valid_rows(wr_tensor *t, uint32_t new_rows);

/* Bounds-checked byte-plane access into t->data.  The whole range
 * [offset, offset + size) must lie inside size_bytes — WR_ERR_INVAL
 * otherwise, and the copy either happens completely or not at all.
 * size == 0 is WR_ERR_INVAL. */
int wri_tensor_write(wr_tensor *t, uint64_t offset, const void *data,
                     uint64_t size);
int wri_tensor_read(const wr_tensor *t, uint64_t offset, void *buf,
                    uint64_t size);

/* --------------------------------------------------------------------------
 * Views
 *
 * A view is a wr_tensor whose data/size_bytes reference another tensor's
 * storage with a different shape.  Views are per-call, stack-allocated,
 * and never freed.  THE load-bearing use: the tied LM head is stored
 * [vocab, hidden] (GGML layout, shared with the embedding); each decode
 * call builds a private [hidden, vocab]-semantics view for the logits
 * matmul instead of mutating the shared descriptor's shape (the origin
 * reshaped in place, which races when several sessions share a model).
 * -------------------------------------------------------------------------- */

/* Build a 2-D view over src's storage with dims [rows, cols].  Dtype and
 * flags (incl. WR_TENSOR_GGML_WEIGHT) carry over; rows*cols must equal
 * src's element count (WR_ERR_INVAL otherwise).  out_view->flags gets
 * WR_TENSOR_ARENA (a view never owns bytes). */
int wri_tensor_view_2d(const wr_tensor *src, uint32_t rows, uint32_t cols,
                       wr_tensor *out_view);

/* --------------------------------------------------------------------------
 * Weight arena
 *
 * One 64 B-aligned bump allocator over a single wr_map_alloc slab, sized
 * once at model load.  Allocation never fails piecemeal mid-load: the
 * loader computes the total up front, creates the arena, then bumps.
 * No free list — the arena is released wholesale with the model.
 * -------------------------------------------------------------------------- */

typedef struct wri_arena {
    uint8_t *base;
    size_t   size;
    size_t   used;
} wri_arena;

/* Reserve a slab of `bytes` (rounded up to page size).  NULL on failure. */
wri_arena *wri_arena_create(size_t bytes);

/* Release the slab and the arena object.  Every tensor pointing into it
 * must already be dead. */
void wri_arena_destroy(wri_arena *a);

/* Bump-allocate `bytes` aligned to 64.  Returns NULL when the slab is
 * exhausted (loader sizing bug — caller turns it into WR_ERR_INTERNAL,
 * it must never be reachable from input data). */
void *wri_arena_alloc(wri_arena *a, size_t bytes);

#endif /* WR_TENSOR_H */
