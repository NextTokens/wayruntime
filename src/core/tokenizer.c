/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * tokenizer.c — public wr_tokenizer facade over the internal BPE engine.
 *
 * A wr_tokenizer owns a full copy of the vocab (wr_bpe copies everything
 * it keeps), so it never depends on the model's GGUF mapping staying
 * open.  Construction needs the GGUF metadata once: when the model
 * retained its container (use_mmap) that handle is reused read-only;
 * otherwise the file is reopened from the model's stored path just long
 * enough to parse the header — weights are never touched.
 *
 * Lifetime: each live tokenizer holds one tokenizer_ref on its model
 * (model->lock guards the count).  wr_model_free refuses while callers'
 * tokenizers are alive, matching the public header's contract.
 */
#include <stdlib.h>
#include <string.h>

#include "core/internal.h"
#include "core/bpe.h"
#include "core/gguf.h"

struct wr_tokenizer {
    wr_model *model;   /* refcount holder; never dereferenced for vocab */
    wr_bpe    bpe;
};

wr_status wr_tokenizer_from_model(const wr_model *m, wr_tokenizer **out)
{
    if (out)
        *out = NULL;
    if (!m || !out)
        return WR_ERR_INVAL;

    /* The public parameter is const (query-shaped API); the refcount
     * bump below is the one interior mutation and it is lock-guarded. */
    wr_model *mm = (wr_model *)(uintptr_t)m;

    wr_tokenizer *t = calloc(1, sizeof *t);
    if (!t)
        return WR_ERR_NOMEM;

    int rc;
    if (mm->gguf) {
        rc = wri_bpe_init(&t->bpe, mm->gguf);
    } else if (mm->gguf_path) {
        wr_gguf *g = NULL;
        rc = wri_gguf_open(mm->gguf_path, &g);
        if (rc == WR_OK) {
            rc = wri_bpe_init(&t->bpe, g);
            wri_gguf_close(g);
        }
    } else {
        rc = WR_ERR_STATE;
    }
    if (rc != WR_OK) {
        free(t);
        return (wr_status)rc;
    }
    /* The vocab was re-read from the file the model was loaded from; a
     * replaced file with more tokens than the model has embedding rows
     * cannot belong to these weights.  (Fewer is normal: embedding
     * tables are commonly padded past the token list.) */
    if (t->bpe.vocab_size > mm->vocab_size) {
        wri_log_msg(0, "tokenizer: vocab of %u tokens exceeds the model's %u "
                "embedding rows - file changed since load? (WR_ERR_STATE)",
                t->bpe.vocab_size, mm->vocab_size);
        wri_bpe_free(&t->bpe);
        free(t);
        return WR_ERR_STATE;
    }

    t->model = mm;
    wr_mutex_lock(mm->lock);
    mm->tokenizer_refs++;
    wr_mutex_unlock(mm->lock);

    *out = t;
    return WR_OK;
}

void wr_tokenizer_free(wr_tokenizer *t)
{
    if (!t)
        return;
    if (t->model) {
        wr_mutex_lock(t->model->lock);
        if (t->model->tokenizer_refs)
            t->model->tokenizer_refs--;
        wr_mutex_unlock(t->model->lock);
    }
    wri_bpe_free(&t->bpe);
    free(t);
}

int wr_tokenize_ex(const wr_tokenizer *t, const char *text,
                   uint32_t *ids, int max_ids, uint32_t flags)
{
    if (!t || !text)
        return WR_ERR_INVAL;
    return wri_bpe_encode(&t->bpe, text, ids, max_ids, flags);
}

int wr_tokenize(const wr_tokenizer *t, const char *text,
                uint32_t *ids, int max_ids)
{
    return wr_tokenize_ex(t, text, ids, max_ids, 0);
}

int wr_detokenize(const wr_tokenizer *t, const uint32_t *ids, int n,
                  char *buf, int max_bytes)
{
    if (!t || (!ids && n > 0) || n < 0)
        return WR_ERR_INVAL;
    return wri_bpe_decode(&t->bpe, ids, n, buf, max_bytes);
}

int wr_token_piece(const wr_tokenizer *t, uint32_t id,
                   char *buf, int max_bytes)
{
    if (!t)
        return WR_ERR_INVAL;
    return wri_bpe_token_piece(&t->bpe, id, buf, max_bytes);
}

int wr_token_id(const wr_tokenizer *t, const char *token_text)
{
    if (!t || !token_text)
        return -1;
    return wri_bpe_token_id(&t->bpe, token_text);
}

int32_t wr_token_eos(const wr_tokenizer *t)
{
    return t ? t->bpe.eos_token_id : -1;
}

int32_t wr_token_bos(const wr_tokenizer *t)
{
    return t ? t->bpe.bos_token_id : -1;
}
