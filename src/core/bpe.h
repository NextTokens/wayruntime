/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * bpe.h — BPE tokenizer (internal engine behind the public wr_tokenizer).
 *
 * Three alphabet modes, auto-detected from GGUF metadata
 * (tokenizer.ggml.model first, vocab-scan heuristic only as fallback):
 *
 *   byte_level  GPT-2/Qwen alphabet — the byte→unicode table is applied
 *               on ENCODE and reversed on DECODE (Ġ=space, Ċ=newline...).
 *               Encode is CANONICAL byte-level BPE: a hand-rolled
 *               Qwen2-family pretokenizer splits the text, then each
 *               pretoken starts from single alphabet symbols and adjacent
 *               pairs contract lowest-merge-rank-first.  Matches the
 *               reference (HF/llama.cpp) tokenization; merges never cross
 *               pretoken boundaries.
 *   spm / plain Two-stage encode: greedy longest-match against the vocab
 *               (stage 1 bounded by the longest vocab entry), then
 *               ranked-merge passes over the id sequence (plain only).
 *   spm         SentencePiece (Llama/Gemma/Mistral) — space is ▁ (U+2581);
 *               encode maps 0x20→▁ honoring add_space_prefix from
 *               metadata, decode maps ▁→0x20.  No merge pass (pieces
 *               match whole in stage 1).
 *   plain       raw char-level.  byte_level and spm are mutually
 *               exclusive.
 *
 * Byte fallback: bytes with no vocab match encode to <0xNN> tokens when
 * the metadata authorizes it; the id of <0xNN> is RESOLVED FROM THE
 * VOCAB, never assumed equal to the byte value (canonical SPM layouts
 * break that assumption).  A byte that resolves nowhere encodes to
 * unk_token_id; if the model has no unk either, encode fails with
 * WR_ERR_UNSUPPORTED rather than emitting token 0.
 *
 * Reentrancy: no function-local statics anywhere; encode's transform and
 * merge scratch are per-call, so one wr_bpe may be shared by concurrent
 * READERS (encode/decode/lookup take no locks and mutate nothing after
 * init).
 */
#ifndef WR_BPE_H
#define WR_BPE_H

#include "core/internal.h"
#include "core/gguf.h"

/* Sentinel for empty hash slots. */
#define WR_BPE_HASH_EMPTY 0xFFFFFFFFu

typedef struct wr_bpe {
    /* Vocab — OWNED copies (the source wr_gguf may be closed after init). */
    char    **vocab;             /* vocab[i] = NUL-terminated string       */
    uint32_t *vocab_len;         /* precomputed strlen(vocab[i])           */
    uint32_t  vocab_size;
    uint32_t  longest_token_len; /* stage-1 try_len bound                  */

    uint8_t   byte_level;
    uint8_t   spm;
    uint8_t   byte_fallback;
    uint8_t   add_space_prefix;

    /* Merges flattened to (left_id, right_id) with implicit rank = index.
     * merge_result[i] is the vocab id of vocab[left]+vocab[right], resolved
     * ONCE at init — a merge whose joined string is not itself a vocab
     * token can never be applied and is dropped during init. */
    uint32_t *merge_left;
    uint32_t *merge_right;
    uint32_t *merge_result;
    uint32_t  merge_count;

    /* (left,right) → rank hash (open addressing) — O(1) merge lookup
     * replacing the origin's O(merge_count) linear scan. */
    uint32_t *merge_hash;        /* slots of packed entry indices          */
    uint32_t  merge_hash_size;   /* power of two                           */

    /* vocab string → id hash (open addressing). */
    uint32_t *tok_hash;
    uint32_t  tok_hash_size;     /* power of two >= 2 * vocab_size         */

    /* byte value → token id for byte fallback (resolved from the vocab's
     * <0xNN> entries at init; -1 where absent). */
    int32_t   byte_token[256];

    /* Special tokens. */
    int32_t   bos_token_id;      /* -1 when absent */
    int32_t   eos_token_id;
    int32_t   pad_token_id;
    int32_t   unk_token_id;
} wr_bpe;

/* Build from parsed GGUF metadata (copies everything it keeps).
 * Errors: WR_ERR_FORMAT (no vocab), WR_ERR_NOMEM. */
int wri_bpe_init(wr_bpe *t, const wr_gguf *g);

void wri_bpe_free(wr_bpe *t);

/* Encode UTF-8 text.  Returns id count or negative wr_status; with
 * out_ids == NULL or max_ids == 0 returns the required count.  flags:
 * WR_TOK_PARSE_SPECIAL (public header) lets special-token strings in
 * `text` map to their ids; without it they tokenize as plain text. */
int wri_bpe_encode(const wr_bpe *t, const char *text,
                   uint32_t *out_ids, int max_ids, uint32_t flags);

/* Decode ids to UTF-8 (one pass, alphabet reversed, byte-fallback tokens
 * emitted as raw bytes).  Truncates only at whole-token boundaries.
 * Returns bytes written (excl. NUL) or negative wr_status. */
int wri_bpe_decode(const wr_bpe *t, const uint32_t *ids, int n_ids,
                   char *out_buf, int max_bytes);

/* Single token piece (same normalization as decode). */
int wri_bpe_token_piece(const wr_bpe *t, uint32_t id,
                        char *out_buf, int max_bytes);

/* Exact-string vocab lookup: id or -1. */
int wri_bpe_token_id(const wr_bpe *t, const char *tok);

#endif /* WR_BPE_H */
