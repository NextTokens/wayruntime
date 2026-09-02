/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * bpe.c — BPE tokenizer implementation (see bpe.h for the design notes).
 *
 * Ported from the origin OS's userland tokenizer.  The spm/plain
 * two-stage encode (greedy longest-match against the vocab, then
 * ranked-merge passes over the id sequence) and the mode-specific
 * alphabet transforms carry over unchanged; what changed for this
 * library:
 *
 *   - persistence (the origin's cross-process tokenizer cache) is gone —
 *     single process, the GGUF metadata is always at hand at init;
 *   - the vocab is COPIED at init, so the source wr_gguf may be closed;
 *   - (left,right)→rank is a hash lookup (the origin scanned the merge
 *     table linearly per pair — O(merge_count) per candidate);
 *   - the merged-pair token id is resolved once at init (merge_result[]),
 *     so encode never joins strings — this also removes the origin's
 *     function-local static join buffer and makes encode reentrant;
 *   - stage-1 try_len is bounded by the longest vocab entry instead of
 *     the whole remaining text;
 *   - byte fallback resolves <0xNN> ids FROM THE VOCAB (the origin
 *     assumed a byte's token id equals the byte value, which is wrong
 *     for canonical SentencePiece layouts); a byte that resolves nowhere
 *     becomes unk_token_id, and if the model declares no unk the encode
 *     fails honestly instead of emitting token 0;
 *   - add_space_prefix comes from tokenizer metadata (the origin
 *     hard-coded the no-prefix behavior of one model family);
 *   - byte_level mode is a full rewrite, not a port: canonical byte-level
 *     BPE — a hand-rolled Qwen2-family pretokenizer (see the scanner
 *     below) splits the raw text, then each pretoken starts from single
 *     alphabet symbols and contracts the lowest-rank adjacent merge pair
 *     until none applies.  This replaces the origin's greedy
 *     longest-match stage 1 for byte-level vocabs, whose boundaries
 *     diverged from the reference tokenizers on whitespace runs,
 *     mid-word merges and punctuation clusters.
 *
 * KNOWN DIVERGENCES from the reference (HF/llama.cpp) tokenizers, kept
 * deliberately small in scope for the supported model list:
 *   - spm/plain modes keep greedy longest-match stage 1 (proven for the
 *     SentencePiece family; no pretokenizer runs there);
 *   - no Unicode normalization (no NFC/NFKC): text is consumed as the
 *     raw UTF-8 bytes handed in;
 *   - the pretokenizer's \p{L} / \p{N} / \s classification uses compact
 *     range tables (major scripts: Latin, Greek, Cyrillic, Armenian,
 *     Hebrew, Arabic, Devanagari, Bengali, Thai, Lao, Georgian, Hangul,
 *     Kana, Bopomofo, CJK, fullwidth/halfwidth forms) instead of the
 *     full Unicode database; codepoints outside the tables classify as
 *     punctuation-class, which can shift pretoken boundaries — never
 *     bytes — for exotic scripts;
 *   - special-token recognition is structural (see special_shape below):
 *     the container parser does not carry tokenizer.ggml.token_type, so
 *     "special" is decided by the token's spelling, not by metadata.
 */

#include "core/bpe.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* Encode caps the input length so every byte/token count below fits an
 * int (the public API's count type) with generous slack. */
#define WRI_BPE_MAX_TEXT ((size_t)(INT_MAX / 8))

/* ------------------------------------------------------------------------
 * Small helpers
 * ------------------------------------------------------------------------ */

static uint32_t fnv1a32(const char *s, size_t len)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static uint32_t fnv1a32_pair(uint32_t l, uint32_t r)
{
    uint8_t b[8] = {
        (uint8_t)l, (uint8_t)(l >> 8), (uint8_t)(l >> 16), (uint8_t)(l >> 24),
        (uint8_t)r, (uint8_t)(r >> 8), (uint8_t)(r >> 16), (uint8_t)(r >> 24)
    };
    return fnv1a32((const char *)b, sizeof(b));
}

static uint32_t next_pow2(uint32_t x)
{
    uint32_t r = 1;
    while (r < x) r <<= 1;
    return r;
}

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

/* Exact <0xNN> spelling (6 chars, case-insensitive hex). */
static int parse_byte_token(const char *s, size_t len, uint8_t *value)
{
    if (!s || len != 6 || s[0] != '<' || s[1] != '0' || s[2] != 'x' ||
        s[5] != '>')
        return 0;
    int hi = hex_nibble(s[3]);
    int lo = hex_nibble(s[4]);
    if (hi < 0 || lo < 0) return 0;
    if (value) *value = (uint8_t)((hi << 4) | lo);
    return 1;
}

/* ------------------------------------------------------------------------
 * Vocab hash (string → id, open addressing, linear probing)
 * ------------------------------------------------------------------------ */

static int tok_lookup(const wr_bpe *t, const char *s, size_t len)
{
    if (!t->tok_hash || t->tok_hash_size == 0) return -1;
    uint32_t mask = t->tok_hash_size - 1;
    uint32_t i = fnv1a32(s, len) & mask;
    for (uint32_t probe = 0; probe < t->tok_hash_size; probe++) {
        uint32_t slot = (i + probe) & mask;
        uint32_t id   = t->tok_hash[slot];
        if (id == WR_BPE_HASH_EMPTY) return -1;
        if ((size_t)t->vocab_len[id] == len &&
            memcmp(t->vocab[id], s, len) == 0)
            return (int)id;
    }
    return -1;
}

static void tok_insert(wr_bpe *t, uint32_t id)
{
    uint32_t mask = t->tok_hash_size - 1;
    uint32_t i = fnv1a32(t->vocab[id], t->vocab_len[id]) & mask;
    for (uint32_t probe = 0; probe < t->tok_hash_size; probe++) {
        uint32_t slot = (i + probe) & mask;
        if (t->tok_hash[slot] == WR_BPE_HASH_EMPTY) {
            t->tok_hash[slot] = id;
            return;
        }
    }
    /* Unreachable: the table is sized to at least twice the entry count. */
}

/* ------------------------------------------------------------------------
 * Merge hash ((left,right) → entry index; entry index == rank)
 * ------------------------------------------------------------------------ */

static uint32_t merge_lookup(const wr_bpe *t, uint32_t l, uint32_t r)
{
    if (!t->merge_hash || t->merge_hash_size == 0) return WR_BPE_HASH_EMPTY;
    uint32_t mask = t->merge_hash_size - 1;
    uint32_t i = fnv1a32_pair(l, r) & mask;
    for (uint32_t probe = 0; probe < t->merge_hash_size; probe++) {
        uint32_t slot = (i + probe) & mask;
        uint32_t e    = t->merge_hash[slot];
        if (e == WR_BPE_HASH_EMPTY) return WR_BPE_HASH_EMPTY;
        if (t->merge_left[e] == l && t->merge_right[e] == r) return e;
    }
    return WR_BPE_HASH_EMPTY;
}

static void merge_insert(wr_bpe *t, uint32_t idx)
{
    uint32_t l = t->merge_left[idx];
    uint32_t r = t->merge_right[idx];
    uint32_t mask = t->merge_hash_size - 1;
    uint32_t i = fnv1a32_pair(l, r) & mask;
    for (uint32_t probe = 0; probe < t->merge_hash_size; probe++) {
        uint32_t slot = (i + probe) & mask;
        uint32_t e    = t->merge_hash[slot];
        if (e == WR_BPE_HASH_EMPTY) {
            t->merge_hash[slot] = idx;
            return;
        }
        /* Duplicate pair: keep the earlier (lower-rank) entry, matching
         * the first-match semantics of the origin's linear scan. */
        if (t->merge_left[e] == l && t->merge_right[e] == r) return;
    }
}

/* ------------------------------------------------------------------------
 * Mode / metadata detection
 * ------------------------------------------------------------------------ */

/* A converter-produced GGUF can omit the byte_fallback boolean while
 * carrying SentencePiece's complete byte alphabet.  The full 0x00..0xFF
 * set is authoritative vocabulary structure; requiring EVERY value keeps
 * a coincidental ordinary token spelled like <0x41> from being mangled. */
static int detect_complete_byte_fallback(const wr_bpe *t)
{
    uint8_t seen[256];
    memset(seen, 0, sizeof(seen));
    uint32_t count = 0;
    for (uint32_t i = 0; i < t->vocab_size; i++) {
        uint8_t value;
        if (parse_byte_token(t->vocab[i], t->vocab_len[i], &value) &&
            !seen[value]) {
            seen[value] = 1;
            count++;
        }
    }
    return count == 256;
}

/* SentencePiece vocabs rarely hold a standalone ▁ token, but ▁-PREFIXED
 * word tokens (0xE2 0x96 0x81 ...) are pervasive — first hit wins. */
static int detect_spm_scan(const wr_bpe *t)
{
    for (uint32_t i = 0; i < t->vocab_size; i++) {
        const char *v = t->vocab[i];
        if (t->vocab_len[i] >= 3 &&
            (unsigned char)v[0] == 0xE2 && (unsigned char)v[1] == 0x96 &&
            (unsigned char)v[2] == 0x81)
            return 1;
    }
    return 0;
}

/* Structural special-token test, used only when WR_TOK_PARSE_SPECIAL is
 * absent: a stage-1 match (or merge result) of this shape is refused so
 * user text can never inject control tokens.  The container parser does
 * not expose tokenizer.ggml.token_type, so shape is all we have:
 *   <|...|>                       ChatML-style markers
 *   <name> with name limited to  [A-Za-z0-9_/.-]  (<s> </s> <unk> <pad>
 *                                 <start_of_turn> <0xNN> ...)
 * Ordinary vocab pieces that happen to spell an HTML-ish tag (<b>, <p>)
 * are also refused and then tokenize as their characters — a documented
 * divergence from token_type-aware tokenizers, accepted as the safe
 * direction (never injects, only splits differently). */
static int special_shape(const char *s, uint32_t len)
{
    if (len < 3 || s[0] != '<' || s[len - 1] != '>') return 0;
    if (len >= 5 && s[1] == '|' && s[len - 2] == '|') return 1;
    for (uint32_t i = 1; i + 1 < len; i++) {
        char c = s[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '/' ||
              c == '.' || c == '-'))
            return 0;
    }
    return 1;
}

static int token_is_special(const wr_bpe *t, uint32_t id)
{
    if ((t->bos_token_id >= 0 && id == (uint32_t)t->bos_token_id) ||
        (t->eos_token_id >= 0 && id == (uint32_t)t->eos_token_id) ||
        (t->pad_token_id >= 0 && id == (uint32_t)t->pad_token_id) ||
        (t->unk_token_id >= 0 && id == (uint32_t)t->unk_token_id))
        return 1;
    return special_shape(t->vocab[id], t->vocab_len[id]);
}

/* ------------------------------------------------------------------------
 * GPT-2 byte-level alphabet
 *
 * Forward map (byte → codepoint): "self" bytes 33..126, 161..172 and
 * 174..255 map to themselves; the rest are relocated into a contiguous
 * printable range so every byte has a visible spelling:
 *   0..32     → U+0100..U+0120   (space → Ġ = U+0120, newline → Ċ)
 *   127..160  → U+0121..U+0142
 *   173       → U+0143
 * ASCII printable staying identity is load-bearing: literal marker
 * strings like "<|im_start|>" survive the transform and still
 * longest-match their vocab entries.
 * ------------------------------------------------------------------------ */

static uint32_t gpt2_byte_to_unicode(unsigned char b)
{
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174))
        return b;
    if (b <= 32)              return 0x100u + b;
    if (b >= 127 && b <= 160) return 0x121u + (uint32_t)(b - 127);
    return 0x143u;            /* b == 173 */
}

/* Inverse map: original byte for an alphabet codepoint, -1 outside it. */
static int gpt2_unicode_to_byte(uint32_t cp)
{
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) ||
        (cp >= 174 && cp <= 255))
        return (int)cp;
    if (cp >= 0x100 && cp <= 0x143) {
        uint32_t idx = cp - 0x100;
        if (idx <= 32) return (int)idx;                 /* bytes 0..32    */
        if (idx <= 66) return (int)(127 + (idx - 33));  /* bytes 127..160 */
        return 173;                                     /* idx == 67      */
    }
    return -1;
}

/* Append codepoint cp (< 0x800 — the alphabet tops out at U+0143) as
 * UTF-8; the caller guarantees capacity.  Returns bytes written. */
static size_t utf8_put(char *buf, size_t p, uint32_t cp)
{
    if (cp < 0x80) {
        buf[p] = (char)cp;
        return 1;
    }
    buf[p]     = (char)(0xC0 | (cp >> 6));
    buf[p + 1] = (char)(0x80 | (cp & 0x3F));
    return 2;
}

/* Decode one UTF-8 codepoint from s[0..len); returns bytes consumed
 * (always >= 1: a malformed lead byte consumes itself as its value). */
static size_t utf8_next(const char *s, size_t len, uint32_t *cp)
{
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && len >= 2) {
        *cp = ((uint32_t)(c & 0x1F) << 6) |
              ((uint32_t)((unsigned char)s[1] & 0x3F));
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && len >= 3) {
        *cp = ((uint32_t)(c & 0x0F) << 12) |
              ((uint32_t)((unsigned char)s[1] & 0x3F) << 6) |
              ((uint32_t)((unsigned char)s[2] & 0x3F));
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && len >= 4) {
        *cp = ((uint32_t)(c & 0x07) << 18) |
              ((uint32_t)((unsigned char)s[1] & 0x3F) << 12) |
              ((uint32_t)((unsigned char)s[2] & 0x3F) << 6) |
              ((uint32_t)((unsigned char)s[3] & 0x3F));
        return 4;
    }
    *cp = c;
    return 1;
}

/* ------------------------------------------------------------------------
 * Byte-level pretokenizer (Qwen2/GPT-4 family split pattern)
 *
 * Hand-rolled scanner equivalent of the reference split regex
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)         contractions, ASCII apostrophe
 *   [^\r\n\p{L}\p{N}]?\p{L}+             letter run, one optional prefix
 *   \p{N}{1,3}                           digit groups of at most three
 *   [ ]?[^\s\p{L}\p{N}]+[\r\n]*          punct run, optional lead space
 *   \s*[\r\n]+                           whitespace ending in newlines
 *   \s+(?!\S)                            all-but-last whitespace
 *   \s+                                  trailing whitespace
 *
 * with the reference's first-match alternation order and its exact
 * quirks: the letter-run prefix class admits any single codepoint that
 * is not CR, LF, letter or digit (whitespace included), only a plain
 * 0x20 space attaches to a punctuation run, digit groups fill
 * left-to-right, and the whitespace lookahead splits off all but the
 * final whitespace codepoint when more text follows.
 *
 * Classification is by compact sorted range tables (binary search); a
 * codepoint outside every table is punctuation-class, exactly like the
 * reference's catch-all.  End of text is WRI_CP_NONE, which classifies
 * as nothing.
 * ------------------------------------------------------------------------ */

#define WRI_CP_NONE 0xFFFFFFFFu

typedef struct { uint32_t lo, hi; } wri_cp_range;

/* \p{L} — Unicode general category L* for the major scripts. */
static const wri_cp_range wri_letter_ranges[] = {
    {0x0041, 0x005A}, {0x0061, 0x007A}, {0x00AA, 0x00AA}, {0x00B5, 0x00B5},
    {0x00BA, 0x00BA}, {0x00C0, 0x00D6}, {0x00D8, 0x00F6}, {0x00F8, 0x02C1},
    {0x02C6, 0x02D1}, {0x02E0, 0x02E4}, {0x02EC, 0x02EC}, {0x02EE, 0x02EE},
    {0x0370, 0x0374}, {0x0376, 0x0377}, {0x037A, 0x037D}, {0x037F, 0x037F},
    {0x0386, 0x0386}, {0x0388, 0x038A}, {0x038C, 0x038C}, {0x038E, 0x03A1},
    {0x03A3, 0x03F5}, {0x03F7, 0x0481}, {0x048A, 0x052F}, {0x0531, 0x0556},
    {0x0559, 0x0559}, {0x0560, 0x0588}, {0x05D0, 0x05EA}, {0x05EF, 0x05F2},
    {0x0620, 0x064A}, {0x066E, 0x066F}, {0x0671, 0x06D3}, {0x06D5, 0x06D5},
    {0x06E5, 0x06E6}, {0x06EE, 0x06EF}, {0x06FA, 0x06FC}, {0x06FF, 0x06FF},
    {0x0710, 0x0710}, {0x0712, 0x072F}, {0x074D, 0x07A5}, {0x07B1, 0x07B1},
    {0x0904, 0x0939}, {0x093D, 0x093D}, {0x0950, 0x0950}, {0x0958, 0x0961},
    {0x0971, 0x0980}, {0x0985, 0x098C}, {0x098F, 0x0990}, {0x0993, 0x09A8},
    {0x09AA, 0x09B0}, {0x09B2, 0x09B2}, {0x09B6, 0x09B9}, {0x09BD, 0x09BD},
    {0x09CE, 0x09CE}, {0x09DC, 0x09DD}, {0x09DF, 0x09E1}, {0x09F0, 0x09F1},
    {0x0E01, 0x0E30}, {0x0E32, 0x0E33}, {0x0E40, 0x0E46}, {0x0E81, 0x0E82},
    {0x0E84, 0x0E84}, {0x0E86, 0x0E8A}, {0x0E8C, 0x0EA3}, {0x0EA5, 0x0EA5},
    {0x0EA7, 0x0EB0}, {0x0EB2, 0x0EB3}, {0x0EBD, 0x0EBD}, {0x0EC0, 0x0EC4},
    {0x0EC6, 0x0EC6}, {0x10A0, 0x10C5}, {0x10D0, 0x10FA}, {0x10FC, 0x10FF},
    {0x1100, 0x11FF}, {0x1E00, 0x1F15}, {0x1F18, 0x1F1D}, {0x1F20, 0x1F45},
    {0x1F48, 0x1F4D}, {0x1F50, 0x1F57}, {0x1F59, 0x1F59}, {0x1F5B, 0x1F5B},
    {0x1F5D, 0x1F5D}, {0x1F5F, 0x1F7D}, {0x1F80, 0x1FB4}, {0x1FB6, 0x1FBC},
    {0x1FBE, 0x1FBE}, {0x1FC2, 0x1FC4}, {0x1FC6, 0x1FCC}, {0x1FD0, 0x1FD3},
    {0x1FD6, 0x1FDB}, {0x1FE0, 0x1FEC}, {0x1FF2, 0x1FF4}, {0x1FF6, 0x1FFC},
    {0x2071, 0x2071}, {0x207F, 0x207F}, {0x2090, 0x209C}, {0x2102, 0x2102},
    {0x2107, 0x2107}, {0x210A, 0x2113}, {0x2115, 0x2115}, {0x2119, 0x211D},
    {0x2124, 0x2124}, {0x2126, 0x2126}, {0x2128, 0x2128}, {0x212A, 0x212D},
    {0x212F, 0x2139}, {0x213C, 0x213F}, {0x2145, 0x2149}, {0x214E, 0x214E},
    {0x2C60, 0x2C7F}, {0x2D00, 0x2D25}, {0x3005, 0x3006}, {0x3031, 0x3035},
    {0x303B, 0x303C}, {0x3041, 0x3096}, {0x309D, 0x309F}, {0x30A1, 0x30FA},
    {0x30FC, 0x30FF}, {0x3105, 0x312F}, {0x3131, 0x318E}, {0x31A0, 0x31BF},
    {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xA000, 0xA48C}, {0xAC00, 0xD7A3},
    {0xF900, 0xFA6D}, {0xFA70, 0xFAD9}, {0xFB00, 0xFB06}, {0xFB13, 0xFB17},
    {0xFB1D, 0xFB1D}, {0xFB1F, 0xFB28}, {0xFB2A, 0xFB36}, {0xFB38, 0xFB3C},
    {0xFB3E, 0xFB3E}, {0xFB40, 0xFB41}, {0xFB43, 0xFB44}, {0xFB46, 0xFBB1},
    {0xFDF0, 0xFDFB}, {0xFE70, 0xFE74}, {0xFE76, 0xFEFC}, {0xFF21, 0xFF3A},
    {0xFF41, 0xFF5A}, {0xFF66, 0xFFBE}, {0xFFC2, 0xFFC7}, {0xFFCA, 0xFFCF},
    {0xFFD2, 0xFFD7}, {0xFFDA, 0xFFDC}, {0x20000, 0x2A6DF},
    {0x2A700, 0x2EBE0},
};

/* \p{N} — decimal digits plus the common Nl/No values. */
static const wri_cp_range wri_number_ranges[] = {
    {0x0030, 0x0039}, {0x00B2, 0x00B3}, {0x00B9, 0x00B9}, {0x00BC, 0x00BE},
    {0x0660, 0x0669}, {0x06F0, 0x06F9}, {0x0966, 0x096F}, {0x09E6, 0x09EF},
    {0x09F4, 0x09F9}, {0x0E50, 0x0E59}, {0x0ED0, 0x0ED9}, {0x2070, 0x2070},
    {0x2074, 0x2079}, {0x2080, 0x2089}, {0x2150, 0x2182}, {0x2185, 0x2189},
    {0x2460, 0x249B}, {0x24EA, 0x24FF}, {0x2776, 0x2793}, {0x3007, 0x3007},
    {0x3021, 0x3029}, {0x3038, 0x303A}, {0xFF10, 0xFF19},
};

/* \s — the reference tables are generated from a Unicode-aware regex \s,
 * which includes the C0 file/group/record/unit separators. */
static const wri_cp_range wri_space_ranges[] = {
    {0x0009, 0x000D}, {0x001C, 0x001F}, {0x0020, 0x0020}, {0x0085, 0x0085},
    {0x00A0, 0x00A0}, {0x1680, 0x1680}, {0x2000, 0x200A}, {0x2028, 0x2029},
    {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000},
};

static int cp_in_ranges(const wri_cp_range *r, size_t count, uint32_t cp)
{
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < r[mid].lo)      hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else                     return 1;
    }
    return 0;
}

static int cp_is_letter(uint32_t cp)
{
    if (cp < 0x80) return (cp | 32u) >= 'a' && (cp | 32u) <= 'z';
    return cp_in_ranges(wri_letter_ranges,
                        sizeof wri_letter_ranges / sizeof *wri_letter_ranges,
                        cp);
}

static int cp_is_number(uint32_t cp)
{
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return cp_in_ranges(wri_number_ranges,
                        sizeof wri_number_ranges / sizeof *wri_number_ranges,
                        cp);
}

static int cp_is_space(uint32_t cp)
{
    if (cp < 0x80)
        return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x1F) ||
               cp == 0x20;
    return cp_in_ranges(wri_space_ranges,
                        sizeof wri_space_ranges / sizeof *wri_space_ranges,
                        cp);
}

/* Codepoint at byte `pos`, or WRI_CP_NONE past the end (*clen = 0 there,
 * so a caller advancing by *clen never moves past the end). */
static uint32_t pt_cp(const char *s, size_t len, size_t pos, size_t *clen)
{
    if (pos >= len) {
        *clen = 0;
        return WRI_CP_NONE;
    }
    uint32_t cp;
    *clen = utf8_next(s + pos, len - pos, &cp);
    return cp;
}

static uint32_t ascii_lower(uint32_t c)
{
    return (c >= 'A' && c <= 'Z') ? c + 32u : c;
}

/* Byte offset one past the letter run starting at p. */
static size_t pt_letter_run_end(const char *s, size_t len, size_t p)
{
    for (;;) {
        size_t cl;
        uint32_t c = pt_cp(s, len, p, &cl);
        if (!cp_is_letter(c)) return p;
        p += cl;
    }
}

/* Byte length of the pretoken starting at `start` (< len).  Always >= 1,
 * so the caller's scan over the fragment terminates. */
static size_t pt_next_len(const char *s, size_t len, size_t start)
{
    size_t l0;
    uint32_t c0 = pt_cp(s, len, start, &l0);

    /* (?i:'s|'t|'re|'ve|'m|'ll|'d) */
    if (c0 == '\'') {
        size_t l1;
        uint32_t w1 = ascii_lower(pt_cp(s, len, start + l0, &l1));
        if (w1 == 's' || w1 == 't' || w1 == 'm' || w1 == 'd')
            return l0 + l1;
        if (w1 == 'r' || w1 == 'v' || w1 == 'l') {
            size_t l2;
            uint32_t w2 = ascii_lower(pt_cp(s, len, start + l0 + l1, &l2));
            if ((w1 == 'r' && w2 == 'e') || (w1 == 'v' && w2 == 'e') ||
                (w1 == 'l' && w2 == 'l'))
                return l0 + l1 + l2;
        }
    }

    /* [^\r\n\p{L}\p{N}]?\p{L}+ — note the prefix class admits whitespace */
    if (!(c0 == '\r' || c0 == '\n' || cp_is_number(c0))) {
        if (cp_is_letter(c0))
            return pt_letter_run_end(s, len, start + l0) - start;
        size_t l1;
        uint32_t c1 = pt_cp(s, len, start + l0, &l1);
        if (cp_is_letter(c1))
            return pt_letter_run_end(s, len, start + l0 + l1) - start;
    }

    /* \p{N}{1,3} — groups fill left-to-right */
    if (cp_is_number(c0)) {
        size_t p = start + l0;
        for (int k = 1; k < 3; k++) {
            size_t cl;
            uint32_t c = pt_cp(s, len, p, &cl);
            if (!cp_is_number(c)) break;
            p += cl;
        }
        return p - start;
    }

    /* [ ]?[^\s\p{L}\p{N}]+[\r\n]* — only a plain 0x20 space attaches */
    {
        size_t p = (c0 == ' ') ? start + l0 : start;
        size_t cl;
        uint32_t c = pt_cp(s, len, p, &cl);
        if (c != WRI_CP_NONE && !cp_is_space(c) && !cp_is_letter(c) &&
            !cp_is_number(c)) {
            do {
                p += cl;
                c = pt_cp(s, len, p, &cl);
            } while (c != WRI_CP_NONE && !cp_is_space(c) &&
                     !cp_is_letter(c) && !cp_is_number(c));
            while (p < len && (s[p] == '\r' || s[p] == '\n'))
                p++;
            return p - start;
        }
    }

    /* Whitespace run: byte end, count, start of the final codepoint, and
     * the end of the last CR/LF inside the run. */
    {
        size_t p = start, last_start = start, rn_end = 0, count = 0;
        for (;;) {
            size_t cl;
            uint32_t c = pt_cp(s, len, p, &cl);
            if (!cp_is_space(c)) break;
            last_start = p;
            p += cl;
            count++;
            if (c == '\r' || c == '\n') rn_end = p;
        }
        if (rn_end > 0) return rn_end - start;                /* \s*[\r\n]+ */
        if (count > 1 && p < len) return last_start - start;  /* \s+(?!\S)  */
        if (count > 0) return p - start;                      /* \s+        */
    }

    return l0;   /* defensive: lone unclassifiable codepoint */
}

/* ------------------------------------------------------------------------
 * Byte-level canonical BPE
 * ------------------------------------------------------------------------ */

/* Encode ONE pretoken: map each raw byte through the alphabet to a
 * single-symbol token id, then contract the lowest-rank adjacent merge
 * pair until no merge applies.  Appends to ids at *io_n (capacity is one
 * id per input byte, guaranteed by the caller).  Merges never cross
 * pretoken boundaries — that is what makes the segmentation canonical. */
static int wri_bl_word(const wr_bpe *t, const char *s, size_t len,
                       uint32_t *ids, size_t *io_n, int parse_special)
{
    size_t base = *io_n;
    size_t m = 0;
    char ub[4];

    for (size_t i = 0; i < len; i++) {
        unsigned char b = (unsigned char)s[i];
        size_t ul = utf8_put(ub, 0, gpt2_byte_to_unicode(b));
        int id = tok_lookup(t, ub, ul);
        if (id >= 0 &&
            (parse_special || !token_is_special(t, (uint32_t)id))) {
            ids[base + m++] = (uint32_t)id;
            continue;
        }
        /* Same fallback ladder as the greedy path: <0xNN> id resolved
         * from the vocab, then unk, then an honest failure. */
        if (t->byte_fallback && t->byte_token[b] >= 0)
            ids[base + m++] = (uint32_t)t->byte_token[b];
        else if (t->unk_token_id >= 0)
            ids[base + m++] = (uint32_t)t->unk_token_id;
        else
            return WR_ERR_UNSUPPORTED;
    }

    if (t->merge_count > 0) {
        /* Until NO merge applies — unlike the greedy path's inherited
         * iteration bound, canonical BPE must run to fixpoint (each pass
         * removes one element, so at most len-1 passes). */
        uint32_t *w = ids + base;
        while (m > 1) {
            uint32_t best_rank = WR_BPE_HASH_EMPTY;
            uint32_t best_id = 0;
            size_t best_pos = 0;
            int have = 0;

            for (size_t i = 0; i + 1 < m; i++) {
                uint32_t r = merge_lookup(t, w[i], w[i + 1]);
                if (r == WR_BPE_HASH_EMPTY) continue;
                if (have && r >= best_rank) continue;
                uint32_t mid = t->merge_result[r];
                if (!parse_special && token_is_special(t, mid)) continue;
                best_rank = r;
                best_pos  = i;
                best_id   = mid;
                have = 1;
            }

            if (!have) break;

            w[best_pos] = best_id;
            if (best_pos + 2 < m)
                memmove(&w[best_pos + 1], &w[best_pos + 2],
                        (m - best_pos - 2) * sizeof(uint32_t));
            m--;
        }
    }

    *io_n = base + m;
    return WR_OK;
}

/* Pretokenize a special-free fragment and BPE-encode each pretoken. */
static int wri_bl_fragment(const wr_bpe *t, const char *s, size_t len,
                           uint32_t *ids, size_t *io_n, int parse_special)
{
    size_t pos = 0;
    while (pos < len) {
        size_t wl = pt_next_len(s, len, pos);
        int st = wri_bl_word(t, s + pos, wl, ids, io_n, parse_special);
        if (st != WR_OK) return st;
        pos += wl;
    }
    return WR_OK;
}

/* Byte-level encode of the whole text into ids (capacity raw_len — one
 * id per byte is the worst case; merges and specials only shrink it).
 * With parse_special the RAW text is partitioned on special-token
 * spellings first (leftmost longest match), exactly the reference
 * behavior; the pretokenizer then never sees the markers, which its
 * split pattern would otherwise cut apart.  Returns the id count or a
 * negative wr_status. */
static int wri_bl_encode(const wr_bpe *t, const char *text, size_t raw_len,
                         uint32_t *ids, int parse_special)
{
    size_t n = 0, frag = 0;
    int st;

    if (parse_special) {
        size_t pos = 0;
        while (pos < raw_len) {
            size_t rem = raw_len - pos;
            size_t max_try = t->longest_token_len;
            if (max_try > rem) max_try = rem;

            int sid = -1;
            size_t used = 0;
            for (size_t tl = max_try; tl > 0; tl--) {
                int cand = tok_lookup(t, text + pos, tl);
                if (cand < 0 || !token_is_special(t, (uint32_t)cand))
                    continue;
                sid = cand;
                used = tl;
                break;
            }
            if (sid < 0) {
                pos++;
                continue;
            }
            st = wri_bl_fragment(t, text + frag, pos - frag, ids, &n, 1);
            if (st != WR_OK) return st;
            ids[n++] = (uint32_t)sid;
            pos += used;
            frag = pos;
        }
    }

    st = wri_bl_fragment(t, text + frag, raw_len - frag, ids, &n,
                         parse_special);
    if (st != WR_OK) return st;
    return (int)n;
}

/* ------------------------------------------------------------------------
 * Init / free
 * ------------------------------------------------------------------------ */

int wri_bpe_init(wr_bpe *t, const wr_gguf *g)
{
    if (!t || !g) return WR_ERR_INVAL;
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < 256; i++) t->byte_token[i] = -1;

    const wr_gguf_tokenizer *src = &g->tokenizer;
    if (!src->tokens || src->vocab_size == 0) return WR_ERR_FORMAT;
    if (src->vocab_size > (1u << 30) || src->merge_count > (1u << 30))
        return WR_ERR_LIMIT;

    t->vocab_size   = src->vocab_size;
    t->bos_token_id = src->bos_token_id;
    t->eos_token_id = src->eos_token_id;
    t->pad_token_id = src->pad_token_id;
    t->unk_token_id = src->unk_token_id;

    /* Copy the vocab — the wr_gguf handle may be closed after init.  A
     * NULL source entry is normalized to "" so vocab[i] is never NULL. */
    t->vocab     = (char **)calloc(t->vocab_size, sizeof(char *));
    t->vocab_len = (uint32_t *)malloc((size_t)t->vocab_size *
                                      sizeof(uint32_t));
    if (!t->vocab || !t->vocab_len) goto nomem;
    uint32_t longest = 0;
    for (uint32_t i = 0; i < t->vocab_size; i++) {
        const char *s = src->tokens[i] ? src->tokens[i] : "";
        size_t len = strlen(s);
        t->vocab[i] = (char *)malloc(len + 1);
        if (!t->vocab[i]) goto nomem;
        memcpy(t->vocab[i], s, len + 1);
        t->vocab_len[i] = (uint32_t)len;
        if (len > longest) longest = (uint32_t)len;
    }
    t->longest_token_len = longest;

    /* String → id hash. */
    t->tok_hash_size = next_pow2(t->vocab_size * 2);
    t->tok_hash = (uint32_t *)malloc((size_t)t->tok_hash_size *
                                     sizeof(uint32_t));
    if (!t->tok_hash) goto nomem;
    for (uint32_t i = 0; i < t->tok_hash_size; i++)
        t->tok_hash[i] = WR_BPE_HASH_EMPTY;
    for (uint32_t i = 0; i < t->vocab_size; i++)
        tok_insert(t, i);

    /* Byte fallback: the metadata boolean wins when present (an explicit
     * false stays false); otherwise only a COMPLETE <0x00>..<0xFF> set
     * auto-enables it. */
    t->byte_fallback = src->byte_fallback_present
        ? (src->byte_fallback ? 1u : 0u)
        : (detect_complete_byte_fallback(t) ? 1u : 0u);

    /* Resolve <0xNN> ids from the vocab (never assume id == byte value —
     * canonical SentencePiece layouts place <0x00> at id 3).  First
     * occurrence wins on duplicates. */
    for (uint32_t i = 0; i < t->vocab_size; i++) {
        uint8_t value;
        if (parse_byte_token(t->vocab[i], t->vocab_len[i], &value) &&
            t->byte_token[value] < 0)
            t->byte_token[value] = (int32_t)i;
    }

    /* Alphabet mode: tokenizer.ggml.model is preferred ("llama" is the
     * SentencePiece family — Llama/Gemma/Mistral; "gpt2" is byte-level —
     * GPT-2/Qwen).  The vocab scan is the fallback for files without the
     * key: ▁-prefixed tokens ⇒ spm, which takes priority over byte-level
     * because an SP vocab can incidentally contain a Ġ char token; a
     * standalone Ġ (U+0120) with no ▁ tokens ⇒ byte-level; neither ⇒
     * plain char-level. */
    if (strcmp(src->model, "llama") == 0) {
        t->spm = 1;
    } else if (strcmp(src->model, "gpt2") == 0) {
        t->byte_level = 1;
    } else {
        t->spm = detect_spm_scan(t) ? 1u : 0u;
        if (!t->spm)
            t->byte_level = (tok_lookup(t, "\xC4\xA0", 2) >= 0) ? 1u : 0u;
    }

    /* add_space_prefix from metadata; absent defaults to the family
     * convention (SPM prepends the dummy ▁, others do not).  Models that
     * disable it — Gemma does — carry the key explicitly. */
    t->add_space_prefix = src->add_space_prefix_present
        ? (src->add_space_prefix ? 1u : 0u)
        : (t->spm ? 1u : 0u);

    /* Merges: resolve "left right" strings to id pairs AND the merged
     * token id, once.  A pair whose halves or whose join is absent from
     * the vocab can never be applied and is dropped here (ranks compact,
     * relative order — the only thing rank comparisons use — is kept). */
    if (src->merge_count > 0 && src->merges) {
        t->merge_left   = (uint32_t *)malloc((size_t)src->merge_count *
                                             sizeof(uint32_t));
        t->merge_right  = (uint32_t *)malloc((size_t)src->merge_count *
                                             sizeof(uint32_t));
        t->merge_result = (uint32_t *)malloc((size_t)src->merge_count *
                                             sizeof(uint32_t));
        char *join = (char *)malloc((size_t)longest * 2 + 1);
        if (!t->merge_left || !t->merge_right || !t->merge_result || !join) {
            free(join);
            goto nomem;
        }
        uint32_t kept = 0;
        for (uint32_t i = 0; i < src->merge_count; i++) {
            const char *m = src->merges[i];
            if (!m) continue;
            const char *sp = strchr(m, ' ');
            if (!sp || sp == m) continue;
            int l = tok_lookup(t, m, (size_t)(sp - m));
            int r = tok_lookup(t, sp + 1, strlen(sp + 1));
            if (l < 0 || r < 0) continue;
            size_t ll = t->vocab_len[(uint32_t)l];
            size_t rl = t->vocab_len[(uint32_t)r];
            memcpy(join, t->vocab[(uint32_t)l], ll);
            memcpy(join + ll, t->vocab[(uint32_t)r], rl);
            int mid = tok_lookup(t, join, ll + rl);
            if (mid < 0) continue;
            t->merge_left[kept]   = (uint32_t)l;
            t->merge_right[kept]  = (uint32_t)r;
            t->merge_result[kept] = (uint32_t)mid;
            kept++;
        }
        free(join);
        t->merge_count = kept;

        if (kept > 0) {
            t->merge_hash_size = next_pow2(kept * 2);
            t->merge_hash = (uint32_t *)malloc((size_t)t->merge_hash_size *
                                               sizeof(uint32_t));
            if (!t->merge_hash) goto nomem;
            for (uint32_t i = 0; i < t->merge_hash_size; i++)
                t->merge_hash[i] = WR_BPE_HASH_EMPTY;
            for (uint32_t i = 0; i < kept; i++)
                merge_insert(t, i);
        }
    }

    return WR_OK;

nomem:
    wri_bpe_free(t);
    return WR_ERR_NOMEM;
}

void wri_bpe_free(wr_bpe *t)
{
    if (!t) return;
    if (t->vocab) {
        for (uint32_t i = 0; i < t->vocab_size; i++)
            free(t->vocab[i]);
        free(t->vocab);
    }
    free(t->vocab_len);
    free(t->merge_left);
    free(t->merge_right);
    free(t->merge_result);
    free(t->merge_hash);
    free(t->tok_hash);
    memset(t, 0, sizeof(*t));
}

/* ------------------------------------------------------------------------
 * Encode
 * ------------------------------------------------------------------------ */

int wri_bpe_encode(const wr_bpe *t, const char *text,
                   uint32_t *out_ids, int max_ids, uint32_t flags)
{
    if (!t || !text || !t->vocab) return WR_ERR_INVAL;
    if (out_ids && max_ids < 0) return WR_ERR_INVAL;

    size_t raw_len = strlen(text);
    if (raw_len > WRI_BPE_MAX_TEXT) return WR_ERR_LIMIT;

    int parse_special = (flags & WR_TOK_PARSE_SPECIAL) != 0;

    /* Byte-level mode: canonical BPE (pretokenizer + lowest-rank pair
     * contraction per pretoken) — see the scanner above.  Per-call
     * scratch only; same output contract as the greedy path below. */
    if (t->byte_level) {
        if (raw_len == 0) return 0;
        uint32_t *bl_ids = (uint32_t *)malloc(raw_len * sizeof(uint32_t));
        if (!bl_ids) return WR_ERR_NOMEM;
        int n_bl = wri_bl_encode(t, text, raw_len, bl_ids, parse_special);
        if (n_bl < 0 || !out_ids || max_ids == 0) {
            free(bl_ids);
            return n_bl;
        }
        if (n_bl > max_ids) {
            free(bl_ids);
            return WR_ERR_LIMIT;
        }
        memcpy(out_ids, bl_ids, (size_t)n_bl * sizeof(uint32_t));
        free(bl_ids);
        return n_bl;
    }

    /* Alphabet transform (spm/plain).  NOTE: no pretokenizer and no
     * Unicode normalization run before this (documented divergence) —
     * the raw UTF-8 bytes are transformed and matched as-is. */
    const char *x = text;
    size_t xlen = raw_len;
    char *xbuf = NULL;

    if (t->spm && raw_len > 0) {
        /* SentencePiece: each space becomes ▁ (3 bytes) so the input
         * matches the vocab's ▁-prefixed word tokens; add_space_prefix
         * prepends the dummy ▁ the reference tokenizer adds. */
        xbuf = (char *)malloc(raw_len * 3 + 4);
        if (!xbuf) return WR_ERR_NOMEM;
        size_t xp = 0;
        if (t->add_space_prefix) {
            xbuf[xp++] = (char)0xE2;
            xbuf[xp++] = (char)0x96;
            xbuf[xp++] = (char)0x81;
        }
        for (size_t i = 0; i < raw_len; i++) {
            if (text[i] == ' ') {
                xbuf[xp++] = (char)0xE2;
                xbuf[xp++] = (char)0x96;
                xbuf[xp++] = (char)0x81;
            } else {
                xbuf[xp++] = text[i];
            }
        }
        xbuf[xp] = '\0';
        x = xbuf;
        xlen = xp;
    }

    if (xlen == 0) {
        free(xbuf);
        return 0;
    }

    /* Per-call id scratch (never partial output into out_ids; also serves
     * the count-only mode).  Stage 1 emits at most one id per byte. */
    uint32_t *ids = (uint32_t *)malloc(xlen * sizeof(uint32_t));
    if (!ids) {
        free(xbuf);
        return WR_ERR_NOMEM;
    }

    /* Stage 1: greedy longest-match, try_len bounded by the longest vocab
     * entry.  Without WR_TOK_PARSE_SPECIAL a special-shaped match is
     * refused and a shorter length is tried — user text cannot inject
     * control tokens. */
    size_t pos = 0;
    size_t n = 0;
    while (pos < xlen) {
        size_t rem = xlen - pos;
        size_t max_try = t->longest_token_len;
        if (max_try > rem) max_try = rem;

        int match = -1;
        size_t used = 0;
        for (size_t try_len = max_try; try_len > 0; try_len--) {
            int cand = tok_lookup(t, x + pos, try_len);
            if (cand < 0) continue;
            if (!parse_special && token_is_special(t, (uint32_t)cand))
                continue;
            match = cand;
            used = try_len;
            break;
        }
        if (match >= 0) {
            ids[n++] = (uint32_t)match;
            pos += used;
            continue;
        }

        /* No vocab match at this byte: byte fallback (<0xNN> id resolved
         * from the vocab), then unk, then an honest failure — never a
         * hard-coded token 0. */
        unsigned char b = (unsigned char)x[pos];
        int fb;
        if (t->byte_fallback && t->byte_token[b] >= 0)
            fb = (int)t->byte_token[b];
        else if (t->unk_token_id >= 0)
            fb = (int)t->unk_token_id;
        else {
            free(ids);
            free(xbuf);
            return WR_ERR_UNSUPPORTED;
        }
        ids[n++] = (uint32_t)fb;
        pos++;
    }

    /* Stage 2: ranked BPE merge — NOT for SentencePiece vocabs, whose
     * pieces match whole in stage 1 (a merge-rank pass over them corrupts
     * the segmentation).  Rank lookup is the (left,right) hash; the
     * merged id was resolved at init, so no strings are touched here.
     * The iteration bound (iter < live n) is inherited from the origin
     * algorithm verbatim. */
    if (!t->spm && t->merge_count > 0) {
        for (size_t iter = 0; iter < n; iter++) {
            uint32_t best_rank = WR_BPE_HASH_EMPTY;
            uint32_t best_id = 0;
            size_t best_pos = 0;
            int have = 0;

            for (size_t i = 0; i + 1 < n; i++) {
                uint32_t r = merge_lookup(t, ids[i], ids[i + 1]);
                if (r == WR_BPE_HASH_EMPTY) continue;
                if (have && r >= best_rank) continue;
                uint32_t mid = t->merge_result[r];
                if (!parse_special && token_is_special(t, mid)) continue;
                best_rank = r;
                best_pos  = i;
                best_id   = mid;
                have = 1;
            }

            if (!have) break;

            ids[best_pos] = best_id;
            if (best_pos + 2 < n)
                memmove(&ids[best_pos + 1], &ids[best_pos + 2],
                        (n - best_pos - 2) * sizeof(uint32_t));
            n--;
        }
    }

    free(xbuf);

    if (!out_ids || max_ids == 0) {
        free(ids);
        return (int)n;
    }
    if (n > (size_t)max_ids) {
        free(ids);
        return WR_ERR_LIMIT;
    }
    memcpy(out_ids, ids, n * sizeof(uint32_t));
    free(ids);
    return (int)n;
}

/* ------------------------------------------------------------------------
 * Decode
 * ------------------------------------------------------------------------ */

/* Expand token `id` into its normalized piece.  With dst == NULL only
 * measures; with dst != NULL the caller guarantees capacity for the
 * measured length (the piece is never longer than the raw vocab string).
 * Returns the piece byte length. */
static size_t piece_expand(const wr_bpe *t, uint32_t id, char *dst)
{
    const char *s = t->vocab[id];
    size_t len = t->vocab_len[id];

    /* <0xNN> emits its raw byte only when the metadata opted this
     * tokenizer into byte fallback; a legitimate ordinary token with that
     * spelling stays literal otherwise.  Checked before the mode paths —
     * it applies to SPM and plain vocabs alike. */
    uint8_t fb;
    if (t->byte_fallback && parse_byte_token(s, len, &fb)) {
        if (dst) dst[0] = (char)fb;
        return 1;
    }

    if (t->byte_level) {
        /* Reverse the byte-level alphabet codepoint-by-codepoint.  A
         * codepoint outside the alphabet is silently dropped — such
         * codepoints cannot appear in a well-formed byte-level vocab, so
         * this only fires on corrupt entries (documented divergence:
         * reference decoders map through a full table and cannot see
         * out-of-alphabet input either). */
        size_t p = 0, k = 0;
        while (k < len) {
            uint32_t cp;
            k += utf8_next(s + k, len - k, &cp);
            int b = gpt2_unicode_to_byte(cp);
            if (b < 0) continue;
            if (dst) dst[p] = (char)b;
            p++;
        }
        return p;
    }

    if (t->spm) {
        /* ▁ (0xE2 0x96 0x81) → space; everything else verbatim. */
        size_t p = 0, k = 0;
        while (k < len) {
            if (k + 2 < len && (unsigned char)s[k] == 0xE2 &&
                (unsigned char)s[k + 1] == 0x96 &&
                (unsigned char)s[k + 2] == 0x81) {
                if (dst) dst[p] = ' ';
                p++;
                k += 3;
                continue;
            }
            if (dst) dst[p] = s[k];
            p++;
            k++;
        }
        return p;
    }

    if (dst) memcpy(dst, s, len);
    return len;
}

int wri_bpe_decode(const wr_bpe *t, const uint32_t *ids, int n_ids,
                   char *out_buf, int max_bytes)
{
    if (!t || (!ids && n_ids > 0) || !out_buf || n_ids < 0 || max_bytes <= 0)
        return WR_ERR_INVAL;

    size_t p = 0;
    size_t cap = (size_t)max_bytes;
    for (int i = 0; i < n_ids; i++) {
        uint32_t id = ids[i];
        if (id >= t->vocab_size) continue;
        /* Control tokens frame the conversation, they are not reply
         * text.  Suppression is strictly by metadata ID — a token-name
         * heuristic could hide ordinary vocabulary. */
        if ((t->bos_token_id >= 0 && id == (uint32_t)t->bos_token_id) ||
            (t->eos_token_id >= 0 && id == (uint32_t)t->eos_token_id) ||
            (t->pad_token_id >= 0 && id == (uint32_t)t->pad_token_id) ||
            (t->unk_token_id >= 0 && id == (uint32_t)t->unk_token_id))
            continue;

        /* Graceful truncation at WHOLE-token boundaries: measure first,
         * stop before a token that will not fit (partial coherent text
         * beats an empty buffer; a half-written multi-byte character
         * would beat neither). */
        size_t need = piece_expand(t, id, NULL);
        if (p + need + 1 > cap) break;
        piece_expand(t, id, out_buf + p);
        p += need;
    }
    out_buf[p] = '\0';
    return (int)p;
}

int wri_bpe_token_piece(const wr_bpe *t, uint32_t id,
                        char *out_buf, int max_bytes)
{
    if (!t || !out_buf || max_bytes <= 0) return WR_ERR_INVAL;
    if (id >= t->vocab_size) return WR_ERR_INVAL;
    size_t need = piece_expand(t, id, NULL);
    if (need + 1 > (size_t)max_bytes) return WR_ERR_LIMIT;
    piece_expand(t, id, out_buf);
    out_buf[need] = '\0';
    return (int)need;
}

int wri_bpe_token_id(const wr_bpe *t, const char *tok)
{
    if (!t || !tok) return -1;
    return tok_lookup(t, tok, strlen(tok));
}
