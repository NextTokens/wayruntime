/* SPDX-License-Identifier: BUSL-1.1 */
/* SPDX-FileCopyrightText: 2026 WayOS Project */
/*
 * tokenizer_fixtures.h — compiled-in tokenizer differential-test corpus.
 *
 * Three fixture sets:
 *
 *   wr_tok_fixture_plain    encoded with flags = 0 against a byte-level
 *                           (GPT-2/Qwen family) vocab and compared 1:1
 *                           with the reference tokenizer's ids for the
 *                           SAME model file.  Deliberately adversarial
 *                           for the pretokenizer: whitespace runs, tabs,
 *                           real newlines (LF/CR/CRLF), contractions in
 *                           mixed case, digit runs of every length 1..6+,
 *                           snake/Camel identifiers, code, URLs, CJK,
 *                           Cyrillic, Greek, Hangul, Arabic, Hebrew,
 *                           emoji + ZWJ sequences, fullwidth forms, NBSP.
 *
 *   wr_tok_fixture_special  encoded with WR_TOK_PARSE_SPECIAL — control
 *                           marker strings must map to their single ids.
 *
 *   wr_tok_fixture_spm      encoded with flags = 0 against a
 *                           SentencePiece vocab (stories15M/llama) as the
 *                           no-regression pin for the ▁ mode: ids must be
 *                           byte-identical before and after any byte-level
 *                           tokenizer change.
 *
 * Strings are C literals: "\n" is a REAL newline byte, "\\n" is
 * backslash + n.  Hex escapes are isolated by literal splicing so a
 * following hex-digit character never extends the escape.
 */
#ifndef WR_TEST_TOKENIZER_FIXTURES_H
#define WR_TEST_TOKENIZER_FIXTURES_H

static const char *const wr_tok_fixture_plain[] = {
    /* prose + tiny edges */
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "a",
    "no",
    "",
    " ",
    "   ",
    /* leading/trailing/inner whitespace */
    " single lead",
    "  leading spaces and\ttabs",
    "trailing spaces   ",
    "word  word",
    "tab\there",
    "\ttab lead",
    /* newlines — real bytes */
    "\n",
    "\n\n\n",
    "\r\n\r\n",
    "a\nb",
    "a\n\nb",
    "line1\nline2\r\nline3\rline4",
    "end with newline\n",
    "spaces before newline  \n  after",
    /* contractions, mixed case */
    "It's a test: don't, I'll, THEY'RE.",
    "can'T stop, won'T stop, 'twas",
    "she'd we've y'all o'clock",
    /* identifiers + code */
    "CamelCaseIdentifiers_and_snake_case_123",
    "SCREAMING_SNAKE_CASE_CONST and lowerCamelCase",
    "int main(void) { return 0; }",
    "for (i = 0; i < n; i++) sum += a[i];",
    "printf(\"%s\\n\", str);",
    "def f(x):\n    return x**2\n",
    "if __name__ == \"__main__\":\n    main()",
    "{\"key\": \"value\", \"n\": 123, \"ok\": true}",
    "SELECT * FROM users WHERE id = 10;",
    "C:\\Users\\test\\file.txt",
    "path/to/file.py --flag=value -v",
    "x = foo(bar, baz[2]) + 42; // comment",
    "<html><body>Hello</body></html>",
    "#hashtag @mention $variable %percent &amp;",
    /* URLs / emails */
    "https://example.com/path?q=hello%20world&x=1",
    "user.name+tag@example-mail.co.uk",
    "ftp://files.example.org:8080/dir/file.tar.gz",
    /* numbers */
    "1",
    "12",
    "123",
    "1234",
    "12345",
    "123456",
    "3.14159",
    "1,234,567.89",
    "Call +1 (555) 010-9999 ext. 42",
    "2+2=4",
    "v2.0.1-beta+build.123",
    "10,000.50 dollars",
    "42nd 3rd 1st place",
    /* accented Latin */
    "caf\xC3\xA9 r\xC3\xA9sum\xC3\xA9 na\xC3\xAFve fa\xC3\xA7"
    "ade",
    "Gr\xC3\xB6\xC3\x9F"
    "enma\xC3\x9Fst\xC3\xA4"
    "be \xC3\xBC"
    "berm\xC3\xA4\xC3\x9F"
    "ig",
    "El ni\xC3\xB1o comi\xC3\xB3 jalape\xC3\xB1os",
    "Za\xC5\xBC\xC3\xB3\xC5\x82\xC4\x87 g\xC4\x99\xC5\x9Bl\xC4\x85 "
    "ja\xC5\xBA\xC5\x84",
    /* non-Latin scripts */
    "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E\xE3\x81\xAE\xE3\x83\x86"
    "\xE3\x82\xAD\xE3\x82\xB9\xE3\x83\x88\xE3\x81\xA7\xE3\x81\x99",
    "\xE4\xB8\xAD\xE6\x96\x87\xE5\x88\x86\xE8\xAF\x8D\xE6\xB5\x8B"
    "\xE8\xAF\x95\xE6\x96\x87\xE6\x9C\xAC",
    "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82, "
    "\xD0\xBC\xD0\xB8\xD1\x80! \xD0\xAD\xD1\x82\xD0\xBE "
    "\xD1\x82\xD0\xB5\xD1\x81\xD1\x82.",
    "\xCE\x93\xCE\xB5\xCE\xB9\xCE\xAC \xCF\x83\xCE\xBF\xCF\x85 "
    "\xCE\x9A\xCF\x8C\xCF\x83\xCE\xBC\xCE\xB5! "
    "\xCE\x95\xCE\xBB\xCE\xBB\xCE\xB7\xCE\xBD\xCE\xB9\xCE\xBA"
    "\xCE\xAC.",
    "\xED\x95\x9C\xEA\xB5\xAD\xEC\x96\xB4 \xED\x85\x8D\xEC\x8A\xA4"
    "\xED\x8A\xB8 \xEC\x98\x88\xEC\x8B\x9C",
    "\xD8\xA7\xD9\x84\xD8\xB9\xD8\xB1\xD8\xA8\xD9\x8A\xD8\xA9 "
    "\xD9\x86\xD8\xB5 \xD8\xAA\xD8\xAC\xD8\xB1\xD9\x8A\xD8\xA8"
    "\xD9\x8A",
    "\xD7\xA2\xD7\x91\xD7\xA8\xD7\x99\xD7\xAA \xD7\x98\xD7\xA7"
    "\xD7\xA1\xD7\x98",
    "mixed \xE4\xB8\xAD\xE6\x96\x87 and English "
    "\xD1\x82\xD0\xB5\xD0\xBA\xD1\x81\xD1\x82 "
    "v\xD0\xBC\xD0\xB5\xD1\x81\xD1\x82\xD0\xB5",
    /* emoji + symbols */
    "\xF0\x9F\x98\x80\xF0\x9F\x98\x83\xF0\x9F\x98\x84 emoji test "
    "\xF0\x9F\x8E\x89\xF0\x9F\x9A\x80",
    "\xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D"
    "\xF0\x9F\x91\xA7\xE2\x80\x8D\xF0\x9F\x91\xA6 family + "
    "\xE2\x9D\xA4\xEF\xB8\x8F heart",
    "\xC2\xBD + \xC2\xBC = \xC2\xBE and x\xC2\xB2 y\xC2\xB3",
    "\xE2\x86\x92 \xE2\x86\x90 \xE2\x86\x91 \xE2\x86\x93 "
    "\xE2\x87\x92 math: \xE2\x88\x91 \xE2\x88\x8F \xE2\x88\x9A "
    "\xE2\x88\x9E",
    /* misc edges */
    "supercalifragilisticexpialidocious "
    "Pneumonoultramicroscopicsilicovolcanoconiosis",
    "e.g. i.e. etc. vs. Dr. Smith's",
    "quoted \"string\" and 'single' quotes",
    "?!?! ... --- ===",
    "a\xC2\xA0"
    "b nbsp",
    "\xEF\xBC\xA6\xEF\xBD\x95\xEF\xBD\x8C\xEF\xBD\x8C\xEF\xBD\x97"
    "\xEF\xBD\x89\xEF\xBD\x84\xEF\xBD\x94\xEF\xBD\x88\xE3\x80\x80"
    "\xEF\xBD\x94\xEF\xBD\x85\xEF\xBD\x98\xEF\xBD\x94",
};

/* Encoded with WR_TOK_PARSE_SPECIAL: control markers map to their ids.
 * (No <|im_end|> here on purpose: it is the Qwen EOS and decode suppresses
 * control ids by contract, which would make the roundtrip column noise.) */
static const char *const wr_tok_fixture_special[] = {
    "<|im_start|>user\nHello",
    "<think>reasoning</think>done",
};

/* SentencePiece regression pin (stories15M / llama family). */
static const char *const wr_tok_fixture_spm[] = {
    "Once upon a time there was a little girl.",
    "The cat sat on the mat.",
    "Hello world",
    " leading space",
    "trailing space ",
    "two  spaces",
    "line\nbreak",
    "don't stop",
    "numbers 123 and 456",
    "punctuation, and; marks!",
};

#define WR_TOK_FIXTURE_PLAIN_COUNT \
    ((int)(sizeof wr_tok_fixture_plain / sizeof wr_tok_fixture_plain[0]))
#define WR_TOK_FIXTURE_SPECIAL_COUNT \
    ((int)(sizeof wr_tok_fixture_special / sizeof wr_tok_fixture_special[0]))
#define WR_TOK_FIXTURE_SPM_COUNT \
    ((int)(sizeof wr_tok_fixture_spm / sizeof wr_tok_fixture_spm[0]))

#endif /* WR_TEST_TOKENIZER_FIXTURES_H */
