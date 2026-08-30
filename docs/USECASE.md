<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 WayOS Project -->

# Use case: schema-shaped output, guaranteed by construction

A worked, tested example of what the token-mask API is for.
Everything below was executed for real — a real local model, real
constrained decoding, a real strict JSON parser — and is re-runnable
as a single script:

```sh
make usecase          # 6 checks, needs a real model (see below)
```

## The problem

A program that consumes LLM output — a tool-call router, a config
generator, an extraction pipeline — needs *machine-readable* output.
An unconstrained model produces it only probabilistically: today the
JSON arrives wrapped in a markdown fence, tomorrow prefixed with
"Sure! Here's", next week truncated mid-string. The usual tax is a
repair layer — regex trimming, "output parsers", retry loops that
re-prompt until something parses — all of it downstream of output
that was already wrong.

The alternative is to make malformed output *unrepresentable*: check
every candidate token against a grammar **before** it is emitted.
Tokens that would break the grammar never enter the output, so the
result is well-formed by construction, whatever the model would
rather say.

## The mechanism

wayruntime's sampler accepts a token-mask callback
(`wr_sampler_set_mask`, contract in
[`include/wayruntime/wayruntime.h`](../include/wayruntime/wayruntime.h)):
during `wr_sample` the callback is consulted per candidate token with
the token's detokenized byte string, and returns allow/forbid. If the
mask forbids every candidate, `wr_sample` returns the best forbidden
token rather than failing — the consumer detects that when its commit
fails, and stops. Honest by contract, no silent fallback.

[`examples/grammar_mask.c`](../examples/grammar_mask.c) (Apache-2.0,
~340 lines, no dependencies beyond the public header) builds a
byte-level automaton for a tiny JSON grammar —

```
object := ws '{' pair (ws ',' pair)* ws '}' ws
pair   := ws '"' key '"' ':' ' '* '"' value '"'
```

— and wires it in twice:

- the **mask** callback (`gm_allow`) trial-feeds a candidate token's
  bytes through a *copy* of the automaton, never mutating live state;
- the **stream** callback (`gm_commit`) advances the real automaton
  with the token that actually won, and stops generation the moment
  the grammar reaches its accept state.

EOS is special-cased by id: it carries no bytes, so it is legal
exactly when the object is complete — which also lets the example
demand *structure*, not just syntax: `gm_require_pairs(&g, 2)` makes
`}` and EOS illegal before the second pair, so a lazy `{}` cannot
happen either.

## How to run it

```sh
make usecase                                  # ../models/Qwen3-0.6B-Q4_K_M.gguf
make usecase MODEL_GGUF=path/to/model.gguf    # any supported GGUF
```

or by hand, against nothing but the public header and the static
library:

```sh
gcc -std=c11 -O2 -Wall -Wextra -Wshadow -Wvla -Iinclude \
    -o grammar_mask examples/grammar_mask.c \
    build/posix/libwayruntime.a -lpthread -lm
./grammar_mask ../models/Qwen3-0.6B-Q4_K_M.gguf
```

## The run (real transcript, 2026-08-30)

Model: Qwen3-0.6B Q4_K_M, greedy sampling, mask installed. Prompt:
`"Here is a small JSON object with string values describing a
cat:\n"`. stdout, verbatim:

```json
{
  "name": "Bella",
  "age":   "12",
  "color": "black and white",
  "eyes": "green and brown",
  "hair": "long and silky",
  "legs": "4",
  "tail": "long"
}
```

stderr summary, exit code 0:

```
grammar_mask: 13 prompt tokens, 61 generated, stop=4, 7 pairs, grammar complete
```

(`stop=4` is `WR_STOP_CALLBACK`: the commit callback ended generation
at the grammar's accept state — the model never even got to ramble.)

The same prompt, same model, same greedy decoding **without** the
mask (`wayrt generate --raw --greedy`), verbatim from the same test
run:

````
```json
{
  "name": "Bella",
  "age": 3,
  ...
}
```
Based on the JSON, what is the name of the cat?
```
A. Bella
B. Bella's
...
````

The model wrapped the object in a markdown fence, then kept going and
wrote itself a quiz. `json.loads` on that stdout: **DOES-NOT-PARSE**.
On the masked stdout: parses, every run, byte-identical.

## The test ↔ doc contract

Every claim above is an assertion in
[`test/usecase_grammar.sh`](../test/usecase_grammar.sh); the script
and this document are maintained together.

| claim | asserted by step |
|---|---|
| the example builds warning-free against the public SDK surface only (full warning set + `-Werror`, empty compiler stderr) | 1 |
| the constrained run succeeds and the grammar reaches its accept state (`grammar complete` on stderr, exit 0) | 2 |
| stdout is valid JSON, verbatim — `python3` `json.load`, no trimming, no repair | 3 |
| the value is a flat object of ≥ 2 `string:string` pairs, the minimum enforced by the mask alone | 4 |
| greedy + masked decoding is deterministic: a second run is byte-identical | 5 |
| the unconstrained contrast is shown (PARSES / DOES-NOT-PARSE), printed but never scored — an unmasked model may emit valid JSON by luck; nothing guarantees it | informational block |
| no temp files are left behind | 6 |

## Honest limits

- **The grammar is deliberately tiny.** A flat object of
  `string:string` pairs: no nesting, no numbers, no booleans, no
  escape sequences (`"` and `\` are simply illegal inside values).
  That is the point of the example — the automaton is ~70 lines you
  can read in one sitting and re-target to your own schema — not a
  general JSON-Schema engine. Note the masked run's `"age": "12"`
  versus the unmasked `"age": 3`: the mask forced even the number
  into a string, exactly as this grammar demands.
- **Failure is honest.** A model that cannot complete the grammar
  within the token budget (`max_tokens` ran out mid-value, say) exits
  nonzero with `grammar INCOMPLETE` on stderr. No fallback, no
  half-object presented as success.
- **Masking has a cost.** The callback trial-feeds one piece lookup
  per *candidate* token per step (the sampler consults it for many
  candidates before one wins). The automaton here is a few branches
  on a struct copy — cheap — but a heavyweight mask callback would
  sit on the sampling hot path.
- **Syntactically valid ≠ true.** The mask guarantees shape, not
  facts. This cat's age, colors, and leg count are whatever the model
  says they are; the guarantee is that your parser will never choke
  on how it says them.
