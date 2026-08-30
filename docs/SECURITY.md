# wayruntime security model

wayruntime's job is to execute language-model inference on files you
did not author. Its security posture follows from one sentence: **a
GGUF file is hostile input until the loader has proven otherwise.**
This document says exactly what that means — and what it does not.

## Trust model

- **The host process is trusted.** wayruntime is a library; it runs
  in your address space with your privileges. It does not sandbox
  itself from you, and it cannot protect you from other code in the
  same process.
- **The GGUF file is untrusted.** Model files come from the
  internet. Every header field, offset, length, count, and string in
  the file is treated as attacker-controlled until validated.
- **Prompts and generated text are data.** The runtime never
  executes, evaluates, or writes files based on model input or
  output. What the model says is untrusted output — downstream
  consumers must treat it accordingly.
- **There is no network.** The tree contains no sockets, no HTTP
  client, no DNS — no network code at all (`grep -ri socket src
  include` is empty). Nothing is ever sent anywhere; the library
  cannot phone home.
- **There are no secrets.** The library handles no keys,
  credentials, or tokens of any kind.

## Threat model

| Surface | Threat | Posture |
|---|---|---|
| GGUF container | truncated/malformed/hostile file: bad magic or version, offsets past EOF, overlapping tensors, absurd tensor counts, unaligned data, string lengths that wrap or run past the file | validated before use: header version gate (GGUF v2/v3 only), every tensor's extent bounds-checked against the real file size, overlap rejection, power-of-2 alignment check, tensor-count cap, wrap-guarded string reads, bounded metadata element accessors → `WR_ERR_FORMAT` |
| Tensor dtypes | a dtype the compute path lacks being executed as garbage numbers | refused explicitly with `WR_ERR_UNSUPPORTED` (the log names the tensor and type); never guessed, never skipped |
| Architectures | an unknown model graph mis-run as a llama | architecture must be recognized or the load is refused (a GPT-2 GGUF is a refusal test case, not a best-effort run) |
| Tokenizer data | hostile vocab/merges metadata from the same untrusted file | same bounded-read validation posture as the rest of the container |
| Control-token injection | user-supplied text smuggling special tokens (`<|im_end|>`, ...) into the context | special-token parsing is **off by default** (`WR_TOK_PARSE_SPECIAL` is opt-in), so user text can never inject control tokens unless the caller asks for it |
| Resource exhaustion | a file whose (valid) sizes demand enormous memory or compute | allocations are computed and bounded up front (weight arena at load, full KV cache at session creation — no mid-generation reallocation); failures are `WR_ERR_NOMEM`/`WR_ERR_LIMIT`. A well-formed huge file still costs real memory and CPU — see limits below |
| Silent corruption | truncation or clamping hiding a problem | no silent clamps anywhere in the API: context overflow is `WR_ERR_CTX_FULL` (nothing consumed), out-of-range configs are errors, an unbindable SIMD request is an error |
| Network egress | exfiltration, telemetry, model download | impossible by construction: no network code exists in the tree |

## Known limits — read these before relying on it

- **Validation is about memory safety and refusal, not semantics.**
  A well-formed GGUF with adversarial *weights* will load and will
  generate whatever it was trained to generate. wayruntime makes no
  claims about model output.
- **Denial of service within validated bounds is possible.** A valid
  file can legitimately declare large tensors and a large context;
  the cost is bounded and computed up front, but "bounded" can still
  mean gigabytes and minutes of CPU. Cap `max_context` and check
  `wr_model_get_info` before committing resources on behalf of
  untrusted users.
- **A mapped model file must remain immutable while the model is live.**
  Model loads prefer a read-only mapping after container validation, so
  modifying or truncating the backing file before `wr_model_free` violates
  the caller contract and can fault on operating systems with mmap-style
  semantics. `WR_MMAP_DISABLED` (or CLI `--no-mmap`) copies the weights and
  avoids that mapped-weight hazard, but the GGUF path must still remain
  stable until every needed tokenizer has been constructed: streamed models
  reopen it lazily for tokenizer metadata. Failure to create a mapping
  already falls back to the same streamed path.
- **The library trusts its caller.** API contract violations
  (freeing a parent before its children, driving one session from
  two threads) are caught and refused where cheap, but the caller is
  inside the trust boundary; this is defense in depth, not a
  sandbox.
- **No external security audit has been performed**, and no
  sustained fuzzing campaign has run yet. The attack surfaces to
  review first: the GGUF container parser, the metadata/string
  readers, and the tokenizer's vocab/merges handling.

## Reporting

Report vulnerabilities privately to the maintainer through the
contact listed on the repository page rather than via public issues.
Include the malformed file (or a generator for it) when the report
concerns the loader — a reproducing input is the fastest path to a
fix.
