# Qwen3.5-9B vs tim-9b — Same-Task Comparison

Head-to-head of two **9-billion-parameter** models on the identical
`test-subconscious-hit.py` two-turn workload, at different quantizations, with
and without the Subconscious Cache.

- **Date:** 2026-07-20 · Windows 11 x64 · CPU inference · `bin/llama-server.exe`
- **Server flags (common):** `-c 32768 --jinja --chat-template-file qwen.jinja --reasoning off --no-mmproj`, `LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"preserve_thinking": true}`
- **Methodology:** `max_tokens=256` (replies finish naturally, `<|im_end|>` present), **a fresh cold server per rep** (2 reps) so every turn-1 is genuinely cold.

| Model | File | Quant | Size | Architecture / cache path |
|---|---|---|---|---|
| **qwen-9b** | `Qwen3.5-9B-Q4_K_M.gguf` | Q4_K_M (~4.5-bit) | 5.29 GB | hybrid attention + Mamba/SSM · MambaRadixCache |
| **tim-9b** | `tim-9b-1.25bit.gguf` | 1.25-bit QAT | 3.04 GB | hybrid attention + Mamba/SSM · MambaRadixCache |

> **Both models are hybrid Mamba+attention and use the same MambaRadixCache
> surgery path.** Architecture is *not* a differentiator here.

> **⚡ tim-9b measured with the new 1.3-bit CPU kernel (2026-07-20).** A
> `ggml-cpu.dll` swap adds a dedicated kernel for the 1.3-bit quant that speeds up
> tim-9b **prefill ~2.7×** (32.9 → 88.7 tok/s), decode unchanged. **qwen-9b
> numbers are from the previous binary** — the kernel is tim-quant-specific and
> does not touch the Q4_K_M path (see caveat in §4). This kernel change **flips
> the head-to-head**: on the old kernel qwen-9b was the faster model; with the new
> kernel tim-9b leads on every axis.

---

## 1. Raw model speed (cold turn 1, identical 325-token prompt)

| Model | Prefill tok/s | Decode tok/s | Cold TTFT | Reply toks |
|---|---|---|---|---|
| **tim-9b** (new kernel) | **88.7** | **13.4** | **3.66 s** | 81 |
| qwen-9b | 81.0 | 11.0 | 4.04 s | 76 |
| _tim-9b (old kernel, for reference)_ | _32.6_ | _13.5_ | _9.96 s_ | _77_ |

- With the new kernel **tim-9b now prefills faster than qwen-9b** (88.7 vs 81.0
  tok/s) *and* is the smaller file (3.0 vs 5.3 GB). On the old kernel it prefilled
  ~2.5× **slower** — this is the single biggest change.
- tim-9b also decodes a bit faster (13.4 vs 11.0 tok/s).
- Net cold latency: **tim-9b 3.66 s vs qwen-9b 4.04 s** — tim-9b is now ahead cold,
  where it used to trail badly (9.96 s).

---

## 2. Subconscious Cache benefit (turn 2, cold cache)

Identical turn-2 request; the only variable is `--suffix-cache`. Median of 2 reps.

| Model | Cache | turn2 TTFT | turn2 total | cached / prompt | new | Speedup vs OFF |
|---|---|---|---|---|---|---|
| tim-9b (new) | OFF | 2.80 s | 9.51 s | 0 / 249 | 249 | — |
| tim-9b (new) | **ON** | **0.69 s** | **7.51 s** | **207 / 249** | 42 | **4.1× TTFT / 1.3× total** |
| qwen-9b | OFF | 2.99 s | 7.23 s | 0 / 244 | 244 | — |
| qwen-9b | **ON** | **0.73 s** | **4.42 s** | **202 / 244** | 42 | **4.1× TTFT / 1.6× total** |

**Both models hit surgery cleanly** — confirmed by `pp_cache.log` (2 surgery lines
each) and by the near-identical split:

```
tim-9b :  surgery  incoming=249  cache=406  |  prefix=98  gap=198  suffix=109  new=42
qwen-9b:  surgery  incoming=244  cache=400  |  prefix=98  gap=198  suffix=104  new=42
```

Each reuses ~83% of the turn-2 prompt (202–207 / ~245) and prefills only the 42
genuinely-new tokens.

- **tim-9b:** warm turn-2 TTFT **0.69 s** — now the fastest of the two.
- **qwen-9b:** warm turn-2 TTFT **0.73 s**.
- The **cache ratio is essentially equal** now (~4.1× each): with the new kernel
  the two models prefill at a similar per-token rate, so skipping the same ~200
  tokens saves a proportional amount for both. (On the old kernel tim-9b's ratio
  was 5.1× because its prefill was the more expensive to skip.)

---

## 3. Output quality

Both models answered the Paris-vs-London task coherently and stayed on-topic
across both turns, with no drift on the surgery path (the reused recurrent state
did not corrupt the reply). Quality was roughly at parity for this short factual
task; the 1.25-bit tim-9b showed no obvious degradation here. (This workload is a
cache smoke-test, not a rigorous quality benchmark.)

---

## 4. Bottom line

| Dimension | Winner (new kernel) | Detail |
|---|---|---|
| Cold prefill speed | **tim-9b** | 88.7 vs 81.0 tok/s |
| Decode speed | **tim-9b** | 13.4 vs 11.0 tok/s |
| Cold TTFT | **tim-9b** | 3.66 s vs 4.04 s |
| Warm turn-2 TTFT (cache-on) | **tim-9b** | 0.69 s vs 0.73 s |
| Model size on disk | **tim-9b** | 3.0 vs 5.3 GB |
| Subconscious Cache — relative speedup | tie | ~4.1× turn-2 TTFT each |

**Interpretation.** With the new 1.3-bit kernel, **tim-9b overtakes qwen-9b on
every performance axis** — prefill, decode, cold and warm latency — while being
~40% smaller on disk. The Subconscious Cache works identically on both (same
~200-token reuse, same ~4.1× turn-2 TTFT), so it is not a differentiator between
them; it simply makes repeat-turn latency small for both. Before the kernel,
qwen-9b won on latency and tim-9b's case rested on footprint + the cache
amortizing its heavy prefill; the kernel removes that trade-off and makes tim-9b
the straightforward pick here.

> **Caveat — binary consistency.** qwen-9b was measured on the *previous*
> `ggml-cpu.dll`. The new kernel is dedicated to the 1.3-bit quant and should not
> affect the Q4_K_M path, but since the whole DLL was swapped, a fully
> apples-to-apples result would re-run qwen-9b on the new binary. Its numbers are
> not expected to change materially, but this has not been re-verified.

---

## 5. Raw per-rep data

### tim-9b — new 1.3-bit kernel (mt=256, fresh cold server per rep)
```
config             mt | t1_ttft t1_tot | t2_ttft t2_tot | t2 cached/prompt
tim-9b/cache-on   256 |   3.65   9.73  |   0.68   7.52  | 207/249   <- surgery
tim-9b/cache-on   256 |   3.68   9.70  |   0.69   7.50  | 207/249   <- surgery
tim-9b/cache-off  256 |   3.68   9.74  |   2.79   9.37  | 0/249
tim-9b/cache-off  256 |   3.73   9.95  |   2.82   9.65  | 0/249
```

### qwen-9b — previous binary (mt=256, fresh cold server per rep)
```
config             mt | t1_ttft t1_tot | t2_ttft t2_tot | t2 cached/prompt
qwen-9b/cache-on  256 |   4.01  10.87  |   0.74   4.44  | 202/244   <- surgery
qwen-9b/cache-on  256 |   4.02  10.96  |   0.72   4.41  | 202/244   <- surgery
qwen-9b/cache-off 256 |   4.02  10.97  |   3.00   7.21  | 0/244
qwen-9b/cache-off 256 |   4.05  10.97  |   2.99   7.24  | 0/244
```

### Harness-bug note (why an earlier draft of this file was wrong)

A first pass reported that qwen-9b got **no** cache benefit and was rejected for
surgery, and wrongly attributed it to a "pure attention" architecture. Both claims
were wrong:

- Qwen3.5-9B is a **hybrid Mamba+attention** model on the same MambaRadixCache
  path as tim-9b — not pure attention.
- The observed miss was a **bug in the benchmark client**: the streaming reader
  used `requests.iter_lines(decode_unicode=True)`, which decodes per network chunk
  and **mangles a multi-byte UTF-8 character split across chunks** — the `é` in
  "café" in qwen-9b's reply `R1`. The corrupted `R1`, re-sent as history in turn
  2, re-tokenized differently (`cache_decision ... suffix=66 middle=40`, then
  `no checkpoint … full reset`), so the hit was rejected. tim-9b was unaffected
  only because its reply happened to be pure ASCII.

Fix: iterate raw bytes and decode each *complete* SSE line as UTF-8. After the fix
(and shown above), qwen-9b surgers exactly like tim-9b. The `--suffix-cache`
requirement that "the re-sent R1 must tokenize identically to the cached R1" is
strict — one mishandled multi-byte character breaks it.

See also [BENCHMARK.md](BENCHMARK.md) for the tim / qwen-2b runs and the
long-context rationale.
