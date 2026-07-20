# Subconscious Cache Benchmark

A controlled benchmark of the **Subconscious Cache** (`--suffix-cache`) in this
`llama-server` build, measured on two local models with the two-turn A·B·C / A·C·D
workload from [`test-subconscious-hit.py`](test-subconscious-hit.py).

- **Date:** 2026-07-20
- **Machine:** Windows 11, x64, CPU inference
- **Server:** bundled `bin/llama-server.exe`, `-c 32768`, `--jinja --chat-template-file qwen.jinja --reasoning off --no-mmproj`, `LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"preserve_thinking": true}`
- **Models:**
  - `qwen-2b` — `Qwen3.5-2B-Q4_K_M.gguf` (~1.2 GB)
  - `tim-9b` — `tim-9b-1.25bit.gguf` (~3.0 GB, Subconscious's own model)

> **⚡ Update 2026-07-20 — tim-9b re-measured with the new 1.3-bit CPU kernel**
>
> After a `ggml-cpu.dll` swap that adds a **dedicated kernel for the 1.3-bit
> quant**, tim-9b was re-run (fresh cold server per rep). The kernel **~2.7×**
> speeds up tim-9b **prefill** and leaves decode unchanged (it is a prefill
> kernel):
>
> | tim-9b | Prefill tok/s | Cold TTFT (325 tok) | Decode tok/s | Warm turn-2 TTFT (cache-on) |
> |---|---|---|---|---|
> | old kernel | 32.9 | 9.9 s | 13.4 | 1.45 s |
> | **new kernel** | **88.7** | **3.66 s** | 13.4 | **0.69 s** |
>
> **All tim-9b numbers below are the new-kernel values.** qwen-2b numbers are from
> the original binary (the new kernel is tim-quant-specific and does not touch the
> Q4_K_M path).

---

## 1. Method

### Workload

Each measurement uses the exact construction from `test-subconscious-hit.py`
(imported verbatim, driven over the streaming API so time-to-first-token is
measured separately from decode):

- **Turn 1 (warmup, cold):** `[system, U1, A1, U2, A2, U3]` → the server generates
  `R1` and caches the whole chain `system+U1+A1+U2+A2+U3+R1`.
- **Turn 2 (target):** `[system, U1, A1, U3, R1, U4]` — the middle turn `U2/A2`
  is **dropped**, Turn 1's actual `R1` is spliced back in as history, and a new
  question `U4` is appended.

This is the canonical shape the cache is built for:

| Segment | Content | Role |
|---|---|---|
| `A` (prefix) | `system + U1 + A1` | shared prefix — reused directly |
| `B` (gap) | `U2 + A2` | exists only in the cache — **dropped** |
| `C` (suffix) | `U3 + R1 + <\|im_end\|>` | shared suffix — KV kept, positions shifted after `A` |
| `D` (new) | `U4` + generation prompt | the only genuinely new tokens |

A standard prefix cache stops at the first divergence (`U2` vs `U3`, right after
`A`) and recomputes everything after it. The Subconscious Cache additionally
reuses `C` and drops `B` — the operation the logs call **surgery**.

### Matrix & methodology

2 models × {`--suffix-cache` **ON**, **OFF**}. `max_tokens=8` keeps the reply short
so the measurement is **prefill-dominated** (this is where the cache acts);
`max_tokens=256` adds a realistic decode tail (replies finish naturally, so
`<|im_end|>` is present and the suffix match is robust). tim-9b uses **a fresh
cold server per rep** (2 reps) — this avoids the `/flush_cache` carryover that
contaminated an earlier `max_tokens=128` pass. Streaming reads decode each SSE
line as UTF-8 from raw bytes (a mid-stream multi-byte decode bug that had
corrupted a non-ASCII `R1` is fixed).

### Metrics

TTFT (time to first token ≈ prefill), total latency, prompt / completion /
**cached** tokens (from `usage.prompt_tokens_details.cached_tokens`), decode
throughput, and the authoritative `surgery` lines from `pp_cache.log`
(`BRAINTREE_PP_CACHE_LOG_PATH`).

---

## 2. Headline result — turn-2 latency, cold cache

The cleanest apples-to-apples comparison: an **identical** Turn 2 request, the
only variable being `--suffix-cache`. Median of reps at `max_tokens=8`.

| Model | Cache | turn2 TTFT | turn2 total | reused tokens | new tokens | Speedup vs OFF |
|---|---|---|---|---|---|---|
| qwen-2b | OFF | 0.613 s | 0.803 s | 0 / 177 | 177 | — |
| qwen-2b | **ON** | **0.226 s** | **0.418 s** | **134 / 177** | 43 | **2.7× TTFT / 1.9× total** |
| tim-9b (new kernel) | OFF | 2.025 s | 2.546 s | 0 / 177 | 177 | — |
| tim-9b (new kernel) | **ON** | **0.756 s** | **1.278 s** | **134 / 177** | 43 | **2.7× TTFT / 2.0× total** |

With the cache on, **~76% of the Turn 2 prompt (134 / 177 tokens) is reused** and
only 43 tokens are actually prefilled.

Measured turn-1 (cold) → turn-2 (warm) within the cache-ON runs, the effect is
larger still, because Turn 2 skips prefilling the *entire* prior history:

| Model | turn1 TTFT (cold) | turn2 TTFT (surgery) | TTFT speedup |
|---|---|---|---|
| qwen-2b | 1.04 s | 0.23 s | **4.6×** |
| tim-9b (new kernel) | 3.61 s | 0.76 s | **4.8×** |

> On the **old** kernel tim-9b's cold turn-1 was 10.34 s and this ratio was 6.8× —
> the faster prefill kernel shrinks the *relative* win (there is less prefill to
> skip) while making every absolute latency smaller. See §4.

---

## 3. The surgery is confirmed — and model-independent

`pp_cache.log` records the split for every Turn 2. The breakdown is
**byte-for-byte identical across both models**, because surgery is a
token-structure operation on the KV cache, independent of how fast the model
runs:

```
surgery  incoming=177  cache=332  |  prefix=98  gap=198  suffix=36   new=43    (mt=8,  qwen AND tim)
surgery  incoming=249  cache=406  |  prefix=98  gap=198  suffix=109  new=42    (mt=256, tim, cold)
```

- `prefix=98` (`A`) reused, `gap=198` (`B`) dropped, `suffix=36/109` (`C`) reused
  via a single RoPE shift, only `new≈43` prefilled.
- Cache **OFF** logged **0** surgery lines, as expected — the feature is off.
- tim-9b logged **2 surgery lines per cache-on config** (both reps), at both
  `mt=8` and `mt=256`.

---

## 4. The win is prefill, not decode

Decode throughput is **unchanged** by the cache — it only skips prefill:

| Model | decode tok/s (cache OFF) | decode tok/s (cache ON) |
|---|---|---|
| qwen-2b | 41.1 | 41.4 |
| tim-9b (new kernel) | 15.1 | 14.8 |

Two consequences:

1. **The absolute saving equals the prefill cost of `B + C`.** The bigger the
   dropped middle plus reused suffix, and the more expensive each token is to
   prefill, the more surgery saves. The new 1.3-bit kernel cut tim-9b's per-token
   prefill cost ~2.7× (32.9 → 88.7 tok/s), so the *absolute* saving per surgery
   shrank accordingly — but so did every cold latency.
2. **Total-time speedup shrinks as the reply grows** (decode dilutes the fixed
   prefill saving — an Amdahl effect). Short answers over long context show the
   effect most clearly.

Because the kernel made tim-9b's prefill cheap, tim-9b (88.7 tok/s) now prefills
at roughly the same per-token rate as the small qwen-2b, so their `mt=8` cache
ratios converged (~2.7× each). The way to a **bigger** cache win is now a bigger
reused suffix `C` — see the `mt=256` rows in §7, where tim reuses 207 tokens and
turn-2 TTFT drops 4.1× vs cache-off.

---

## 5. Why the Subconscious Cache is better for long-context tasks

Everything above is measured on a *toy* three-turn conversation. The reason the
cache matters far more as context grows follows directly from the mechanics.

### 5.1 Standard prefix caching recomputes the entire tail after any middle edit

A normal KV/prefix cache reuses tokens only **up to the first divergence**. In
real multi-turn and agent workloads the divergence is almost never at the end —
an earlier turn gets **dropped, summarized, truncated, or edited** to fit the
window. Everything *after* that edit point (often the bulk of the conversation)
is recomputed from scratch, even though it is textually unchanged.

The Subconscious Cache reuses the shared **suffix** too, so the recompute is
bounded to the genuinely new tokens `D`:

```
cached : A . B . C          new input : A . C . D
standard prefix cache reuses:  A          → recompute (C + D)   = most of a long history
subconscious cache reuses:     A + C      → recompute (D only)  = just the new turn
```

**The longer the context, the larger `C` is** — and `C` is exactly the part
standard caching throws away. In this benchmark `C` was tiny (36–109 tokens) yet
already 76–83% of the prompt was reused. In a long session `C` can be thousands of
tokens; the reused fraction approaches 100% of the unchanged history and the
avoided-prefill saving grows with it.

### 5.2 Prefill cost grows with context length; decode does not

TTFT is dominated by prefill, and prefill is roughly linear (attention,
super-linear) in prompt length. Long-context tasks are typically
**prefill-heavy, decode-light** — a large prompt (retrieved docs, tool outputs,
long agent history) followed by a short answer. That is precisely the regime
where this benchmark shows the biggest wins:

- `tim-9b` cold prefill of the full history (new kernel): **3.7 s** → after
  surgery: **0.69 s**.
- Scale the history from hundreds of tokens to tens of thousands, and the cold
  prefill scales up with it, while the surgery path still only prefills the new
  turn `D`. The gap widens with every added token of context — regardless of how
  fast the prefill kernel is.

### 5.3 KV-cache memory stays bounded across a long session

With surgery, only the truly new tokens `D` allocate fresh KV each turn; the
reused prefix and suffix KV are kept in place (the suffix is repositioned, not
reallocated). Over a long multi-turn or agent session this keeps KV-cache growth
**bounded to what actually changed**, instead of paying to re-materialize the
full history on every turn. The longer the conversation, the larger the memory
saving — the same reason the latency saving grows. This is orthogonal to the
prefill kernel: a faster kernel makes recompute cheaper, but the cache avoids the
recompute *and* the memory re-allocation entirely.

### 5.4 Expensive prefill (and big reuse) amplify the effect

Because the saving is "prefill cost of the reused tokens," the relative win is
largest when per-token prefill is expensive **and/or** the reused suffix is large.
This benchmark shows both levers directly:

- **Prefill cost:** on the *old* kernel tim-9b's prefill was ~3× costlier, and its
  `mt=8` surgery ratio was 3.8× vs qwen-2b's 2.7×. The new kernel cut that cost,
  and the `mt=8` ratios converged (~2.7× each) — a clean demonstration that the
  cache's relative benefit tracks prefill cost.
- **Reuse size:** hold the kernel fixed and grow `C`. At `mt=256` tim reuses 207
  tokens (vs 134 at `mt=8`) and turn-2 TTFT drops 4.1× vs cache-off. Long context
  makes `C` large, so the benefit grows even when each token is cheap to prefill.

**In short:** as context length grows, standard caching wastes an ever-larger
tail of recompute on unchanged history, while the Subconscious Cache's cost
tracks only the new tokens. Latency saving, memory saving, and the fraction of
work avoided all increase monotonically with context length — which is why the
feature is aimed squarely at long-context, multi-turn, and agent workloads.

---

## 6. Note on an earlier contaminated pass

An earlier version of this benchmark ran `max_tokens=128` with `/flush_cache`
between reps in one long-lived server. Flush did **not** reliably clear the slot
(context checkpoints survived it), so later reps ran warm and their "cached"
figures were ordinary prefix carryover, not surgery. That pass has been
**replaced** by the fresh-cold-server-per-rep methodology used here (`mt=8` and
`mt=256`), which needs no flush and gives a genuinely cold turn-1 every rep.

---

## 7. Full summary table (median of reps)

```
config                    mt | t1_ttft t1_tot t2_ttft t2_tot | ttft_spd tot_spd | t2_cached t2_prompt t2_new | dec_tps
--------------------------------------------------------------------------------------------------------------------
qwen-2b/cache-off          8 |   1.03   1.23   0.61   0.80 |   1.68x   1.53x |     0/177        177     177 |  41.07
qwen-2b/cache-on           8 |   1.04   1.24   0.23   0.42 |   4.62x   2.96x |   134/177        177      43 |  41.37
tim-9b/cache-off  (new)    8 |   3.57   4.10   2.03   2.55 |   1.76x   1.61x |     0/177        177     177 |  15.1
tim-9b/cache-on   (new)    8 |   3.61   4.15   0.76   1.28 |   4.75x   3.25x |   134/177        177      43 |  14.8
tim-9b/cache-off  (new)  256 |   3.70   9.74   2.80   9.51 |   1.32x   1.02x |     0/249        249     249 |  13.2
tim-9b/cache-on   (new)  256 |   3.66   9.71   0.69   7.51 |   5.30x   1.29x |   207/249        249      42 |  13.4
```

- `ttft_spd` / `tot_spd` = turn1/turn2 within the same config.
- Cross-config cache benefit (cache-ON turn2 vs cache-OFF turn2): qwen-2b `mt=8`
  **2.7× TTFT**; tim-9b `mt=8` **2.7× TTFT**, `mt=256` **4.1× TTFT** (bigger reused `C`).

---

## 8. Conclusions

1. The Subconscious Cache **fires reliably** on both models, reusing **76–83%** of
   the Turn 2 prompt on this toy workload.
2. Same-model, same-prompt, the cache alone gives **2.7× (qwen-2b)** and **2.7×
   (tim-9b `mt=8`) / 4.1× (tim-9b `mt=256`)** faster turn-2 TTFT at cold cache.
3. It is a **prefill / latency + KV-memory** optimization: decode speed and
   output quality are unchanged (replies stayed on-topic).
4. The new 1.3-bit CPU kernel makes tim-9b prefill **2.7× faster** (32.9 → 88.7
   tok/s) with decode unchanged — it lowers every latency and, by making prefill
   cheaper, slightly reduces the cache's *relative* speedup while the *absolute*
   latencies all improve.
5. The cache benefit **grows with context length, with the size of the
   dropped/edited middle, and with per-token prefill cost** — which is exactly why
   it targets long-context, multi-turn, and agent workloads.

### Reproduce

```powershell
$env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = '{"preserve_thinking": true}'
$env:BRAINTREE_PP_CACHE_LOG_PATH = "$PWD\pp_cache.log"
.\bin\llama-server.exe -m C:\Users\lhyth\Desktop\models\tim-9b-1.25bit.gguf `
  -c 32768 --host 127.0.0.1 --port 8080 --suffix-cache `
  --jinja --chat-template-file qwen.jinja --reasoning off --no-mmproj
# then, in another shell:
python test-subconscious-hit.py --model tim --port 8080 --disable-thinking
```

See [COMPARISON-9B.md](COMPARISON-9B.md) for the 9B head-to-head (Qwen3.5-9B vs
tim-9b) with the same new kernel.
