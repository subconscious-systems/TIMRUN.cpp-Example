"""Smoke test: trigger a Subconscious Cache hit on a running sglang server.

Construction
------------
Turn 1 (warmup) sends a 3-turn conversation::

    [system, U1, A1, U2, A2, U3]

The server generates R1 (the response to U3) and ``cache_finished_req`` inserts
the full chain into the radix tree::

    cached chain = system + U1 + A1 + U2 + A2 + U3 + R1 + EOS

The Mamba/SSM state anchored at the end of this chain represents "model has
processed everything through R1's last token".

Turn 2 (target) drops the *middle* turn but **adds R1 back as an assistant
message** and appends a brand-new user follow-up U4::

    [system, U1, A1, U3, R1, U4]

This is the canonical ``A · C · D`` shape against the cached ``A · B · C``:

    A = system + U1 + A1           (prefix match)
    B = U2 + A2                    (gap; cached only)
    C = U3 + R1 + EOS              (suffix match; in both)
    D = U4 + <generation prompt>   (new tail; turn 2 only)

Standard prefix matching stops where the cached chain's `U2` diverges from
the new input's `U3`. The Subconscious matcher DFS-scans the subtree under
that exit node, Z-matches the cached tail `U2 + A2 + U3 + R1 + EOS` against
the new remaining `U3 + R1 + U4 + <gen prompt>`, finds the long shared
suffix `U3 + R1 + EOS`, and triggers surgery: A's KV is reused, C's KV is
copied with one RoPE rotation back to position |A|, the Mamba state at the
cached chain's end is copied into the new request (positionally aligned to
|A|+|C|, i.e. right before D), and only D gets a real forward.

Why R1 must be included
-----------------------
If Turn 2 stopped at U3 (without R1), the matcher would still find prefix A,
but the cached tail's suffix `U3 + R1 + EOS` would not appear as a prefix of
the new remaining `U3 + <gen prompt>` — the prompt diverges right after U3.
More importantly, the Mamba anchor at the cached chain's end represents the
state AFTER seeing R1+EOS; transplanting it into a request that's only at
"end of U3, about to generate" would produce a positional mismatch between
the recurrent state and the attention KV. Including R1 in Turn 2 makes the
anchor's position match.

Requirements
------------
The target server must be launched with ``--subconscious-cache`` and radix
cache enabled. ``page_size >= 1`` is supported and there is no attention-backend
restriction (surgery operates on the standard MHA/MLA KV pool directly)::

    python -m sglang.launch_server \\
        --model-path <model> \\
        --subconscious-cache \\
        [--attention-backend flashinfer|triton|fa3] \\
        [--page-size 1|16|...]   # default 1; >1 exercises the paged matcher

The same matcher + surgery serve two cache paths:
  * ``MambaRadixCache`` — hybrid attention + Mamba/SSM models (Nemotron-H,
    Qwen3-Next, Kimi K2.5); page_size>1 needs the extra-buffer SSM mode.
  * ``RadixCache`` — pure-attention / MLA models (DeepSeek, Kimi K2.5 text).

For page_size > 1, launch with e.g. ``--page-size 16`` and re-run: the matcher's
sub-page common-prefix remainder (``r``) path fires whenever the prune boundary
is not page-aligned. NOTE on donor-branch eviction after surgery, which differs
by cache path:
  * RadixCache (non-recurrent / MLA, e.g. Kimi/DeepSeek): the donor is evicted
    for ANY page_size, so a ``[subconscious] donor eviction (RadixCache): ...``
    line appears regardless of page_size.
  * MambaRadixCache: the donor is evicted only at page_size == 1; at
    page_size > 1 it is reclaimed by LRU and no mamba donor-evict line appears
    (its mamba-anchor transplant is not yet implemented for page_size > 1).

How to read the result
----------------------
1. **Server log** (most reliable). Look for these lines emitted while Turn 2
   is being scheduled::

        [subconscious] surgery_HIT  prefix_len=... gap_len=... suffix_len=...
        [subconscious] mamba override: src_idx=... dst_idx=... all_layers_match=True
        [subconscious] surgery DONE: rewrote prefix_indices for 1 req(s)

2. **Client side** (latency signal). Turn 2's TTFT should drop by roughly the
   prefill cost of `B + C`. The script prints both wall-clock latencies; on a
   hybrid-Mamba model with non-trivial `|B|+|C|`, Turn 2 is typically ~2-5x
   faster than Turn 1.

3. **Token equality**. The script also prints both replies; Turn 2's reply
   should be *semantically* about comparing Paris and London (just like
   Turn 1's), confirming the model didn't drift on a corrupted recurrent
   state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List

try:
    import requests
except ImportError:
    print("This script needs `requests` (pip install requests)", file=sys.stderr)
    sys.exit(1)


SYSTEM_PROMPT = (
    "You are a helpful and concise assistant. Answer clearly in plain English."
)

# U1 / A1: opening turn — establishes a common prefix.
U1 = "What is the capital of France, and what is it best known for?"
A1 = (
    "The capital of France is Paris. It is best known for the Eiffel Tower, "
    "the Louvre Museum, world-class cuisine, and a long tradition of art, "
    "literature, philosophy, and fashion that has shaped European culture "
    "for centuries."
)

# U2 / A2: the *middle* turn that will be dropped in Turn 2.
# Make it long enough that the prune is non-trivial.
U2 = (
    "Tell me a long, detailed historical fact about Paris that a first-time "
    "tourist would find interesting before they visit. Aim for at least four "
    "sentences."
)
A2 = (
    "Paris is divided into twenty arrondissements arranged in a clockwise "
    "spiral starting from the historic center on the Right Bank of the "
    "Seine. The numbering begins at the Louvre area and winds outward, so "
    "the lowest numbers cover the oldest neighbourhoods and the highest "
    "numbers reach the modern outer districts. The Seine itself was the "
    "city's original commercial lifeline, and many of its bridges -- "
    "including the famous Pont Neuf, whose name confusingly means 'new "
    "bridge' even though it is the oldest standing one in Paris -- have "
    "carried foot and carriage traffic for centuries. Walking along the "
    "river from the 4th arrondissement to the 7th is a fast way to see "
    "Notre-Dame, the Conciergerie, and the Musee d'Orsay in one afternoon."
)

# U3: the third user message — IDENTICAL across Turn 1 and Turn 2. In Turn 1
# it is the *last* user (the one the model answers). In Turn 2 it sits as
# history, followed by Turn 1's R1 + the new U4.
U3 = (
    "Now compare Paris with London in exactly three concise sentences. "
    "Focus on culture, transport, and food."
)

# U4: the brand-new user message Turn 2 actually asks the model to answer.
# This becomes ``D`` — the only segment that needs a real forward in Turn 2.
U4 = (
    "Great. One more: which of those two cities is generally cheaper for a "
    "tourist on a weeklong trip, and by how much roughly?"
)


def _warmup_messages() -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": U1},
        {"role": "assistant", "content": A1},
        {"role": "user", "content": U2},
        {"role": "assistant", "content": A2},
        {"role": "user", "content": U3},
    ]


def _target_messages(r1: str) -> List[Dict[str, str]]:
    """Drop U2/A2, include the *actual* R1 from Turn 1 as an assistant
    message, then append U4. This is the canonical A·C·D vs cached A·B·C
    shape — see module docstring for why R1 must be included.

    ``r1`` is Turn 1's reply content exactly as the server returned it; do
    NOT trim or rewrite, otherwise the suffix C will diverge from the cached
    chain and the matcher will reject the hit (or accept a much shorter L).
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": U1},
        {"role": "assistant", "content": A1},
        # dropped: U2 + A2  (this is B)
        # {"role": "user", "content": U2},
        # {"role": "assistant", "content": A2},
        {"role": "user", "content": U3},
        {"role": "assistant", "content": r1},   # ← Turn 1's reply, becomes part of C
        {"role": "user", "content": U4},        # ← D
    ]


def _post_chat(
    base: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    extra_body: Dict[str, Any] | None = None,
    timeout: int = 600,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if extra_body:
        payload.update(extra_body)
    t0 = time.perf_counter()
    r = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=timeout)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    body["_client_elapsed_s"] = elapsed
    return body


def _try_flush(base: str) -> None:
    try:
        requests.post(f"{base}/flush_cache", timeout=10)
        time.sleep(0.3)
    except Exception as e:
        print(f"[warn] /flush_cache failed ({e!r}); continuing anyway.")


def _summarize(label: str, resp: Dict[str, Any]) -> None:
    choice = resp["choices"][0]
    print(choice)
    content = (choice.get("message") or {}).get("content", "") or ""
    finish = choice.get("finish_reason")
    usage = resp.get("usage", {})
    elapsed = resp.get("_client_elapsed_s", float("nan"))
    print(f"\n--- {label} ---")
    print(f"  client_elapsed_s : {elapsed:7.3f}")
    print(f"  finish_reason    : {finish}")
    print(f"  usage            : {usage}")
    head = content.strip().replace("\n", " ")
    if len(head) > 240:
        head = head[:240] + " ..."
    print(f"  reply_head       : {head}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=7007)
    p.add_argument(
        "--model",
        required=True,
        help="Model name as the running server reports it "
        "(must match what was passed to --model-path / --served-model-name).",
    )
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument(
        "--no-flush",
        action="store_true",
        help="Skip the /flush_cache call before warmup (use to test across "
        "an already-populated cache).",
    )
    p.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass chat_template_kwargs={'enable_thinking': False}. Useful for "
        "reasoning models (e.g. Nemotron) to keep responses short and make "
        "the latency delta easy to read.",
    )
    args = p.parse_args()

    base = f"http://{args.host}:{args.port}"

    extra: Dict[str, Any] = {
        "chat_template_kwargs": {
            "subtask_buffer_size": 2,
            "min_subtask_length": 2
        }
    }
    if args.disable_thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        # Thinking-mode templates (Nemotron, Qwen3 reasoning) emit
        # ``<think>...</think>`` into the cached chain at R1's position, but
        # most templates also strip historical thinking when re-rendering
        # ``R1`` as an assistant message in Turn 2 (``truncate_history_thinking``).
        # That makes Turn 2's rendered C diverge from the cached C right inside
        # the think block, dropping the matched suffix L below threshold and
        # killing the hit. Strongly recommend ``--disable-thinking`` for this
        # smoke test.
        print(
            "[warn] thinking is enabled. For Nemotron / Qwen3-Reasoning style "
            "templates, historical thinking is stripped when R1 is re-rendered "
            "in Turn 2, which makes the suffix C diverge from the cached chain "
            "and breaks the hit. Pass --disable-thinking unless you know your "
            "template preserves history thinking verbatim."
        )

    if not args.no_flush:
        print("[flush] /flush_cache")
        _try_flush(base)

    print(
        "\n[turn 1] warmup: sending 3-turn conversation "
        "[sys, U1, A1, U2, A2, U3] -> server caches "
        "sys+U1+A1+U2+A2+U3+R1+EOS"
    )
    warm = _post_chat(
        base, args.model, _warmup_messages(), args.max_tokens, extra_body=extra
    )
    _summarize("turn 1 (warmup, cold path)", warm)

    # Extract Turn 1's reply verbatim — this is the R1 we splice into Turn 2's
    # history. Must NOT be trimmed/rewritten or the suffix C will diverge.
    try:
        r1 = warm["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        print("[error] Turn 1 response had no choices[0].message.content; abort.")
        return 2
    if not r1.strip():
        print("[error] Turn 1 returned empty content; nothing to splice as R1.")
        return 2

    # Small gap so the server's cache_finished_req for turn 1 has definitely run
    # before turn 2 enters the scheduler. Not strictly required (single-threaded
    # scheduler ordering already guarantees it) but makes log reading easier.
    time.sleep(0.5)

    print(
        "\n[turn 2] target: drop U2/A2, splice Turn 1's actual R1 back as "
        "history, append new U4 -> should subconscious-hit\n"
        "         messages = [sys, U1, A1, U3, R1, U4]\n"
        f"         R1 length (chars) = {len(r1)}"
    )
    hit = _post_chat(
        base, args.model, _target_messages(r1), args.max_tokens, extra_body=extra
    )
    _summarize("turn 2 (target, should hit subconscious)", hit)

    e1 = warm.get("_client_elapsed_s", float("nan"))
    e2 = hit.get("_client_elapsed_s", float("nan"))
    speedup = (e1 / e2) if e2 and e2 > 0 else float("nan")
    print(
        f"\n[result] turn1={e1:.3f}s  turn2={e2:.3f}s  speedup={speedup:.2f}x"
    )

    print(
        "\n[verify] Look in the sglang server log for these lines emitted "
        "while Turn 2 was being scheduled:\n"
        "    [subconscious] surgery_HIT  prefix_len=|A| gap_len=|B| suffix_len=|C| ...\n"
        "    [subconscious] mamba override: ... all_layers_match=True\n"
        "    [subconscious] surgery DONE: rewrote prefix_indices for 1 req(s)\n"
        "If those three lines appear for Turn 2, the cache fired correctly.\n"
        "Expected rough magnitudes for this test:\n"
        "    |A| ~ tokens of (sys + U1 + A1)            ~ a few hundred\n"
        "    |B| ~ tokens of (U2 + A2)                  ~ a few hundred\n"
        "    |C| ~ tokens of (U3 + R1 + <|im_end|>)     >> 8 (min suffix len)\n"
        "If you see [subconscious] no-surgery ... std_prefix_partial instead, "
        "the matcher found A but failed the Z-match — most common cause is a "
        "chat-template difference between cached and re-rendered R1 (e.g. "
        "history thinking stripped). Re-run with --disable-thinking.\n"
    )

    # Heuristic client-side signal (defensive — server log is authoritative).
    if speedup == speedup and speedup < 1.2:  # NaN-safe `>= 1.2` check
        print(
            "[warn] turn 2 was not noticeably faster than turn 1. This does "
            "NOT prove the cache missed (decoding cost dominates for short "
            "max_tokens), but if your server log also shows no surgery_HIT, "
            "see the troubleshooting hints above."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
