# TIMRUN.cpp Server

A self-contained Windows x64 build of `llama-server` with the **Subconscious Cache**
feature enabled, plus everything needed to run it and to reproduce a cache hit.

## Contents

```
HP/
  README.md                  this file
  qwen.jinja                 chat template REQUIRED to hit the Subconscious Cache
  cacert.pem                 CA bundle used by -hf HTTPS downloads
  test-subconscious-hit.py   script that triggers a Subconscious Cache hit
  bin/
    llama-server.exe         the server
    *.dll                    all runtime dependencies (llama/ggml/mtmd, OpenSSL,
                             MSVC C++ runtime, OpenMP) - no install required
```

## Requirements

- Windows 10/11, x64.
- Nothing to install: the MSVC runtime, OpenMP, and OpenSSL DLLs are all bundled in `bin/`.
- (Only to run the test script) Python 3 with `requests`: `pip install requests`.

The server exposes an OpenAI-compatible API at `http://<host>:<port>/v1/chat/completions`
(and `/health`, `/props`, `/completion`).

---

## 1. Start the server (with the Subconscious Cache enabled)

The Subconscious Cache only fires when you launch with the **bundled `qwen.jinja`
template** and the flags below. Open a terminal in this folder.

**cmd.exe:**
```bat
set SSL_CERT_FILE=%CD%\cacert.pem
set LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"preserve_thinking": true}
bin\llama-server.exe -m PATH\TO\model.gguf -c 32768 --host 127.0.0.1 --port 8080 --suffix-cache --jinja --chat-template-file qwen.jinja --reasoning off
```

**PowerShell:**
```powershell
$env:SSL_CERT_FILE = "$PWD\cacert.pem"
$env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = '{"preserve_thinking": true}'
.\bin\llama-server.exe -m PATH\TO\model.gguf -c 32768 --host 127.0.0.1 --port 8080 --suffix-cache --jinja --chat-template-file qwen.jinja --reasoning off
```

What each part does:

| flag | why |
|------|-----|
| `--suffix-cache` | Turns the Subconscious Cache ON (it is OFF by default). |
| `--jinja --chat-template-file qwen.jinja` | REQUIRED. The bundled template renders assistant history with the same `<think>` block the reply was generated with, so the cache can match the shared suffix. The models' own built-in templates strip it and the cache never fires. |
| `LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"preserve_thinking": true}` | Keeps history reasoning verbatim (passed via env var to avoid shell quoting issues). |
| `--reasoning off` | These are thinking models; this keeps the `<think>` block symmetric between generation and history so the suffix matches. |
| `SSL_CERT_FILE=...\cacert.pem` | Only needed for `-hf` downloads (HTTPS). Not needed when loading a local file with `-m`. |
| `-c 32768` | Context size. Use what your RAM allows. |

Quick check once it is up:
```bat
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":64}"
```

---

## 2. Load a model

### Recommended models via `-hf` (auto-download from HuggingFace)

`-hf` downloads the model over HTTPS, so you must set `SSL_CERT_FILE=%CD%\cacert.pem`
first (see above). Models are cached under `%USERPROFILE%\.cache\huggingface\hub`
and reused on the next run.

**Model 1 - Qwen3.5-2B (Q4_K_M, ~1.2 GB), a small general model:**
```bat
set SSL_CERT_FILE=%CD%\cacert.pem
set LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"preserve_thinking": true}
bin\llama-server.exe -hf unsloth/Qwen3.5-2B-GGUF:Q4_K_M -c 32768 --host 127.0.0.1 --port 8080 --suffix-cache --jinja --chat-template-file qwen.jinja --reasoning off
```

**Model 2 - tim-9b (1.3-bit QAT, ~3 GB), Subconscious's model:**
```bat
set SSL_CERT_FILE=%CD%\cacert.pem
set LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"preserve_thinking": true}
bin\llama-server.exe -hf SubconsciousDev/tim-9b-1.3bit-gguf -c 32768 --host 127.0.0.1 --port 8080 --suffix-cache --jinja --chat-template-file qwen.jinja --reasoning off
```

Notes:
- The `-hf` form is `<user>/<repo>[:quant]`. For Qwen we pin `:Q4_K_M`; for tim-9b the
  repo has a single file so no tag is needed.
- Add `--no-mmproj` if you only need text and want to skip the multimodal projector download.
- You may see `failed to create symlink ... switching to degraded mode`. It is harmless -
  Windows needs admin / Developer Mode to create symlinks, so it copies the file instead.
  Enable Developer Mode or run the terminal as Administrator to silence it.

### Manual download (no HTTPS / no `cacert.pem` needed)

Download the `.gguf` from the HuggingFace page in a browser, then load it with `-m`:
```bat
bin\llama-server.exe -m C:\models\model.gguf -c 32768 --host 127.0.0.1 --port 8080 --suffix-cache --jinja --chat-template-file qwen.jinja --reasoning off
```
- Qwen3.5-2B: https://huggingface.co/unsloth/Qwen3.5-2B-GGUF  (pick the `Q4_K_M` file)
- tim-9b: https://huggingface.co/SubconsciousDev/tim-9b-1.3bit-gguf/blob/main/tim-9b-1.3bit.gguf

---

## 3. Subconscious Cache

### What it is

A normal KV cache only helps when a new request shares a **prefix** with a cached one.
But in multi-turn chat / agent workloads the new prompt often shares a prefix **and** a
suffix with the cache and differs only in the middle - for example an earlier turn was
dropped, summarized, or edited. Standard prefix caching stops at the first difference and
recomputes everything after it.

The **Subconscious Cache** recovers that reuse. Writing the cached sequence as `A . B . C`
and the new input as `A . C . D`:

- `A` - shared prefix, reused directly.
- `B` - the middle that only exists in the cache; dropped.
- `C` - shared suffix; its KV is kept and its positions are shifted left to sit right
  after `A` (no recompute).
- `D` - the only genuinely new part; the only thing the model actually runs.

The server logs this split as `prefix / gap / suffix / new` (= A / B / C / D).

### The included test: `test-subconscious-hit.py`

The script sends exactly the `A . C . D` shape:

- **Turn 1 (warmup):** `[system, U1, A1, U2, A2, U3]` - the server caches the whole chain
  plus the reply `R1` it generates.
- **Turn 2 (target):** `[system, U1, A1, U3, R1, U4]` - it **drops the middle turn**
  (`U2/A2`), splices Turn 1's actual reply `R1` back in as history, and asks a new
  question `U4`.

So `A = system+U1+A1`, `C = U3+R1`, `D = U4`. The server matches `A` and `C`, drops the
gap, and performs the reuse ("surgery").

Run it against a server started as in section 1 (use `--disable-thinking` so the model's
`<think>` block stays symmetric):
```bat
python test-subconscious-hit.py --model tim --port 8080 --disable-thinking
```

To see the hit on the server side, start the server with an extra env var and watch the file:
```bat
set BRAINTREE_PP_CACHE_LOG_PATH=%CD%\pp_cache.log
```
Turn 2 then logs a line like:
```
surgery   incoming=245  cache=401  |  prefix=98  gap=198  suffix=105  new=42
```
You can also see it in any client: Turn 2's response `usage.prompt_tokens_details.cached_tokens`
will be large (only `new` tokens are freshly computed).

### Why THIS file hits (and a naive replay might not)

The one hard requirement is that the re-sent `R1` tokenizes **identically** to the `R1`
that is already in the cache. These are thinking models, so their built-in chat template
**drops the `<think>` block when re-rendering an assistant turn as history** - which makes
the suffix diverge and the cache miss. The bundled `qwen.jinja` keeps the block, and
`--reasoning off` / `--disable-thinking` keeps it symmetric between generation and history.
That is why you must launch with `--chat-template-file qwen.jinja`.

### Why it is worth it

- **Latency.** Turn 2 skips the prefill of `B + C` and only computes `D`. Measured here:
  tim-9b went from 4.9 s (cold) to 2.2 s (~2.2x), and the full script run went 17.8 s ->
  10.7 s (1.66x). The larger the dropped middle `B` and reused suffix `C`, the bigger the win.
- **KV-cache memory on long sessions.** Only the truly new tokens (`D`) take fresh KV each
  turn; the reused prefix and suffix are not re-allocated. Over a long multi-turn / agent
  session this keeps KV-cache growth bounded instead of paying for the full history on every
  turn - the longer the conversation, the larger the memory saving.
