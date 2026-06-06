#!/usr/bin/env bash
# living-kv llama.cpp demos — one-command installer.
#
#   curl -fsSL https://raw.githubusercontent.com/helgard-orlm/living-kv-cache/master/llamacpp/install.sh | bash
#
# What it does (idempotent, everything under ~/living-kv-cache by default):
#   1. checks tools (git, g++, cmake, curl)
#   2. clones llama.cpp pinned to a known-good tag and builds libllama (CUDA if available, else CPU)
#   3. builds the poc18 demo binaries
#   4. downloads a small demo model (TinyLlama Q4, ~0.6 GB); FULL=1 also pulls Qwen2.5-7B-1M (~4.7 GB)
#   5. pulls the nomic embedder via ollama if ollama is installed (needed for poc18b..g catalogs)
#   6. runs poc18a as a smoke test
#
# Env overrides: LIVINGKV_DIR, LLAMA_TAG, FULL=1, SKIP_TEST=1
set -euo pipefail

DIR="${LIVINGKV_DIR:-$HOME/living-kv-cache}"
LLAMA_TAG="${LLAMA_TAG:-b9297}"   # tag the demos were tested against (state format is build-sensitive)
REPO="https://github.com/helgard-orlm/living-kv-cache"
TINY_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
QWEN_URL="https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/resolve/main/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"

say()  { printf '\033[1;32m[living-kv]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[living-kv]\033[0m %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || fail "missing '$1' — install it first (e.g. sudo apt install $2)"; }
need git git; need g++ g++; need cmake cmake; need curl curl
JOBS="$(nproc 2>/dev/null || echo 4)"

# 1. repo
if [ -d "$DIR/.git" ]; then say "repo exists: $DIR (git pull)"; git -C "$DIR" pull --ff-only || true
else say "cloning $REPO -> $DIR"; git clone --depth 1 "$REPO" "$DIR"; fi
cd "$DIR"

# 2. llama.cpp (pinned) + libllama
if [ ! -f llama.cpp/build/bin/libllama.so ]; then
    [ -d llama.cpp ] || { say "cloning llama.cpp @$LLAMA_TAG"; git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp; }
    CUDA_FLAG=""
    if command -v nvcc >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/nvcc ]; then
        CUDA_FLAG="-DGGML_CUDA=ON"; say "CUDA found -> GPU build"
    else
        say "no CUDA toolkit -> CPU build (demos work, just slower)"
    fi
    say "building libllama (-j$JOBS, takes a few minutes)..."
    cmake -S llama.cpp -B llama.cpp/build $CUDA_FLAG -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF >/dev/null
    cmake --build llama.cpp/build -j"$JOBS" --target llama >/dev/null
fi
LL="$DIR/llama.cpp"

# 3. demo binaries
say "building poc18 demos..."
mkdir -p bin
for f in llamacpp/poc18*.cpp; do
    out="bin/$(basename "${f%_*.cpp}" | sed 's/\.cpp$//')"
    g++ -O2 -o "$out" "$f" -I "$LL/include" -I "$LL/ggml/include" -L "$LL/build/bin" -lllama -Wl,-rpath,"$LL/build/bin"
done
say "built: $(ls bin | tr '\n' ' ')"

# 4. models
mkdir -p models
TINY="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
[ -s "$TINY" ] || { say "downloading TinyLlama demo model (~0.6 GB)..."; curl -fL --progress-bar -o "$TINY" "$TINY_URL"; }
QWEN="models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"
if [ "${FULL:-0}" = "1" ] && [ ! -s "$QWEN" ]; then
    say "FULL=1 -> downloading Qwen2.5-7B-Instruct-1M Q4_K_M (~4.7 GB)..."
    curl -fL --progress-bar -o "$QWEN" "$QWEN_URL"
fi

# 5. catalog embedder (poc18b..g need it; poc18a does not)
if command -v ollama >/dev/null 2>&1; then
    ollama list 2>/dev/null | grep -q nomic-embed-text || { say "pulling nomic-embed-text via ollama..."; ollama pull nomic-embed-text || true; }
else
    say "NOTE: ollama not found. poc18a works without it; poc18b..g need the catalog embedder:"
    say "      curl -fsSL https://ollama.com/install.sh | sh   &&   ollama pull nomic-embed-text"
fi

# 6. smoke test
if [ "${SKIP_TEST:-0}" != "1" ]; then
    say "smoke test: poc18a (bit-exact KV cut/restore roundtrip)..."
    ./bin/poc18a "$TINY" | grep -E "^(A |B |C |PASS|CHECK)" || fail "smoke test did not run"
fi

say ""
say "Done. Everything lives in: $DIR"
say "Try next:"
say "  cd $DIR"
say "  ./bin/poc18a $TINY                              # roundtrip primitive"
if [ -s "$QWEN" ]; then M="$QWEN"; else M="$TINY  (re-run with FULL=1 for the real 1M model)"; fi
say "  ./bin/poc18b ${M%% *} 8192                       # catalog + selective restore"
say "  ./bin/poc18c ${M%% *} 131072 4096 q8_0           # 131k tokens through a 4k buffer"
say "  ./bin/poc18d ${M%% *} 8192                       # living dynamics (decay/revival)"
say "  ./bin/poc18g ${M%% *} 65536 budget               # decode speed where full cache OOMs"
