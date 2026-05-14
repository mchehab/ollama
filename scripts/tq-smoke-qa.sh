#!/usr/bin/env bash
# tq-smoke-qa.sh — TurboQuant smoke test: start server, send factual prompt,
# assert coherent response and gpu-native path, report throughput.
#
# Usage:
#   ./scripts/tq-smoke-qa.sh
#   SMOKE_MODEL=gemma4:e2b OLLAMA_MODELS=/usr/share/ollama/.ollama/models \
#     USE_CHAT=1 ./scripts/tq-smoke-qa.sh
#
# Env vars:
#   SMOKE_MODEL       model tag (default: llama3.2:3b)
#   OLLAMA_MODELS     models directory (default: ~/.ollama/models)
#   PRESETS           space-separated preset list (default: all 6 ship presets)
#   SMOKE_PORT        serve port (default: 11448)
#   SMOKE_PROMPT      prompt text (default: "The capital of France is")
#   SMOKE_EXPECT      substring expected in response (default: Paris)
#   NUM_PREDICT       tokens to generate (default: 32)
#   USE_CHAT          1 = use /api/chat (instruction models), 0 = /api/generate (default)

set -euo pipefail

SMOKE_MODEL="${SMOKE_MODEL:-llama3.2:3b}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
PRESETS="${PRESETS:-tq2 tq3 tq4 tq2k tq3k tq4k}"
SMOKE_PORT="${SMOKE_PORT:-11448}"
SMOKE_PROMPT="${SMOKE_PROMPT:-The capital of France is}"
SMOKE_EXPECT="${SMOKE_EXPECT:-Paris}"
NUM_PREDICT="${NUM_PREDICT:-256}"
USE_CHAT="${USE_CHAT:-0}"
SMOKE_SLEEP="${SMOKE_SLEEP:-5}"
HOST="127.0.0.1:${SMOKE_PORT}"
BINARY="${BINARY:-./dist/bin/ollama}"

pass=0; fail=0

run_preset() {
    local preset="$1"
    local logfile
    logfile=$(mktemp /tmp/tq-smoke-XXXXXX.log)

    OLLAMA_NEW_ENGINE=1 \
    OLLAMA_FLASH_ATTENTION=1 \
    OLLAMA_KV_CACHE_TYPE="$preset" \
    OLLAMA_MODELS="$OLLAMA_MODELS" \
    OLLAMA_HOST="$HOST" \
        "$BINARY" serve >"$logfile" 2>&1 &
    local srv_pid=$!

    # wait for server ready (up to 60s)
    local i=0
    until curl -sf "http://$HOST/api/version" >/dev/null 2>&1; do
        sleep 1; i=$((i+1))
        if [ $i -ge 60 ]; then
            echo "FAIL [$preset] server never became ready"
            kill "$srv_pid" 2>/dev/null; wait "$srv_pid" 2>/dev/null
            sleep 2
            rm -f "$logfile"; fail=$((fail+1)); return
        fi
    done

    local response
    if [ "$USE_CHAT" = "1" ]; then
        response=$(curl -s "http://$HOST/api/chat" -d \
            "{\"model\":\"$SMOKE_MODEL\",
              \"messages\":[{\"role\":\"user\",\"content\":\"$SMOKE_PROMPT\"}],
              \"stream\":false,\"options\":{\"num_predict\":$NUM_PREDICT,\"num_ctx\":512}}" \
            | jq -r '.message.content // empty' 2>/dev/null || true)
    else
        response=$(curl -s "http://$HOST/api/generate" -d \
            "{\"model\":\"$SMOKE_MODEL\",\"prompt\":\"$SMOKE_PROMPT\",
              \"stream\":false,\"options\":{\"num_predict\":$NUM_PREDICT,\"num_ctx\":512}}" \
            | jq -r '.response // empty' 2>/dev/null || true)
    fi

    kill "$srv_pid" 2>/dev/null; wait "$srv_pid" 2>/dev/null
    sleep "$SMOKE_SLEEP"  # let port fully release before next server start

    # assertions
    local ok=1

    if echo "$response" | grep -qi "$SMOKE_EXPECT"; then
        echo "  response OK: $(echo "$response" | tr '\n' ' ' | head -c 80)"
    else
        echo "  FAIL: response missing '$SMOKE_EXPECT' — got: $(echo "$response" | head -c 120)"
        ok=0
    fi

    local activation
    activation=$(grep "TQ_ACTIVATION" "$logfile" | grep "gpu_active=true" | tail -1 || true)
    if [ "$preset" = "f16" ]; then
        : # no TQ_ACTIVATION expected for f16
    elif [ -n "$activation" ]; then
        echo "  TQ_ACTIVATION OK: $(echo "$activation" | grep -o 'gpu_active=true.*')"
    else
        local last_activation
        last_activation=$(grep "TQ_ACTIVATION" "$logfile" | tail -1 || true)
        echo "  FAIL: no gpu_active=true TQ_ACTIVATION — last: $last_activation"
        ok=0
    fi

    rm -f "$logfile"

    if [ $ok -eq 1 ]; then
        echo "PASS [$preset]"
        pass=$((pass+1))
    else
        echo "FAIL [$preset]"
        fail=$((fail+1))
    fi
}

echo "=== tq-smoke-qa: $SMOKE_MODEL ==="
echo "    presets: $PRESETS"
echo "    endpoint: $([ "$USE_CHAT" = "1" ] && echo "/api/chat" || echo "/api/generate")"
echo "    expect:  '$SMOKE_EXPECT' in response to '$SMOKE_PROMPT'"
echo ""

for preset in $PRESETS; do
    echo "--- $preset ---"
    run_preset "$preset"
    echo ""
done

echo "=== result: $pass passed, $fail failed ==="
[ $fail -eq 0 ]
