#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:9910}"
SESSION_ID="${SESSION_ID:-smoke-test}"
STRICT="${STRICT:-0}"
PASS=0
FAIL=0
TMP_RESP="$(mktemp)"

cleanup() {
  rm -f "$TMP_RESP"
}
trap cleanup EXIT

red() { printf '\033[0;31m%s\033[0m\n' "$1"; }
green() { printf '\033[0;32m%s\033[0m\n' "$1"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$1"; }

check_contains() {
  local name="$1"
  local body="$2"
  local expect="$3"
  if echo "$body" | grep -Eq "$expect"; then
    green "[PASS] $name"
    PASS=$((PASS + 1))
  else
    red "[FAIL] $name"
    echo "  expected contains: $expect"
    echo "  actual: ${body:0:240}"
    FAIL=$((FAIL + 1))
  fi
}

check_status_200() {
  local name="$1"
  local url="$2"
  local code
  : > "$TMP_RESP"
  code=$(curl -s -o "$TMP_RESP" -w '%{http_code}' "$url" || true)
  if [[ "$code" == "200" ]]; then
    green "[PASS] $name"
    PASS=$((PASS + 1))
  else
    red "[FAIL] $name"
    echo "  http_code=$code"
    echo "  body=$(head -c 240 "$TMP_RESP" 2>/dev/null || true)"
    FAIL=$((FAIL + 1))
  fi
}

yellow "== SuperBizAgent Python Smoke Test =="
yellow "BASE_URL=$BASE_URL"
yellow "STRICT=$STRICT"

# 1) health
check_status_200 "GET /milvus/health" "$BASE_URL/milvus/health"

# 2) chat
chat_resp=$(curl -s -X POST "$BASE_URL/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"Id\":\"$SESSION_ID\",\"Question\":\"你好，请回复测试通过\"}" || true)
check_contains "POST /api/chat returns success" "$chat_resp" '"code":200'
check_contains "POST /api/chat has answer" "$chat_resp" '"answer"'

# 3) chat_stream (SSE)
chat_stream_resp=$(curl -sN -X POST "$BASE_URL/api/chat_stream" \
  -H 'Content-Type: application/json' \
  -d "{\"Id\":\"$SESSION_ID\",\"Question\":\"现在几点\"}" | head -n 80 || true)
check_contains "POST /api/chat_stream emits message" "$chat_stream_resp" 'event: message'
check_contains "POST /api/chat_stream emits done" "$chat_stream_resp" '"type": "done"|"type":"done"'

# 4) ai_ops (SSE)
aiops_resp=$(curl -sN -X POST "$BASE_URL/api/ai_ops" | head -n 50 || true)
check_contains "POST /api/ai_ops emits message" "$aiops_resp" 'event: message'
check_contains "POST /api/ai_ops contains analysis text" "$aiops_resp" '正在读取告警|分析输入|报告'

if [[ "$STRICT" == "1" ]]; then
  check_contains "POST /api/chat_stream contains eventKey" "$chat_stream_resp" '"eventKey"\s*:\s*"assistant\.(content\.delta|tool\.started|tool\.finished|done|error)"'
  check_contains "POST /api/ai_ops contains planner decision" "$aiops_resp" 'decision=PLAN|decision=EXECUTE|decision=FINISH'
  check_contains "POST /api/ai_ops contains loop progress" "$aiops_resp" 'Round [0-9]+|闭环执行完成'
  check_contains "POST /api/ai_ops contains eventKey" "$aiops_resp" '"eventKey"\s*:\s*"assistant\.(content\.delta|planner\.step|done|error)"'
  check_contains "POST /api/ai_ops contains plannerDecision" "$aiops_resp" '"plannerDecision"\s*:\s*"(PLAN|EXECUTE|FINISH)"'
  check_contains "POST /api/ai_ops contains plannerStatus" "$aiops_resp" '"plannerStatus"\s*:\s*"(PLAN|EXECUTE|FINISH)"'
  check_contains "POST /api/ai_ops contains executorStatus" "$aiops_resp" '"executorStatus"\s*:\s*"(PENDING|SUCCESS|FAILED|INFO|SKIPPED)"'
fi

# 5) session info
session_resp=$(curl -s "$BASE_URL/api/chat/session/$SESSION_ID" || true)
check_contains "GET /api/chat/session/{id}" "$session_resp" '"sessionId"'

# 6) clear session
clear_resp=$(curl -s -X POST "$BASE_URL/api/chat/clear" \
  -H 'Content-Type: application/json' \
  -d "{\"Id\":\"$SESSION_ID\"}" || true)
check_contains "POST /api/chat/clear" "$clear_resp" '"code":200|"message":"success"'

echo
yellow "== Smoke Test Summary =="
echo "PASS=$PASS"
echo "FAIL=$FAIL"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi

green "All checks passed."
