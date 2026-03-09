#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:9910}"
TOPIC="${TOPIC:-熬夜后如何快速恢复气色}"
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
    echo "  actual: ${body:0:300}"
    FAIL=$((FAIL + 1))
  fi
}

yellow "== XHS Workflow Smoke Test =="
yellow "BASE_URL=$BASE_URL"
yellow "TOPIC=$TOPIC"

resp=$(curl -s -X POST "$BASE_URL/api/xhs/generate" \
  -H 'Content-Type: application/json' \
  -d "{\"topic\":\"$TOPIC\"}" || true)

check_contains "POST /api/xhs/generate returns code=200" "$resp" '"code":200'
check_contains "POST /api/xhs/generate returns success" "$resp" '"success":true'
check_contains "POST /api/xhs/generate includes draft" "$resp" '"draft":"'
check_contains "POST /api/xhs/generate includes iterations" "$resp" '"iterations":'
check_contains "POST /api/xhs/generate includes feedback" "$resp" '"feedback":"'

stream_resp=$(curl -sN -X POST "$BASE_URL/api/xhs/generate_stream" \
  -H 'Content-Type: application/json' \
  -d "{\"topic\":\"$TOPIC\"}" | head -n 80 || true)

check_contains "POST /api/xhs/generate_stream emits message" "$stream_resp" 'event: message'
check_contains "POST /api/xhs/generate_stream emits step event" "$stream_resp" 'assistant\.xhs\.step'
check_contains "POST /api/xhs/generate_stream emits done" "$stream_resp" 'assistant\.done'

echo
yellow "== XHS Smoke Summary =="
echo "PASS=$PASS"
echo "FAIL=$FAIL"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi

green "All XHS checks passed."