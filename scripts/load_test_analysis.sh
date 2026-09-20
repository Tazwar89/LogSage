#!/usr/bin/env bash
# Verifies logsage-analysis holds under repeated real (non-mock) analysis calls.
# Usage: [BASE_URL=http://localhost:8002] ./load_test_analysis.sh <trace_id_prefix> <count>
# Requires: trace_ids already present in Redis (upload real logs first via ingestion).
# Exits non-zero if any call returns a non-200 status.

BASE_URL="${BASE_URL:-https://logsage-analysis.fly.dev}"
PREFIX="${1:-sample.log}"
COUNT="${2:-30}"
FAILS=0

echo "Firing $COUNT sequential /analyze calls against $BASE_URL"
for i in $(seq 0 $((COUNT - 1))); do
  TRACE_ID="${PREFIX}-${i}"
  STATUS=$(curl -s -o "/tmp/resp_$i.json" -w "%{http_code}" "$BASE_URL/analyze/$TRACE_ID")
  echo "[$i] trace_id=$TRACE_ID status=$STATUS"
  [ "$STATUS" = "200" ] || FAILS=$((FAILS + 1))
  sleep 0.5
done

echo "Failures: $FAILS / $COUNT"
echo "For Fly: fly status -a logsage-analysis ; fly logs -a logsage-analysis | grep -i -E 'oom|kill|restart'"
[ "$FAILS" -eq 0 ]