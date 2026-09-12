#!/usr/bin/env bash
# Verifies logsage-analysis holds under repeated real (non-mock) analysis calls.
# Usage: ./load_test_analysis.sh <trace_id_prefix> <count>
# Requires: trace_ids already present in Redis (upload real logs first via ingestion).

BASE_URL="https://logsage-analysis.fly.dev"
PREFIX="${1:-sample.log}"
COUNT="${2:-30}"

echo "Firing $COUNT sequential /analyze calls against $BASE_URL"
for i in $(seq 0 $((COUNT - 1))); do
  TRACE_ID="${PREFIX}-${i}"
  STATUS=$(curl -s -o /tmp/resp_$i.json -w "%{http_code}" "$BASE_URL/analyze/$TRACE_ID")
  echo "[$i] trace_id=$TRACE_ID status=$STATUS"
  sleep 0.5
done

echo "Done. Now check memory/restarts:"
echo "  fly status -a logsage-analysis"
echo "  fly logs -a logsage-analysis | grep -i -E 'oom|kill|restart'"
