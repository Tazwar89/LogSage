"""
Redis-backed persistence for parsed log entries.

Replaces the previous in-memory `parsed_logs_db` dict, which lost all data
on container restart. Each parsed log entry is stored as a JSON string under
key `log:{trace_id}`, with a separate set `log:index` tracking all trace_ids
so /logs can list them without a Redis KEYS scan (which is discouraged in
production Redis usage).
"""
import json
import os
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

LOG_INDEX_KEY = "log:index"


def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


class LogStore:
    """Thin wrapper around Redis for storing/retrieving parsed log entries."""
    def __init__(self, client=None):
        self.client = client or get_redis_client()


    def save(self, trace_id: str, entry: dict):
        self.client.set(f"log:{trace_id}", json.dumps(entry))
        self.client.sadd(LOG_INDEX_KEY, trace_id)


    def get(self, trace_id: str):
        raw = self.client.get(f"log:{trace_id}")

        if raw is None:
            return None

        else:
            return json.loads(raw)


    def list_trace_ids(self):
        return sorted(self.client.smembers(LOG_INDEX_KEY))


    def clear(self):
        """Useful for tests / resetting state between baseline uploads."""
        trace_ids = self.client.smembers(LOG_INDEX_KEY)

        for tid in trace_ids:
            self.client.delete(f"log:{tid}")

        self.client.delete(LOG_INDEX_KEY)