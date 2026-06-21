import threading
from collections import deque
from typing import Dict, Any

class MetricsRegistry:
    """
    Thread-safe in-memory metrics registry supporting counter increments,
    rolling-window latency observations, and JSON / Prometheus formatting.
    """
    def __init__(self):
        self._lock = threading.Lock()
        
        # Counters
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.local_bypass_count = 0
        self.llm_calls = 0
        self.rag_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.validator_overrides = 0
        self.emergency_cases = 0

        # Rolling window latencies (max 1000 observations to prevent unbounded memory growth)
        self.overall_latencies = deque(maxlen=1000)
        self.stage_latencies = {
            "feature_extraction": deque(maxlen=1000),
            "rule_engine": deque(maxlen=1000),
            "analyze_node": deque(maxlen=1000),
            "rag_retrieval": deque(maxlen=1000),
            "triage_node": deque(maxlen=1000),
            "validator_node": deque(maxlen=1000)
        }

    def increment(self, metric_name: str, value: int = 1):
        """
        Increments a given counter metric thread-safely.
        """
        with self._lock:
            if hasattr(self, metric_name):
                setattr(self, metric_name, getattr(self, metric_name) + value)

    def observe_latency(self, stage_name: str, latency_ms: float):
        """
        Observes latency for a specific stage or overall pipeline execution thread-safely.
        """
        with self._lock:
            if stage_name == "overall":
                self.overall_latencies.append(latency_ms)
            elif stage_name in self.stage_latencies:
                self.stage_latencies[stage_name].append(latency_ms)

    def _get_avg(self, dq: deque) -> float:
        if not dq:
            return 0.0
        return round(sum(dq) / len(dq), 2)

    def _get_p95(self, dq: deque) -> float:
        if not dq:
            return 0.0
        sorted_list = sorted(list(dq))
        idx = int(len(sorted_list) * 0.95)
        idx = min(idx, len(sorted_list) - 1)
        return round(sorted_list[idx], 2)

    def export_json(self) -> Dict[str, Any]:
        """
        Exports the metrics registry state as a JSON-serializable dictionary.
        """
        with self._lock:
            total = self.total_requests
            hits = self.cache_hits
            hit_rate = round((hits / total) * 100, 2) if total > 0 else 0.0
            
            output = {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "local_bypass_count": self.local_bypass_count,
                "llm_calls": self.llm_calls,
                "rag_calls": self.rag_calls,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": hit_rate,
                "validator_overrides": self.validator_overrides,
                "emergency_cases": self.emergency_cases,
                "avg_latency_ms": self._get_avg(self.overall_latencies),
                "p95_latency_ms": self._get_p95(self.overall_latencies)
            }
            
            # Export average latency for individual stages
            for stage, dq in self.stage_latencies.items():
                output[f"avg_{stage}_latency_ms"] = self._get_avg(dq)
                
            return output

    def export_prometheus(self) -> str:
        """
        Exports metrics in Prometheus exposition plain text format.
        """
        metrics = self.export_json()
        lines = []
        for k, v in metrics.items():
            lines.append(f"# HELP triage_{k} Current value for metrics {k}")
            lines.append(f"# TYPE triage_{k} gauge")
            lines.append(f"triage_{k} {v}")
        return "\n".join(lines) + "\n"

# Global Singleton metrics registry instance
metrics_registry = MetricsRegistry()
