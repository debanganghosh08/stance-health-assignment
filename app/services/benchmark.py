import os
import time
import asyncio
import logging
import argparse
import json
from typing import List, Dict, Any
from app.services.triage import TriageService
from app.services.cache import cache_service

# Disable excessive logging for cleaner benchmark console output
logging.getLogger("app.services.triage").setLevel(logging.WARNING)
logging.getLogger("app.graph.triage_graph").setLevel(logging.WARNING)
logging.getLogger("app.services.rag").setLevel(logging.WARNING)
logging.getLogger("app.services.llm").setLevel(logging.WARNING)

async def run_benchmark(size: str):
    triage_service = TriageService()
    
    # 5 standard RAG/LLM heavy queries
    small_queries = [
        "I feel a strange buzzing in my elbow when I snap my fingers.",
        "I have a weird rash on my forearm that does not itch but looks like a red circle.",
        "My toe has been tingling since this morning after I stubbed it on a chair.",
        "Persistent lower back pain after working at my desk for 10 hours.",
        "My left foot feels a bit numb and cold when I sit for too long."
    ]
    
    if size == "small":
        sample_queries = small_queries
    else:
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fallback_path = os.path.join(base_dir, "resources", "cases_fallback.json")
            with open(fallback_path, "r", encoding="utf-8") as f:
                cases_data = json.load(f)
            all_cases = cases_data.get("cases", [])
            messages = [c["message"] for c in all_cases if "message" in c]
            
            if size == "mixed":
                sample_queries = messages[:20]
            else:  # batch
                sample_queries = messages[:100]
        except Exception as e:
            print(f"Error loading fallback cases: {e}. Defaulting to small queries.")
            sample_queries = small_queries

    print("=" * 60)
    print("SYMPTOM TRIAGE AGENT - CACHE BENCHMARK UTILITY")
    print("=" * 60)
    print(f"Benchmark Size Category: {size.upper()}")
    print(f"Loaded {len(sample_queries)} benchmark queries.")
    
    # --- RUN 1: Cold Cache ---
    print("\n[RUN 1] Clearing cache for COLD cache evaluation...")
    cache_service.clear()
    
    print("Executing cold queries...")
    cold_start = time.time()
    cold_results = []
    for query in sample_queries:
        res = await triage_service.triage_symptoms(query, patient_id="benchmark_patient")
        cold_results.append(res)
    cold_duration = time.time() - cold_start
    
    # Verify no cache hits occurred
    cold_hits = sum(1 for r in cold_results if r.cache_hit)
    print(f"Cold Run Completed. Hits: {cold_hits}/{len(sample_queries)}")
    
    # --- RUN 2: Warm Cache ---
    print("\n[RUN 2] Executing same queries for WARM cache evaluation...")
    warm_start = time.time()
    warm_results = []
    for query in sample_queries:
        res = await triage_service.triage_symptoms(query, patient_id="benchmark_patient")
        warm_results.append(res)
    warm_duration = time.time() - warm_start
    
    warm_hits = sum(1 for r in warm_results if r.cache_hit)
    print(f"Warm Run Completed. Hits: {warm_hits}/{len(sample_queries)}")
    
    # --- Comparison and Output ---
    improvement = ((cold_duration - warm_duration) / cold_duration) * 100 if cold_duration > 0 else 0
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 60)
    print("Cold Cache:")
    print(f"Total Time: {cold_duration:.2f}s")
    print(f"Avg Latency: {(cold_duration / len(sample_queries)) * 1000:.1f}ms")
    print(f"Cache Hits: {cold_hits}")
    
    print("\nWarm Cache:")
    print(f"Total Time: {warm_duration:.2f}s")
    print(f"Avg Latency: {(warm_duration / len(sample_queries)) * 1000:.1f}ms")
    print(f"Cache Hits: {warm_hits}")
    
    print("\nImprovement:")
    print(f"{improvement:.1f}%")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run caching performance benchmarks.")
    parser.add_argument(
        "--size", "-s",
        choices=["small", "mixed", "batch"],
        default="small",
        help="Sizing configuration for benchmark: small (5), mixed (20), batch (100) queries."
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.size))
