import pytest
from app.models.schemas import TriageResponse, UrgencyLevel
from app.services.cache import get_normalized_hash, cache_service, normalize_message, RedisCache
from app.services.triage import TriageService

@pytest.mark.anyio
async def test_normalization_and_hashing():
    # Semantically equivalent queries
    q1 = "Chest pain!!!"
    q2 = "chest pain"
    q3 = "CHEST PAIN"
    q4 = "  chest   pain  "
    
    assert normalize_message(q1) == "chestpain"
    assert normalize_message(q2) == "chestpain"
    assert normalize_message(q3) == "chestpain"
    assert normalize_message(q4) == "chestpain"
    
    h1 = get_normalized_hash(q1)
    h2 = get_normalized_hash(q2)
    h3 = get_normalized_hash(q3)
    h4 = get_normalized_hash(q4)
    
    assert h1 == h2 == h3 == h4

@pytest.mark.anyio
async def test_memory_cache_fallback():
    # Create a RedisCache instance with a bad URL to simulate Redis connection failure
    bad_redis = RedisCache("redis://invalid_host_1234:9999")
    assert not bad_redis.is_active
    
    # Verify our CacheService falls back to MemoryCache and handles gets/sets successfully
    cache_service.clear()
    
    test_key = "test:fallback_key"
    test_val = {"status": "ok"}
    
    # This should store in MemoryCache since RedisCache is inactive or fails
    cache_service.set(test_key, test_val)
    
    retrieved = cache_service.get(test_key)
    assert retrieved == test_val

@pytest.mark.anyio
async def test_conditional_caching():
    triage_service = TriageService()
    
    # 1. Cheap Query (Matches local rules, bypasses LLM and RAG)
    cheap_query = "I have a cough"
    cache_service.clear()
    
    h_cheap = get_normalized_hash(cheap_query)
    cache_key_cheap = f"triage:{h_cheap}"
    
    res_cheap = await triage_service.triage_symptoms(cheap_query, patient_id="pat_cheap")
    
    # Since it is a cheap local rule match, it must not be cached!
    assert res_cheap.local_triage_used is True
    assert res_cheap.rag_used is False
    assert cache_service.get(cache_key_cheap) is None
    
    # 2. Expensive Query (Requires LLM and/or RAG)
    expensive_query = "I feel a strange buzzing in my elbow when I snap my fingers."
    h_exp = get_normalized_hash(expensive_query)
    cache_key_exp = f"triage:{h_exp}"
    
    res_exp1 = await triage_service.triage_symptoms(expensive_query, patient_id="pat_exp")
    assert res_exp1.cache_hit is False
    
    # Since it is expensive (requires LLM/RAG), it must be cached
    cached_val = cache_service.get(cache_key_exp)
    assert cached_val is not None
    assert cached_val["urgency"] in [UrgencyLevel.SELF_CARE.value, UrgencyLevel.NON_URGENT.value]
    
    # Run again to verify cache hit
    res_exp2 = await triage_service.triage_symptoms(expensive_query, patient_id="pat_exp")
    assert res_exp2.cache_hit is True
    assert res_exp2.cache_layer == "triage_response"


@pytest.mark.anyio
async def test_multi_level_caching():
    triage_service = TriageService()
    cache_service.clear()
    
    # Needs a query that triggers both RAG and LLM without triggering any local rule bypass
    expensive_query = "I feel a strange buzzing in my elbow when I snap my fingers."
    h = get_normalized_hash(expensive_query)
    
    # Run 1: Cold Cache
    res1 = await triage_service.triage_symptoms(expensive_query, patient_id="pat_ml")
    assert res1.cache_hit is False
    
    # Verify that RAG cache and LLM cache now contain the entries
    assert cache_service.get(f"rag:{h}") is not None
    assert cache_service.get(f"llm:{h}") is not None
    
    # Delete triage response cache to force pipeline execution
    cache_service.delete(f"triage:{h}")
    
    # Run 2: Triage response cache miss, but RAG/LLM caches should hit
    res2 = await triage_service.triage_symptoms(expensive_query, patient_id="pat_ml")
    # Triage response cache was deleted, but it had inner cache hits (RAG or LLM)
    assert res2.cache_hit is True
    assert res2.cache_layer in ("rag", "llm")
