# @Author: RebeccaZhou
# @Description: Cache package
#              缓存模块包

"""Exact cache module: normalized-question-hash → answer JSON for single-turn RAG."""
from app.cache.exact_cache import ExactCache, normalize, cache_key

__all__ = ["ExactCache", "normalize", "cache_key"]
