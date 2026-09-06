import pytest
from app.search.models import SearchResult
from app.search.engine import WebSearchEngine

def test_search_result_dataclass():
    res = SearchResult(
        url="https://github.com/navya185",
        platform="github.com",
        title="navya185 GitHub Profile",
        timestamp="2026-09-06",
        author="navya185",
        raw_content="Python Blockchain Face Search Pipeline"
    )
    assert res.platform == "github.com"
    assert res.url == "https://github.com/navya185"
    d = res.to_dict()
    assert d["title"] == "navya185 GitHub Profile"

def test_web_search_engine_domain_extraction():
    engine = WebSearchEngine()
    assert engine.extract_domain("https://www.twitter.com/user/123") == "twitter.com"
    assert engine.extract_domain("https://github.com/navya185") == "github.com"

def test_web_search_engine_query():
    engine = WebSearchEngine()
    results = engine.search(query="python programming", max_results=2)
    assert isinstance(results, list)
    assert len(results) > 0
    assert isinstance(results[0], SearchResult)
    assert len(results[0].url) > 0
