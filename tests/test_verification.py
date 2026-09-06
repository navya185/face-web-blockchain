import pytest
from app.search.models import SearchResult
from app.verification.hasher import ContentHasher
from app.verification.verifier import PipelineVerifier

def test_content_hasher():
    res = SearchResult(
        url="https://example.com/post",
        platform="example.com",
        title="Sample Title",
        timestamp="2026-09-06",
        author="Bob",
        raw_content="Original web post content payload."
    )
    h1 = ContentHasher.hash_search_result(res)
    h2 = ContentHasher.hash_search_result(res)

    assert len(h1) == 64
    assert h1 == h2  # Deterministic

def test_pipeline_verifier_match():
    h1 = "c" * 64
    ver = PipelineVerifier.verify_onchain_hash(h1, h1)
    assert ver["verified"] is True
    assert ver["status"] == "VERIFIED"

def test_pipeline_verifier_tampered():
    h1 = "c" * 64
    h2 = "d" * 64
    ver = PipelineVerifier.verify_onchain_hash(h1, h2)
    assert ver["verified"] is False
    assert ver["status"] == "NOT VERIFIED / TAMPERED"

def test_demonstrate_tampering():
    res = SearchResult(
        url="https://example.com/post",
        platform="example.com",
        title="Authentic Content",
        timestamp="2026-09-06",
        author="Alice",
        raw_content="This is genuine un-tampered web content."
    )
    onchain_hash = ContentHasher.hash_search_result(res)

    tamper_report = PipelineVerifier.demonstrate_tampering(res, onchain_hash)
    assert tamper_report["tampered"] is True
    assert tamper_report["is_verified"] is False
    assert tamper_report["verification_status"] == "NOT VERIFIED / TAMPERED"
    assert tamper_report["diff_detected"] is True
