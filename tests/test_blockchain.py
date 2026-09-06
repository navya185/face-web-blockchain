import pytest
from app.blockchain.client import BlockchainClient

def test_blockchain_client_deployment_and_record():
    client = BlockchainClient()
    assert client.w3.is_connected()
    assert client.contract_address is not None
    assert client.contract_address.startswith("0x")

    test_hash = "a" * 64
    test_url = "https://example.com/post/1"
    meta = {"title": "Test Post", "author": "Alice"}

    # Submit content fingerprint on-chain
    tx_hash = client.submit_content_hash(test_hash, test_url, meta)
    assert len(tx_hash) > 0

    # Retrieve stored on-chain record
    record = client.get_onchain_record(test_hash)
    assert record["exists"] is True
    assert record["content_hash"] == test_hash
    assert record["source_url"] == test_url
    assert record["metadata"]["title"] == "Test Post"

    # Check verify_hash_onchain
    assert client.verify_hash_onchain(test_hash) is True
    assert client.verify_hash_onchain("b" * 64) is False
