import os
import sys
import argparse
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.face.embedder import FaceEmbedder
from app.face.exceptions import FaceProcessingError
from app.search.engine import WebSearchEngine
from app.verification.hasher import ContentHasher
from app.verification.verifier import PipelineVerifier
from app.blockchain.client import BlockchainClient

def run_pipeline(image_path: str, search_query: str = "python developer profile"):
    print("==========================================================================")
    print("  FACE SCAN -> WEB SEARCH -> BLOCKCHAIN FINGERPRINT -> VERIFICATION PIPELINE  ")
    print("==========================================================================")

    # 1. Face Identification
    print(f"\n[STEP 1] Loading Target Face Image: {image_path}")
    embedder = FaceEmbedder()
    try:
        target_embedding, face_box = embedder.encode_face(image_path, require_single_face=True)
        print(f"  [+] Face Detected at Bounding Box: {face_box}")
        print(f"  [+] Face Embedding Generated (128-d vector, sample[:5]): {target_embedding[:5].round(4)}")
    except FaceProcessingError as e:
        print(f"  [-] Face Identification Error: {e}")
        return

    # 2. Web & Social Search
    print(f"\n[STEP 2] Executing Real Web & Social Media Search (Query: '{search_query}')")
    search_engine = WebSearchEngine()
    results = search_engine.search(query=search_query, max_results=3)

    if not results:
        print("  [-] No search results returned.")
        return

    print(f"  [+] Discovered {len(results)} search results:")
    for idx, res in enumerate(results):
        print(f"      {idx+1}. [{res.platform}] {res.title} -> {res.url}")

    selected_result = results[0]
    print(f"\n[STEP 3] Selected Result for Fingerprinting:")
    print(f"  URL: {selected_result.url}")
    print(f"  Platform: {selected_result.platform}")
    print(f"  Title: {selected_result.title}")
    print(f"  Author: {selected_result.author}")

    # 3. Match Verification
    print(f"\n[STEP 4] Performing Face Similarity Verification on Discovered Content")
    verifier = PipelineVerifier(embedder=embedder)
    match_report = verifier.verify_face_match(target_embedding, selected_result)

    print(f"  [+] Face Verified in Content Image: {match_report.get('face_verified')}")
    print(f"  [+] Similarity Score: {match_report.get('similarity_score')} (Threshold: {match_report.get('threshold')})")
    print(f"  [+] Is Match: {match_report.get('is_match')}")

    # 4. Cryptographic Hashing
    print(f"\n[STEP 5] Generating SHA-256 Content Fingerprint")
    content_hash = ContentHasher.hash_search_result(selected_result)
    print(f"  [+] Discovered Content SHA-256 Fingerprint:\n      {content_hash}")

    # 5. Blockchain Transaction
    print(f"\n[STEP 6] Submitting Content Fingerprint to EVM Blockchain")
    blockchain = BlockchainClient()
    tx_hash = blockchain.submit_content_hash(
        content_hash=content_hash,
        source_url=selected_result.url,
        metadata={
            "title": selected_result.title,
            "platform": selected_result.platform,
            "author": selected_result.author,
            "similarity_score": match_report.get("similarity_score", 0.0)
        }
    )
    print(f"  [+] Blockchain Transaction Hash: {tx_hash}")

    # 6. Verification & On-Chain Read
    print(f"\n[STEP 7] Verifying On-Chain Fingerprint Record")
    onchain_record = blockchain.get_onchain_record(content_hash)
    print(f"  [+] On-Chain Recorded Hash: {onchain_record['content_hash']}")
    print(f"  [+] On-Chain Timestamp: {onchain_record['timestamp']}")
    print(f"  [+] Submitter Wallet: {onchain_record['submitter']}")

    verification_status = verifier.verify_onchain_hash(content_hash, onchain_record["content_hash"])
    print(f"  [+] Verification Result: {verification_status['status']} (Verified: {verification_status['verified']})")

    # 7. Tamper Verification Demo
    print(f"\n[STEP 8] Executing Tamper Detection Demonstration")
    tamper_demo = verifier.demonstrate_tampering(selected_result, onchain_record["content_hash"])
    print(f"  [!] Modified Payload SHA-256 Fingerprint:\n      {tamper_demo['tampered_hash']}")
    print(f"  [!] Verification Result on Modified Content: {tamper_demo['verification_status']}")
    print(f"  [!] Tamper Diff Detected: {tamper_demo['diff_detected']}")
    print("==========================================================================")
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY  ")
    print("==========================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Face Search Blockchain Pipeline")
    default_img = os.path.join(os.path.dirname(__file__), "..", "data", "sample_faces", "target_person.jpg")
    parser.add_argument("--image", type=str, default=default_img, help="Path to input face image")
    parser.add_argument("--query", type=str, default="python developer profile", help="Search query string")
    args = parser.parse_args()

    run_pipeline(args.image, args.query)
