import os
import sys
import json
import streamlit as st
import numpy as np
from PIL import Image

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.face.embedder import FaceEmbedder
from app.face.exceptions import FaceProcessingError
from app.search.engine import WebSearchEngine
from app.search.models import SearchResult
from app.verification.hasher import ContentHasher
from app.verification.verifier import PipelineVerifier
from app.blockchain.client import BlockchainClient

# Streamlit Page Config
st.set_page_config(
    page_title="Face Scan & Blockchain Content Verifier",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Face Scan → Web Search → Blockchain Verification Pipeline")
st.caption("A real-time end-to-end pipeline verifying web content authenticity using facial recognition and EVM smart contracts.")

# Initialize session state variables
if "embedder" not in st.session_state:
    st.session_state.embedder = FaceEmbedder()
if "search_engine" not in st.session_state:
    st.session_state.search_engine = WebSearchEngine()
if "blockchain" not in st.session_state:
    st.session_state.blockchain = BlockchainClient()
if "verifier" not in st.session_state:
    st.session_state.verifier = PipelineVerifier(embedder=st.session_state.embedder)

if "target_embedding" not in st.session_state:
    st.session_state.target_embedding = None
if "face_box" not in st.session_state:
    st.session_state.face_box = None
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_result" not in st.session_state:
    st.session_state.selected_result = None
if "content_hash" not in st.session_state:
    st.session_state.content_hash = None
if "tx_hash" not in st.session_state:
    st.session_state.tx_hash = None
if "onchain_record" not in st.session_state:
    st.session_state.onchain_record = None

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
search_query = st.sidebar.text_input("Search Query / Name", value="python developer profile")
target_url_input = st.sidebar.text_input("Direct Target Web URL (Optional)", value="")
similarity_threshold = st.sidebar.slider("Face Match Similarity Threshold", 0.50, 0.95, 0.70, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Blockchain Node Info")
st.sidebar.info(f"Connected RPC: {st.session_state.blockchain.rpc_url or 'Built-in EVM TestNet'}")
st.sidebar.text(f"Contract: {st.session_state.blockchain.contract_address[:10]}...")
st.sidebar.text(f"Account: {st.session_state.blockchain.account[:10]}...")

# ---------------------------------------------------------
# STEP 1: FACE IDENTIFICATION
# ---------------------------------------------------------
st.header("1. Upload Face Image & Encode")
uploaded_file = st.file_uploader("Upload Target Face Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Uploaded Input Face Image", use_container_width=True)

    if st.button("🔍 2. Identify Face"):
        with st.spinner("Detecting face and computing 128-d encoding..."):
            try:
                emb, box = st.session_state.embedder.encode_face(image, require_single_face=True)
                st.session_state.target_embedding = emb
                st.session_state.face_box = box
                st.success(f"✅ Single Face Detected successfully! Bounding Box: {box}")
            except FaceProcessingError as e:
                st.error(f"❌ Face Identification Error: {e}")

if st.session_state.target_embedding is not None:
    with col2:
        st.subheader("Face Embedding Features (Sample)")
        st.code(f"Vector Shape: {st.session_state.target_embedding.shape}\nSample: {st.session_state.target_embedding[:10].round(4)}")
        st.warning("⚠️ Disclaimer: Face recognition is probabilistic and not 100% accurate.")

# ---------------------------------------------------------
# STEP 2: WEB / SOCIAL MEDIA SEARCH
# ---------------------------------------------------------
st.markdown("---")
st.header("3. Genuine Web & Social Media Search")

if st.button("🌐 Execute Web Search"):
    if st.session_state.target_embedding is None:
        st.warning("Please identify a face in Step 1 first!")
    else:
        with st.spinner(f"Executing real web search for '{search_query}'..."):
            results = st.session_state.search_engine.search(
                query=search_query,
                target_url=target_url_input if target_url_input.strip() else None,
                max_results=5
            )
            st.session_state.search_results = results
            if results:
                st.session_state.selected_result = results[0]
                st.success(f"✅ Discovered {len(results)} matching search results!")
            else:
                st.error("No web search results discovered.")

if st.session_state.search_results:
    st.subheader("Discovered Search Results")
    options = [f"[{res.platform}] {res.title} ({res.url})" for res in st.session_state.search_results]
    selected_idx = st.selectbox("Select Discovered Result for Blockchain Verification:", range(len(options)), format_func=lambda idx: options[idx])
    st.session_state.selected_result = st.session_state.search_results[selected_idx]

    res = st.session_state.selected_result
    st.json({
        "URL": res.url,
        "Platform/Domain": res.platform,
        "Title": res.title,
        "Author/Account": res.author,
        "Timestamp": res.timestamp,
        "Relevance Score": res.relevance_score,
        "Raw Content Preview": res.raw_content[:200]
    })

# ---------------------------------------------------------
# STEP 3: FACE MATCH VERIFICATION & HASHING
# ---------------------------------------------------------
if st.session_state.selected_result and st.session_state.target_embedding is not None:
    st.markdown("---")
    st.header("4. Content Match Verification & Cryptographic Fingerprint")

    if st.button("🔬 Verify Match & Generate SHA-256 Fingerprint"):
        res = st.session_state.selected_result
        match_info = st.session_state.verifier.verify_face_match(
            target_embedding=st.session_state.target_embedding,
            search_result=res,
            threshold=similarity_threshold
        )
        c_hash = ContentHasher.hash_search_result(res)
        st.session_state.content_hash = c_hash

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader("Face Similarity Check")
            st.metric("Face Verified", str(match_info.get("face_verified")))
            st.metric("Similarity Score", f"{match_info.get('similarity_score', 0.0)}")
            st.write(match_info.get("message", "Comparison complete."))

        with col_m2:
            st.subheader("SHA-256 Content Fingerprint")
            st.code(c_hash)

# ---------------------------------------------------------
# STEP 4: BLOCKCHAIN TRANSACTION & ON-CHAIN VERIFICATION
# ---------------------------------------------------------
if st.session_state.content_hash:
    st.markdown("---")
    st.header("5. Blockchain On-Chain Record & Verification")

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        if st.button("⛓️ Submit Fingerprint to Blockchain"):
            with st.spinner("Submitting transaction to EVM smart contract..."):
                res = st.session_state.selected_result
                tx_hash = st.session_state.blockchain.submit_content_hash(
                    content_hash=st.session_state.content_hash,
                    source_url=res.url,
                    metadata={
                        "title": res.title,
                        "platform": res.platform,
                        "author": res.author
                    }
                )
                st.session_state.tx_hash = tx_hash
                st.success(f"✅ Recorded on-chain! Tx Hash: {tx_hash}")

    if st.session_state.tx_hash:
        with col_b2:
            st.subheader("Verify On-Chain Record")
            onchain_rec = st.session_state.blockchain.get_onchain_record(st.session_state.content_hash)
            st.session_state.onchain_record = onchain_rec

            st.write(f"**Stored Hash:** `{onchain_rec['content_hash']}`")
            st.write(f"**Block Timestamp:** `{onchain_rec['timestamp']}`")
            st.write(f"**Submitter:** `{onchain_rec['submitter']}`")

            ver_check = PipelineVerifier.verify_onchain_hash(
                st.session_state.content_hash,
                onchain_rec['content_hash']
            )
            if ver_check["verified"]:
                st.success("🟢 STATUS: VERIFIED (On-Chain Hash Matches Discovered Content)")
            else:
                st.error("🔴 STATUS: NOT VERIFIED / TAMPERED")

# ---------------------------------------------------------
# STEP 5: TAMPER VERIFICATION DEMO
# ---------------------------------------------------------
if st.session_state.onchain_record and st.session_state.selected_result:
    st.markdown("---")
    st.header("6. ⚠️ Tamper Verification Demonstration")
    st.write("Modify the discovered payload by adding/altering text to test if the cryptographic hash changes and fails on-chain verification.")

    tamper_input = st.text_input("Payload Modification Tag", value="[ATTACKER ALTERED CONTENT]")

    if st.button("🚨 Run Tamper Verification Test"):
        orig_res = st.session_state.selected_result
        onchain_h = st.session_state.onchain_record["content_hash"]

        tamper_report = PipelineVerifier.demonstrate_tampering(orig_res, onchain_h, tamper_input)

        st.subheader("Tamper Test Results")
        st.write(f"**Original On-Chain Recorded Hash:**")
        st.code(tamper_report["original_onchain_hash"])

        st.write(f"**Tampered Content Newly Calculated Hash:**")
        st.code(tamper_report["tampered_hash"])

        if not tamper_report["is_verified"]:
            st.error(f"🔴 VERIFICATION RESULT: {tamper_report['verification_status']}")
            st.warning("⚠️ TAMPER DETECTED! SHA-256 fingerprint mismatch between altered payload and blockchain record.")
        else:
            st.success("Verified")
