# Face Scan → Web/Social Search → Blockchain Verification Pipeline

A production-ready, end-to-end Python application that accepts a face image, detects and generates facial feature embeddings, executes genuine web and social media searches, matches face embeddings against discovered content, hashes the matched content with SHA-256, records the cryptographic fingerprint on an EVM smart contract, verifies the on-chain record, and demonstrates tamper detection.

---

## 🛠️ What It Does

This system bridges computer vision, real-time web intelligence, and Web3 blockchain immutability into a 12-step verification pipeline:

1. **Face Scan / Input**: Accepts user-provided face images.
2. **Face Detection & Encoding**: Detects human faces and computes 128-dimensional facial feature vector embeddings. Validates that exactly one face is present (`NoFaceFoundError`, `MultipleFacesFoundError`).
3. **Genuine Web/Social Media Search**: Queries real-time search backends (SerpAPI Google Lens, Google CSE, Bing API, or live DDGS open web search) to discover real web & social media content.
4. **Discovered Content Retrieval**: Captures full metadata (URL, domain, platform, title, timestamp, author, raw payload, image URL).
5. **Face Match Verification**: Downloads candidate content images, extracts faces, and calculates Euclidean distance & Cosine similarity against the target face embedding.
6. **Cryptographic Fingerprinting**: Computes a deterministic SHA-256 hash of the discovered content payload.
7. **Blockchain Transaction**: Submits the SHA-256 fingerprint and metadata to a Solidity smart contract (`ContentVerifier.sol`) deployed on an EVM blockchain (built-in Python EVM testnet or Anvil/Ganache local node).
8. **On-Chain Verification**: Retrieves stored fingerprint details from the blockchain, re-hashes local content, compares hashes, and reports `VERIFIED`.
9. **Tamper Demonstration**: Modifies 1 byte/character of the discovered content payload, re-computes the hash, compares against the on-chain record, and reports `NOT VERIFIED / TAMPERED`.

---

## 🏗️ Architecture

```
┌──────────────────┐
│ Target Face Scan │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Face Detection & 128-d Feature Encoding      │
│ (OpenCV Haar Cascades / Contour Feature Map) │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Genuine Web & Social Media Search Engine     │
│ (SerpAPI Google Lens / Google CSE / DDGS)    │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Content Retrieval & Face Similarity Check    │
│ (Cosine Similarity & Threshold Verification) │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ SHA-256 Cryptographic Content Fingerprint    │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Web3 EVM Smart Contract (ContentVerifier.sol)│
│ (Built-in EVM TestNet or Anvil/Ganache Node) │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ On-Chain Hash Retrieval & Match Verification │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Tamper Verification Demonstration            │
│ (1-byte Payload Mod -> TAMPERED DETECTED)    │
└──────────────────────────────────────────────┘
```

---

## 💻 Technologies Used

- **Language**: Python 3.10+
- **Computer Vision**: OpenCV (`opencv-python`), NumPy, SciPy, Pillow
- **Web Search**: `ddgs`, `requests`, `beautifulsoup4`, SerpAPI / Google CSE integration
- **Blockchain & Smart Contracts**: Solidity (`^0.8.0`), Web3.py, `py-evm`, `eth-tester`, `py-solc-x`
- **User Interface**: Streamlit
- **Testing**: Pytest

---

## 🚀 Prerequisites

- Python 3.10 or higher
- Git

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/navya185/face-web-blockchain.git
   cd face-web-blockchain
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows Powershell:
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configurable options in `.env`:
```ini
# Optional Web Search API Keys (Falls back to live DDGS open web search if empty)
SERPAPI_API_KEY=your_serpapi_key_here
GOOGLE_SEARCH_API_KEY=your_google_key_here
GOOGLE_CSE_ID=your_cse_id_here
BING_SEARCH_API_KEY=your_bing_key_here

# Blockchain Configuration (Defaults to zero-setup built-in EVM testnode)
RPC_URL=http://127.0.0.1:8545
PRIVATE_KEY=
CONTRACT_ADDRESS=
```

---

## ⛓️ Starting Local Blockchain & Deploying Smart Contract

### Option A: Out-of-the-Box Built-in EVM (Zero Setup Required)
If `RPC_URL` is omitted or local node is offline, the application automatically initializes a Python-native EVM node (`EthereumTesterProvider`) in memory.

### Option B: Local Anvil / Ganache Node
To run an external EVM node:

1. Start **Anvil** (Foundry):
   ```bash
   anvil
   ```
   Or **Ganache**:
   ```bash
   ganache-cli
   ```

2. Deploy the smart contract using the deployment script:
   ```bash
   python scripts/deploy_contract.py
   ```

   *Output:*
   ```text
   === Deploying ContentVerifier Smart Contract ===
   Connected Web3 Node: HTTPProvider(http://127.0.0.1:8545)
   Deployer Account: 0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf
   Deployed Contract Address: 0xF2E246BB76DF876Cef8b38ae84130F4F55De395b
   Deployment Tx Hash: fe9eed81bf7a867bdfb80b54be09b63...
   === Deployment Complete ===
   ```

---

## 🏃 Running the Application

### 1. Generating Sample Test Data
Create sample synthetic face images for local testing:
```bash
python scripts/generate_sample_data.py
```

### 2. Running End-to-End CLI Pipeline
```bash
python scripts/run_pipeline.py --image data/sample_faces/target_person.jpg --query "python developer profile"
```

### 3. Launching Streamlit Interactive UI
```bash
python -m streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Run full test suite using Pytest:
```bash
python -m pytest tests/
```

---

## 🔍 How Search, Face Matching & Blockchain Work

### 1. Real Web & Social Media Search Engine
The system performs **actual live searches** without hardcoded posts:
- **SerpAPI Engine**: Executes Google Lens reverse-image search or Google organic search when `SERPAPI_API_KEY` is provided.
- **Google CSE Engine**: Queries Google Custom Search API when `GOOGLE_SEARCH_API_KEY` and `GOOGLE_CSE_ID` are set.
- **DDGS Open Search Engine**: Performs live web queries when no API keys are present.
- **Direct Web Scraper**: Fetches web pages directly via `requests` and `BeautifulSoup` to extract Open Graph metadata, title, author, date, body content, and image links.

### 2. Face Detection & Similarity Matching
- **Detector**: Locates face bounding boxes `(x, y, w, h)`. Enforces error checks:
  - `NoFaceFoundError`: 0 faces detected.
  - `MultipleFacesFoundError`: > 1 faces detected when single face is required.
- **Embedder**: Generates a normalized 128-dimensional HOG and spatial feature vector.
- **Matcher**: Calculates Cosine Similarity & Euclidean Distance between target face embedding and candidate content image faces against a configurable threshold (e.g. `0.70`).

### 3. Smart Contract & On-Chain Verification
The smart contract `ContentVerifier.sol` stores:
```solidity
struct ContentRecord {
    string contentHash;    // SHA-256 hex digest
    string sourceUrl;      // Source web URL
    uint256 timestamp;     // Block timestamp
    string metadataJson;   // JSON metadata (title, author, platform)
    address submitter;     // Submitter wallet address
}
```
1. Content payload is fingerprinted using **SHA-256**.
2. Fingerprint is written on-chain (`registerContent`).
3. Verification retrieves the recorded hash from the blockchain (`getContentRecord`).
4. Re-hashes local payload and verifies exact match (`VERIFIED`).

### 4. Tamper Detection Demonstration
If an attacker alters even 1 byte/character of the discovered content payload:
```text
Original Fingerprint : 33b02b6e914137aca6c7a1e202e868e0aceca8ab3a4eba5d2b24280d46e0948c
Tampered Fingerprint : 76748175140858bfa4cfe156c1d7fa197cd2b3cd1ca8e2e58bae19515de72411
Status               : NOT VERIFIED / TAMPERED
```

---

## 📋 Example CLI Output

```text
==========================================================================
  FACE SCAN -> WEB SEARCH -> BLOCKCHAIN FINGERPRINT -> VERIFICATION PIPELINE  
==========================================================================

[STEP 1] Loading Target Face Image: data/sample_faces/target_person.jpg
  [+] Face Detected at Bounding Box: (60, 45, 180, 210)
  [+] Face Embedding Generated (128-d vector): [0.1218 0.0157 0. 0. 0.]

[STEP 2] Executing Real Web & Social Media Search (Query: 'python developer profile')
  [+] Discovered 3 search results:
      1. [enhancv.com] Python Developer Resume Examples -> https://enhancv.com/...
      2. [resumeworded.com] Python Developer Resume Examples -> https://resumeworded.com/...

[STEP 3] Selected Result for Fingerprinting:
  URL: https://enhancv.com/...
  Platform: enhancv.com
  Title: Python Developer Resume Examples

[STEP 4] Performing Face Similarity Verification on Discovered Content
  [+] Face Verified in Content Image: False
  [+] Similarity Score: 0.0

[STEP 5] Generating SHA-256 Content Fingerprint
  [+] SHA-256 Fingerprint: 33b02b6e914137aca6c7a1e202e868e0aceca8ab3a4eba5d2b24280d46e0948c

[STEP 6] Submitting Content Fingerprint to EVM Blockchain
  [+] Blockchain Transaction Hash: 4bd4e210bb30e29b59bfbb0814fbbed2557e1e6...

[STEP 7] Verifying On-Chain Fingerprint Record
  [+] On-Chain Recorded Hash: 33b02b6e914137aca6c7a1e202e868e0aceca8ab3a4eba5d2b24280d46e0948c
  [+] Submitter Wallet: 0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf
  [+] Verification Result: VERIFIED (Verified: True)

[STEP 8] Executing Tamper Detection Demonstration
  [!] Modified Payload SHA-256 Fingerprint: 76748175140858bfa4cfe156c1d7fa197...
  [!] Verification Result on Modified Content: NOT VERIFIED / TAMPERED
  [!] Tamper Diff Detected: True
==========================================================================
```

---

## 📂 Project Structure

```
face-web-blockchain/
├── app/
│   ├── __init__.py
│   ├── face/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   ├── embedder.py
│   │   └── exceptions.py
│   ├── search/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── models.py
│   ├── blockchain/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── deployer.py
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── hasher.py
│   │   └── verifier.py
│   └── main.py
├── contracts/
│   ├── ContentVerifier.sol
│   └── ContentVerifier.json
├── tests/
│   ├── __init__.py
│   ├── test_face.py
│   ├── test_search.py
│   ├── test_blockchain.py
│   └── test_verification.py
├── scripts/
│   ├── deploy_contract.py
│   ├── run_pipeline.py
│   └── generate_sample_data.py
├── data/
│   ├── sample_faces/
│   │   └── target_person.jpg
│   └── .gitkeep
├── .env.example
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## ⚠️ Important Limitations & Security Notes

- **API Limits & Scraping Restrictions**: Web search engines and social platforms enforce rate limits or prohibit automated scraping. Use official API keys (`SERPAPI_API_KEY`, etc.) in `.env` for production workloads.
- **Probabilistic Face Matching**: Face recognition algorithms return similarity confidence scores and can produce false positives or false negatives under variable lighting, angles, or resolution.
- **Blockchain Scope**: A blockchain proves that a specific SHA-256 fingerprint was recorded at a specific block timestamp by a submitter address. It **does NOT guarantee** that the original third-party web content itself was truthful or factual.
- **Privacy Protection**: Never store raw face images, personal identification numbers, or private key data on-chain. Only cryptographic SHA-256 fingerprints are stored on the public ledger.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
