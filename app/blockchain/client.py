import os
import json
from typing import Dict, Any, Optional, Tuple
from web3 import Web3
from web3.contract import Contract
from dotenv import load_dotenv

from app.blockchain.deployer import ContractDeployer

load_dotenv()

class BlockchainClient:
    """
    Web3.py Client wrapper for EVM blockchain interaction (Anvil / Ganache / EthTester / Public Testnet).
    """
    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        contract_address: Optional[str] = None
    ):
        self.rpc_url = rpc_url or os.getenv("RPC_URL")
        self.private_key = private_key or os.getenv("PRIVATE_KEY")
        self.contract_address = contract_address or os.getenv("CONTRACT_ADDRESS")

        # Initialize Web3 provider
        if self.rpc_url:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            if not self.w3.is_connected():
                print(f"[BlockchainClient] Warning: Could not connect to RPC URL {self.rpc_url}. Falling back to in-memory EVM.")
                self._setup_eth_tester()
        else:
            self._setup_eth_tester()

        # Set account
        if self.private_key:
            self.account = self.w3.eth.account.from_key(self.private_key).address
        else:
            self.account = self.w3.eth.accounts[0]

        # Load or deploy contract
        self.artifact = ContractDeployer.load_contract_artifact()
        self.abi = self.artifact["abi"]

        if self.contract_address and self.w3.is_checksum_address(self.contract_address):
            self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.abi)
            self.deployment_tx_hash = "PRE_DEPLOYED"
        else:
            self.contract, self.deployment_tx_hash = ContractDeployer.deploy_contract(
                self.w3,
                account=self.account,
                private_key=self.private_key
            )
            self.contract_address = self.contract.address

    def _setup_eth_tester(self):
        """Initializes Python-native EVM test provider (EthereumTesterProvider)."""
        from web3.providers.eth_tester import EthereumTesterProvider
        from eth_tester import EthereumTester, PyEVMBackend
        
        tester = EthereumTester(backend=PyEVMBackend())
        self.w3 = Web3(EthereumTesterProvider(tester))

    def submit_content_hash(
        self,
        content_hash: str,
        source_url: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Submits SHA-256 content fingerprint to smart contract on-chain.
        Returns transaction hash hex string.
        """
        metadata_json = json.dumps(metadata, sort_keys=True)

        if self.private_key:
            tx = self.contract.functions.registerContent(
                content_hash,
                source_url,
                metadata_json
            ).build_transaction({
                "from": self.account,
                "nonce": self.w3.eth.get_transaction_count(self.account),
                "gas": 500000,
                "gasPrice": self.w3.eth.gas_price
            })
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
            return tx_hash.hex()
        else:
            tx_hash = self.contract.functions.registerContent(
                content_hash,
                source_url,
                metadata_json
            ).transact({"from": self.account})
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            return receipt.transactionHash.hex()

    def get_onchain_record(self, content_hash: str) -> Dict[str, Any]:
        """
        Retrieves recorded content details from smart contract.
        """
        result = self.contract.functions.getContentRecord(content_hash).call()
        chash, url, timestamp, meta_json, submitter, exists = result

        meta_dict = {}
        if meta_json:
            try:
                meta_dict = json.loads(meta_json)
            except Exception:
                meta_dict = {"raw": meta_json}

        return {
            "content_hash": chash,
            "source_url": url,
            "timestamp": timestamp,
            "metadata": meta_dict,
            "submitter": submitter,
            "exists": exists
        }

    def verify_hash_onchain(self, content_hash: str) -> bool:
        """
        Checks if given content hash exists on-chain.
        """
        return bool(self.contract.functions.verifyContent(content_hash).call())

    def get_total_records(self) -> int:
        """Returns total count of fingerprints recorded on-chain."""
        return int(self.contract.functions.getRecordCount().call())
