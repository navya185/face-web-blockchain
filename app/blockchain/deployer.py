import os
import json
from typing import Tuple, Dict, Any, Optional
from web3 import Web3
from web3.contract import Contract

class ContractDeployer:
    """
    Handles loading, compiling, and deploying ContentVerifier.sol.
    """
    CONTRACT_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "contracts", "ContentVerifier.json")
    CONTRACT_SOL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "contracts", "ContentVerifier.sol")

    @classmethod
    def load_contract_artifact(cls) -> Dict[str, Any]:
        """
        Loads pre-compiled contract ABI and Bytecode artifact.
        If JSON is missing, attempts dynamic solc compilation.
        """
        abs_json = os.path.abspath(cls.CONTRACT_JSON_PATH)
        if os.path.exists(abs_json):
            with open(abs_json, "r") as f:
                return json.load(f)

        # Fallback to py-solc-x dynamic compilation if available
        try:
            import solcx
            abs_sol = os.path.abspath(cls.CONTRACT_SOL_PATH)
            if os.path.exists(abs_sol):
                compiled = solcx.compile_files([abs_sol], solc_version="0.8.20")
                key = [k for k in compiled.keys() if "ContentVerifier" in k][0]
                return {
                    "abi": compiled[key]["abi"],
                    "bin": compiled[key]["bin"]
                }
        except Exception as e:
            print(f"[ContractDeployer] Dynamic compilation notice: {e}")

        raise FileNotFoundError(f"Contract artifact not found at {abs_json}")

    @classmethod
    def deploy_contract(cls, w3: Web3, account: Optional[str] = None, private_key: Optional[str] = None) -> Tuple[Contract, str]:
        """
        Deploys ContentVerifier smart contract on Web3 connection.
        Returns (ContractInstance, transaction_hash_hex).
        """
        artifact = cls.load_contract_artifact()
        abi = artifact["abi"]
        bytecode = artifact["bin"]

        if not account:
            account = w3.eth.accounts[0]

        ContractFactory = w3.eth.contract(abi=abi, bytecode=bytecode)

        if private_key:
            # Build and sign transaction
            tx = ContractFactory.constructor().build_transaction({
                "from": account,
                "nonce": w3.eth.get_transaction_count(account),
                "gas": 3000000,
                "gasPrice": w3.eth.gas_price
            })
            signed_tx = w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            contract = w3.eth.contract(address=receipt.contractAddress, abi=abi)
            return contract, receipt.transactionHash.hex()
        else:
            # Unsigned deployment for local EthTester or unlocked local nodes
            tx_hash = ContractFactory.constructor().transact({"from": account})
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            contract = w3.eth.contract(address=receipt.contractAddress, abi=abi)
            return contract, receipt.transactionHash.hex()
