import os
import sys
from web3 import Web3

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.blockchain.client import BlockchainClient

def main():
    print("=== Deploying ContentVerifier Smart Contract ===")
    client = BlockchainClient()
    print(f"Connected Web3 Node: {client.w3.provider}")
    print(f"Deployer Account: {client.account}")
    print(f"Deployed Contract Address: {client.contract_address}")
    print(f"Deployment Tx Hash: {client.deployment_tx_hash}")
    print("=== Deployment Complete ===")

if __name__ == "__main__":
    main()
