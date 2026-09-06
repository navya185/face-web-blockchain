// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ContentVerifier
 * @dev Smart contract to store and verify cryptographic fingerprints of web/social media content.
 */
contract ContentVerifier {
    struct ContentRecord {
        string contentHash;    // SHA-256 hash of the discovered content
        string sourceUrl;      // Source URL where content was discovered
        uint256 timestamp;     // Block timestamp when registered
        string metadataJson;   // JSON metadata (author, title, platform, match score)
        address submitter;     // Wallet address of the submitter
    }

    // Mapping from contentHash to ContentRecord
    mapping(string => ContentRecord) private records;

    // Array of recorded content hashes for enumeration
    string[] public recordHashes;

    event ContentRegistered(
        string indexed contentHash,
        string sourceUrl,
        uint256 timestamp,
        address submitter
    );

    /**
     * @dev Register a new content fingerprint on-chain.
     */
    function registerContent(
        string calldata _contentHash,
        string calldata _sourceUrl,
        string calldata _metadataJson
    ) external returns (bool) {
        require(bytes(_contentHash).length > 0, "Content hash cannot be empty");
        require(records[_contentHash].timestamp == 0, "Content hash already registered");

        records[_contentHash] = ContentRecord({
            contentHash: _contentHash,
            sourceUrl: _sourceUrl,
            timestamp: block.timestamp,
            metadataJson: _metadataJson,
            submitter: msg.sender
        });

        recordHashes.push(_contentHash);

        emit ContentRegistered(_contentHash, _sourceUrl, block.timestamp, msg.sender);
        return true;
    }

    /**
     * @dev Retrieve stored content record by fingerprint hash.
     */
    function getContentRecord(string calldata _contentHash)
        external
        view
        returns (
            string memory contentHash,
            string memory sourceUrl,
            uint256 timestamp,
            string memory metadataJson,
            address submitter,
            bool exists
        )
    {
        ContentRecord memory record = records[_contentHash];
        if (record.timestamp == 0) {
            return ("", "", 0, "", address(0), false);
        }
        return (
            record.contentHash,
            record.sourceUrl,
            record.timestamp,
            record.metadataJson,
            record.submitter,
            true
        );
    }

    /**
     * @dev Verify whether a content hash exists on-chain.
     */
    function verifyContent(string calldata _contentHash) external view returns (bool) {
        return records[_contentHash].timestamp > 0;
    }

    /**
     * @dev Get total count of registered records.
     */
    function getRecordCount() external view returns (uint256) {
        return recordHashes.length;
    }
}
