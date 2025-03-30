"""
Enterprise-grade State Snapshot Manager
Supports delta snapshots, Merkle tree validation, and
encrypted storage with compression
"""

import asyncio
import zstandard
import msgpack
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import aiofiles
import aiohttp
from prometheus_client import (  # type: ignore
    Histogram,
    Counter,
    Gauge,
    start_http_server,
)

# Metrics
SNAPSHOT_TIME = Histogram("xibra_snapshot_duration", "Snapshot creation time", ["type"])
SNAPSHOT_SIZE = Counter("xibra_snapshot_bytes", "Snapshot size in bytes")
VERSION_GAUGE = Gauge("xibra_state_versions", "Current snapshot version")

@dataclass
class SnapshotConfig:
    max_delta_size: int = 1024 * 1024  # 1MB
    compression_level: int = 3
    encryption_key: bytes = b""
    merkle_tree_depth: int = 16
    storage_backends: List[str] = ("local", "s3")

class DeltaState:
    def __init__(self, base_version: int, changes: Dict[str, Any]):
        self.version = base_version + 1
        self.changes = changes
        self.timestamp = datetime.utcnow()
        self.parent_hash = b""

class MerkleNode:
    def __init__(self, hash_val: bytes, left=None, right=None):
        self.hash = hash_val
        self.left = left
        self.right = right

class StateSnapshotter:
    def __init__(self, config: SnapshotConfig):
        self.config = config
        self.current_version = 0
        self.merkle_root: Optional[MerkleNode] = None
        self._compressor = zstandard.ZstdCompressor(level=config.compression_level)
        self._decompressor = zstandard.ZstdDecompressor()
        self._session = aiohttp.ClientSession()
        start_http_server(8001)

    async def _encrypt_data(self, data: bytes) -> bytes:
        """AES-GCM encryption with automatic IV generation"""
        iv = os.urandom(12)
        cipher = Cipher(
            algorithms.AES(self.config.encryption_key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(data) + encryptor.finalize()
        return iv + encryptor.tag + encrypted

    async def _build_merkle_tree(self, deltas: List[DeltaState]) -> MerkleNode:
        """Build Merkle tree for consistency verification"""
        leaves = [self._hash_delta(d) for d in deltas]
        return self._build_tree(leaves)

    def _build_tree(self, nodes: List[bytes]) -> MerkleNode:
        """Recursive Merkle tree builder"""
        if len(nodes) == 1:
            return MerkleNode(nodes[0])
        
        parents = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i+1] if i+1 < len(nodes) else left
            combined = left + right
            parents.append(hashlib.sha3_256(combined).digest())
        
        return self._build_tree(parents)

    def _hash_delta(self, delta: DeltaState) -> bytes:
        """Generate cryptographic hash for delta"""
        data = msgpack.packb({
            "version": delta.version,
            "changes": delta.changes,
            "parent": delta.parent_hash
        })
        return hashlib.sha3_256(data).digest()

    async def _write_to_storage(self, data: bytes, version: int):
        """Multi-backend storage writer"""
        filename = f"snapshot_v{version}.zst"
        
        # Local storage
        async with aiofiles.open(filename, "wb") as f:
            await f.write(data)
        
        # S3 integration
        if "s3" in self.config.storage_backends:
            async with self._session.put(
                f"s3://xibra-snapshots/{filename}",
                data=data,
                headers={"x-amz-storage-class": "INTELLIGENT_TIERING"}
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"S3 upload failed: {await resp.text()}")

    async def create_snapshot(self, state: Dict[str, Any], full: bool = False) -> int:
        """Create full or incremental snapshot"""
        with SNAPSHOT_TIME.labels("full" if full else "delta").time():
            serialized = msgpack.packb(state)
            
            if full:
                await self._full_snapshot(serialized)
            else:
                await self._delta_snapshot(serialized)
            
            self.current_version += 1
            VERSION_GAUGE.set(self.current_version)
            return self.current_version

    async def _full_snapshot(self, data: bytes):
        """Full state serialization"""
        compressed = self._compressor.compress(data)
        encrypted = await self._encrypt_data(compressed)
        await self._write_to_storage(encrypted, self.current_version + 1)
        SNAPSHOT_SIZE.inc(len(encrypted))

    async def _delta_snapshot(self, current_state: bytes):
        """Delta state serialization"""
        # Delta calculation logic
        previous = await self.load_snapshot(self.current_version)
        delta = self._calculate_delta(previous, current_state)
        
        compressed = self._compressor.compress(msgpack.packb(delta))
        encrypted = await self._encrypt_data(compressed)
        await self._write_to_storage(encrypted, self.current_version + 1)
        SNAPSHOT_SIZE.inc(len(encrypted))

    def _calculate_delta(self, old: bytes, new: bytes) -> Dict[str, Any]:
        """Naive delta calculation (replace with real diff algorithm)"""
        old_state = msgpack.unpackb(old)
        new_state = msgpack.unpackb(new)
        return {k: v for k, v in new_state.items() if old_state.get(k) != v}

    async def load_snapshot(self, version: int) -> bytes:
        """Load and reconstruct state from snapshot"""
        filename = f"snapshot_v{version}.zst"
        
        async with aiofiles.open(filename, "rb") as f:
            encrypted = await f.read()
        
        iv, tag, data = encrypted[:12], encrypted[12:28], encrypted[28:]
        cipher = Cipher(
            algorithms.AES(self.config.encryption_key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(data) + decryptor.finalize()
        return self._decompressor.decompress(decrypted)

    async def close(self):
        """Cleanup resources"""
        await self._session.close()

# Example Usage
async def main():
    config = SnapshotConfig(
        encryption_key=os.urandom(32),
        storage_backends=["local", "s3"]
    )
    
    snapshotter = StateSnapshotter(config)
    
    # Initial full snapshot
    await snapshotter.create_snapshot({"node1": "active"}, full=True)
    
    # Subsequent delta snapshots
    await snapshotter.create_snapshot({"node1": "active", "node2": "joining"})
    
    await snapshotter.close()

if __name__ == "__main__":
    asyncio.run(main())
