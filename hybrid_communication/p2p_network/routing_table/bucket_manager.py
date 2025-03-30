"""
Distributed Bucket Manager for P2P Routing Table
Implements XOR-based metric space with configurable bucket sizes
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from abc import ABC, abstractmethod
from collections import OrderedDict
import time

logger = logging.getLogger("xibra.p2p.routing")

@dataclass(frozen=True)
class NodeInfo:
    node_id: bytes
    endpoint: str
    last_seen: float = time.time()
    protocol_version: int = 1

class BucketFullException(Exception):
    def __init__(self, bucket_index: int, replacement_candidate: NodeInfo):
        super().__init__(f"Bucket {bucket_index} full")
        self.replacement_candidate = replacement_candidate

class IBucketStrategy(ABC):
    @abstractmethod
    def should_update(self, existing: NodeInfo, new: NodeInfo) -> bool:
        pass
    
    @abstractmethod
    def get_replacement_candidate(self, nodes: OrderedDict[bytes, NodeInfo]) -> NodeInfo:
        pass

class LRUBucketStrategy(IBucketStrategy):
    def should_update(self, existing: NodeInfo, new: NodeInfo) -> bool:
        return new.last_seen > existing.last_seen
    
    def get_replacement_candidate(self, nodes: OrderedDict[bytes, NodeInfo]) -> NodeInfo:
        return next(iter(nodes.values()))

class RoutingTable:
    def __init__(
        self,
        local_node_id: bytes,
        bucket_size: int = 20,
        replacement_strategy: IBucketStrategy = LRUBucketStrategy(),
        num_buckets: int = 256
    ):
        self.local_node_id = local_node_id
        self.bucket_size = bucket_size
        self.strategy = replacement_strategy
        self.buckets: List[OrderedDict[bytes, NodeInfo]] = [
            OrderedDict() for _ in range(num_buckets)
        ]
        self.lock = asyncio.Lock()

    def _get_bucket_index(self, node_id: bytes) -> int:
        """Calculate XOR distance bucket index"""
        xor_result = int.from_bytes(self.local_node_id, 'big') ^ int.from_bytes(node_id, 'big')
        return xor_result.bit_length() - 1 if xor_result else 0

    async def update_node(self, node: NodeInfo) -> bool:
        """Insert or update node with thread-safe LRU logic"""
        async with self.lock:
            bucket_index = self._get_bucket_index(node.node_id)
            bucket = self.buckets[bucket_index]
            
            if node.node_id in bucket:
                if self.strategy.should_update(bucket[node.node_id], node):
                    bucket.move_to_end(node.node_id)
                    bucket[node.node_id] = node
                    logger.debug(f"Updated node {node.node_id.hex()[:8]} in bucket {bucket_index}")
                return True
            
            if len(bucket) < self.bucket_size:
                bucket[node.node_id] = node
                logger.info(f"Added node {node.node_id.hex()[:8]} to bucket {bucket_index}")
                return True
            
            try:
                candidate = self.strategy.get_replacement_candidate(bucket)
                raise BucketFullException(bucket_index, candidate)
            except StopIteration:
                logger.error(f"Empty bucket {bucket_index} but full?")
                return False

    async def remove_node(self, node_id: bytes) -> bool:
        """Remove node from routing table"""
        async with self.lock:
            bucket_index = self._get_bucket_index(node_id)
            if node_id in self.buckets[bucket_index]:
                del self.buckets[bucket_index][node_id]
                logger.warning(f"Removed node {node_id.hex()[:8]} from bucket {bucket_index}")
                return True
            return False

    async def find_closest_nodes(
        self,
        target_id: bytes,
        count: int,
        exclude: Optional[Set[bytes]] = None
    ) -> List[NodeInfo]:
        """Find closest nodes using XOR distance metric"""
        exclude = exclude or set()
        results = []
        
        async with self.lock:
            bucket_index = self._get_bucket_index(target_id)
            
            # Search target bucket
            for node in reversed(self.buckets[bucket_index].values()):
                if node.node_id not in exclude:
                    results.append(node)
                    if len(results) >= count:
                        return sorted(results, key=lambda n: self._xor_distance(n.node_id, target_id))[:count]
            
            # Expand to neighboring buckets
            lower, upper = bucket_index - 1, bucket_index + 1
            while len(results) < count and (lower >= 0 or upper < len(self.buckets)):
                if lower >= 0:
                    results.extend(self.buckets[lower].values())
                    lower -= 1
                if upper < len(self.buckets):
                    results.extend(self.buckets[upper].values())
                    upper += 1
            
            # Sort and truncate
            return sorted(
                [n for n in results if n.node_id not in exclude],
                key=lambda n: self._xor_distance(n.node_id, target_id)
            )[:count]

    def _xor_distance(self, a: bytes, b: bytes) -> int:
        return int.from_bytes(a, 'big') ^ int.from_bytes(b, 'big')

    async def get_all_nodes(self) -> List[NodeInfo]:
        """Get all nodes in routing table"""
        async with self.lock:
            return [node for bucket in self.buckets for node in bucket.values()]

    async def bucket_stats(self) -> Dict[str, int]:
        """Get routing table statistics"""
        async with self.lock:
            return {
                "total_buckets": len(self.buckets),
                "occupied_buckets": sum(1 for b in self.buckets if len(b) > 0),
                "total_nodes": sum(len(b) for b in self.buckets)
            }

class BucketManager:
    def __init__(
        self,
        local_node_id: bytes,
        refresh_interval: int = 300,
        bucket_size: int = 20
    ):
        self.routing_table = RoutingTable(local_node_id, bucket_size)
        self.refresh_task: Optional[asyncio.Task] = None
        self.refresh_interval = refresh_interval

    async def start(self) -> None:
        """Start background bucket maintenance tasks"""
        self.refresh_task = asyncio.create_task(self._bucket_refresh_loop())
        logger.info("Bucket manager started")

    async def stop(self) -> None:
        """Stop background tasks"""
        if self.refresh_task:
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("Bucket manager stopped")

    async def _bucket_refresh_loop(self) -> None:
        """Periodically refresh stale buckets"""
        while True:
            await self._refresh_buckets()
            await asyncio.sleep(self.refresh_interval)

    async def _refresh_buckets(self) -> None:
        """Ping least-recently seen nodes in each bucket"""
        all_nodes = await self.routing_table.get_all_nodes()
        now = time.time()
        
        for node in all_nodes:
            if now - node.last_seen > self.refresh_interval:
                # TODO: Implement actual ping RPC
                logger.debug(f"Refreshing node {node.node_id.hex()[:8]}")
                # Update last_seen on successful ping
                await self.routing_table.update_node(
                    NodeInfo(node.node_id, node.endpoint, time.time())
                )

# Unit Tests
import unittest
from unittest import IsolatedAsyncioTestCase

class TestBucketManager(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.local_id = bytes.fromhex("89d785479d2e723ae355")
        self.manager = BucketManager(self.local_id, bucket_size=2)
        await self.manager.start()

    async def asyncTearDown(self):
        await self.manager.stop()

    async def test_node_insertion(self):
        node = NodeInfo(bytes.fromhex("1234"), "1.2.3.4:5678")
        await self.manager.routing_table.update_node(node)
        stats = await self.manager.routing_table.bucket_stats()
        self.assertEqual(stats["total_nodes"], 1)

    async def test_bucket_replacement(self):
        nodes = [
            NodeInfo(bytes([i]*20), f"10.0.0.{i}:8888")
            for i in range(3)
        ]
        
        for node in nodes:
            await self.manager.routing_table.update_node(node)
        
        stats = await self.manager.routing_table.bucket_stats()
        self.assertEqual(stats["total_nodes"], 2)
