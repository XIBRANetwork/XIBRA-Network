"""
Advanced Distance Calculator for P2P Routing
Supports XOR, Latency-Weighted, and Geo-Aware distance metrics
"""

from __future__ import annotations
import math
import logging
import asyncio
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Protocol
from functools import lru_cache
from collections import defaultdict
import time

logger = logging.getLogger("xibra.p2p.distance")

class DistanceCalculationError(Exception):
    """Base exception for distance calculation failures"""
    pass

class InvalidNodeIDError(DistanceCalculationError):
    """Raised when node IDs have incompatible lengths"""
    pass

@dataclass(frozen=True)
class NetworkProfile:
    latency: float  # milliseconds
    bandwidth: float  # Mbps
    stability: float  # 0.0-1.0 reliability factor

class IDistanceMetric(Protocol):
    async def calculate(self, node_a: bytes, node_b: bytes) -> float:
        ...

class BaseDistanceCalculator:
    def __init__(self, id_length: int = 20):
        self.id_length = id_length
        self._profile_cache: Dict[bytes, NetworkProfile] = defaultdict(
            lambda: NetworkProfile(0, 0, 0)
        )
        self._lock = asyncio.Lock()

    def _validate_ids(self, node_a: bytes, node_b: bytes) -> None:
        if len(node_a) != self.id_length or len(node_b) != self.id_length:
            raise InvalidNodeIDError(
                f"Node ID length mismatch: {len(node_a)} vs {len(node_b)}"
            )

    async def update_network_profile(self, node_id: bytes, profile: NetworkProfile) -> None:
        async with self._lock:
            self._profile_cache[node_id] = profile

    @lru_cache(maxsize=2048)
    def _xor_distance(self, node_a: bytes, node_b: bytes) -> int:
        return int.from_bytes(node_a, 'big') ^ int.from_bytes(node_b, 'big')

class XORDistanceCalculator(BaseDistanceCalculator):
    """Pure Kademlia-style XOR distance"""
    async def calculate(self, node_a: bytes, node_b: bytes) -> float:
        self._validate_ids(node_a, node_b)
        return float(self._xor_distance(node_a, node_b))

class HybridDistanceCalculator(BaseDistanceCalculator):
    """
    Combines XOR with network conditions using adaptive weights:
    distance = α*(xor) + β*(latency) + γ*(1/stability)
    """
    def __init__(
        self,
        id_length: int = 20,
        alpha: float = 0.6,
        beta: float = 0.3,
        gamma: float = 0.1
    ):
        super().__init__(id_length)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._weight_update_interval = 300  # seconds
        self._last_weight_update = time.time()

    async def _auto_adjust_weights(self) -> None:
        """Dynamically adjust weights based on network conditions"""
        if time.time() - self._last_weight_update > self._weight_update_interval:
            async with self._lock:
                # Placeholder for real weight adjustment logic
                total_nodes = len(self._profile_cache)
                if total_nodes > 100:
                    self.alpha = 0.5
                    self.beta = 0.4
                    self.gamma = 0.1
                self._last_weight_update = time.time()

    async def calculate(self, node_a: bytes, node_b: bytes) -> float:
        await self._auto_adjust_weights()
        self._validate_ids(node_a, node_b)
        
        xor_dist = self._xor_distance(node_a, node_b)
        profile_a = self._profile_cache[node_a]
        profile_b = self._profile_cache[node_b]

        latency = abs(profile_a.latency - profile_b.latency)
        stability = (profile_a.stability + profile_b.stability) / 2
        
        return (
            self.alpha * xor_dist +
            self.beta * latency +
            self.gamma * (1 / max(stability, 0.01))
        )

class GeoAwareDistanceCalculator(BaseDistanceCalculator):
    """Combines XOR with geographical distance (requires GeoIP data)"""
    def __init__(self, id_length: int = 20):
        super().__init__(id_length)
        self._geo_cache: Dict[bytes, Tuple[float, float]] = {}  # node_id -> (lat, lon)

    async def update_geo_data(self, node_id: bytes, lat: float, lon: float) -> None:
        async with self._lock:
            self._geo_cache[node_id] = (lat, lon)

    def _haversine(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate great-circle distance in kilometers"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        R = 6371  # Earth radius in km
        φ1 = math.radians(lat1)
        φ2 = math.radians(lat2)
        Δφ = math.radians(lat2 - lat1)
        Δλ = math.radians(lon2 - lon1)

        a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

    async def calculate(self, node_a: bytes, node_b: bytes) -> float:
        self._validate_ids(node_a, node_b)
        
        xor_dist = self._xor_distance(node_a, node_b)
        
        if node_a not in self._geo_cache or node_b not in self._geo_cache:
            logger.warning("Missing geo data, falling back to XOR")
            return float(xor_dist)

        geo_dist = self._haversine(
            self._geo_cache[node_a],
            self._geo_cache[node_b]
        )
        
        return 0.7 * xor_dist + 0.3 * geo_dist

class DistanceCalculatorFactory:
    @staticmethod
    def create_calculator(
        strategy: str = "xor",
        **kwargs
    ) -> BaseDistanceCalculator:
        strategies = {
            "xor": XORDistanceCalculator,
            "hybrid": HybridDistanceCalculator,
            "geo": GeoAwareDistanceCalculator
        }
        
        if strategy not in strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Choose from {list(strategies.keys())}")
        
        return strategies[strategy](**kwargs)

# Unit Tests
import unittest
from unittest import IsolatedAsyncioTestCase

class TestDistanceCalculators(IsolatedAsyncioTestCase):
    async def test_xor_calculation(self):
        calculator = XORDistanceCalculator()
        id1 = b'\x00' * 20
        id2 = b'\xff' * 20
        distance = await calculator.calculate(id1, id2)
        self.assertEqual(distance, int.from_bytes(id2, 'big'))

    async def test_hybrid_weights(self):
        calculator = HybridDistanceCalculator(alpha=0.5, beta=0.3, gamma=0.2)
        await calculator.update_network_profile(b'\x01'*20, NetworkProfile(10, 100, 0.9))
        await calculator.update_network_profile(b'\x02'*20, NetworkProfile(20, 50, 0.8))
        distance = await calculator.calculate(b'\x01'*20, b'\x02'*20)
        self.assertGreater(distance, 0)

    async def test_geo_fallback(self):
        calculator = GeoAwareDistanceCalculator()
        with self.assertLogs(logger, level='WARNING'):
            distance = await calculator.calculate(b'\x03'*20, b'\x04'*20)
            self.assertGreater(distance, 0)

if __name__ == "__main__":
    unittest.main()
