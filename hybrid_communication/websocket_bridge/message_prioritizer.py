"""
Enterprise Message Prioritization Engine
Combines deadline-aware, QoS-based, and ML-predicted prioritization
with real-time adaptive weight adjustment for multi-agent systems
"""

import asyncio
import time
import heapq
import logging
from typing import Dict, List, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np
from sklearn.ensemble import IsolationForest  # Requires scikit-learn

logger = logging.getLogger("xibra.priority")

class PriorityException(Exception):
    """Base prioritization error"""
    pass

class MessageExpired(PriorityException):
    """Message TTL exceeded"""
    pass

class StrategyWeights:
    """Dynamic strategy weight controller"""
    def __init__(self):
        self.weights = {
            'deadline': 0.4,
            'qos': 0.3,
            'ml': 0.2,
            'security': 0.1
        }
        self.decay_factor = 0.98
        self.min_weight = 0.05

    def adjust_weights(self, feedback: Dict[str, float]):
        """Reinforcement learning-based weight adjustment"""
        total = sum(feedback.values())
        for strategy in self.weights:
            if feedback.get(strategy, 0) > 0:
                self.weights[strategy] = max(
                    self.min_weight,
                    self.weights[strategy] * self.decay_factor +
                    (feedback[strategy]/total) * (1 - self.decay_factor)
                )

class MLPriorityPredictor:
    """Anomaly detection for priority prediction"""
    def __init__(self):
        self.clf = IsolationForest(n_estimators=100)
        self._training_data = []
        self._trained = False

    async def train_model(self, historical_data: List[List[float]]):
        """Async model training"""
        if len(historical_data) < 1000:
            logger.warning("Insufficient training data")
            return
        self.clf.fit(historical_data)
        self._trained = True

    async def predict_priority(self, features: List[float]) -> float:
        """Predict message urgency score"""
        if not self._trained:
            return 0.5  # Default neutral score
        return self.clf.decision_function([features])[0]

@dataclass(order=True)
class PrioritizedMessage:
    """Message container with computed priority"""
    priority: float
    message_id: str = field(compare=False)
    payload: dict = field(compare=False)
    metadata: dict = field(compare=False)

class MessagePrioritizer:
    def __init__(self, max_queue_size: int = 10000):
        self.heap = []
        self.message_map = {}
        self.strategy_weights = StrategyWeights()
        self.ml_predictor = MLPriorityPredictor()
        self.max_queue_size = max_queue_size
        self._lock = asyncio.Lock()
        self._feedback_buffer = []

    async def add_message(self, message: dict):
        """Async message insertion with priority calculation"""
        async with self._lock:
            if len(self.heap) >= self.max_queue_size:
                self._evict_low_priority()

            priority = await self._calculate_priority(message)
            if time.time() > message.get('deadline', float('inf')):
                raise MessageExpired("Message TTL exceeded")

            pm = PrioritizedMessage(
                priority=priority,
                message_id=message['id'],
                payload=message['payload'],
                metadata=message.get('metadata', {})
            )
            
            heapq.heappush(self.heap, pm)
            self.message_map[message['id']] = pm

    async def get_next(self) -> Optional[PrioritizedMessage]:
        """Retrieve highest priority message"""
        async with self._lock:
            if not self.heap:
                return None
            return heapq.heappop(self.heap)

    async def recalculate_priorities(self):
        """Periodic priority recomputation"""
        async with self._lock:
            new_heap = []
            for pm in self.heap:
                new_priority = await self._calculate_priority({
                    **pm.payload,
                    'metadata': pm.metadata
                })
                new_pm = PrioritizedMessage(
                    priority=new_priority,
                    message_id=pm.message_id,
                    payload=pm.payload,
                    metadata=pm.metadata
                )
                heapq.heappush(new_heap, new_pm)
            self.heap = new_heap

    async def _calculate_priority(self, message: dict) -> float:
        """Multi-strategy priority calculation"""
        strategies = {
            'deadline': self._deadline_priority(message),
            'qos': self._qos_priority(message),
            'ml': await self._ml_priority(message),
            'security': self._security_priority(message)
        }
        
        weights = self.strategy_weights.weights
        return sum(strategies[s] * weights[s] for s in strategies)

    def _deadline_priority(self, message: dict) -> float:
        """Time-sensitive priority calculation"""
        now = time.time()
        deadline = message.get('deadline', now + 3600)
        time_left = deadline - now
        return 1 - min(max(time_left / 3600, 0), 1)

    def _qos_priority(self, message: dict) -> float:
        """Quality-of-Service based priority"""
        qos = message.get('metadata', {}).get('qos', 0)
        return {
            0: 0.3,  # Best-effort
            1: 0.6,  # Guaranteed delivery
            2: 0.9   # Real-time critical
        }.get(qos, 0.3)

    async def _ml_priority(self, message: dict) -> float:
        """ML-predicted priority score"""
        features = [
            message.get('retry_count', 0),
            message['metadata'].get('sender_rep_score', 0.5),
            len(message['payload']),
            time.time() - message['metadata'].get('created_at', time.time())
        ]
        return await self.ml_predictor.predict_priority(features)

    def _security_priority(self, message: dict) -> float:
        """Security level multiplier"""
        sec_level = message.get('metadata', {}).get('security_level', 0)
        return {0: 0.1, 1: 0.5, 2: 1.0}.get(sec_level, 0.1)

    def _evict_low_priority(self):
        """Evict lowest 5% messages when queue full"""
        cutoff = int(self.max_queue_size * 0.95)
        self.heap = heapq.nsmallest(cutoff, self.heap)
        heapq.heapify(self.heap)
        removed_ids = set(self.message_map) - {pm.message_id for pm in self.heap}
        for msg_id in removed_ids:
            del self.message_map[msg_id]

    async def feedback_loop(self, feedback: Dict[str, float]):
        """Adaptive learning from system feedback"""
        self._feedback_buffer.append(feedback)
        if len(self._feedback_buffer) >= 100:
            await self._process_feedback_batch()

    async def _process_feedback_batch(self):
        """Batch feedback processing"""
        strategy_scores = {s: 0.0 for s in self.strategy_weights.weights}
        for fb in self._feedback_buffer:
            for strategy in strategy_scores:
                strategy_scores[strategy] += fb.get(strategy, 0)
        
        self.strategy_weights.adjust_weights(strategy_scores)
        await self.ml_predictor.train_model(
            [list(fb.values()) for fb in self._feedback_buffer]
        )
        self._feedback_buffer.clear()

# Integration with XIBRA Monitoring
class PriorityMonitor:
    def __init__(self, prioritizer: MessagePrioritizer):
        self.prioritizer = prioritizer
        self._stats = {
            'processed': 0,
            'expired': 0,
            'avg_priority': 0.0
        }

    async def track_message(self, message: PrioritizedMessage):
        """Update monitoring statistics"""
        self._stats['processed'] += 1
        self._stats['avg_priority'] = (
            self._stats['avg_priority'] * (self._stats['processed'] - 1) +
            message.priority
        ) / self._stats['processed']

# Unit Tests
import unittest
from unittest.mock import AsyncMock

class TestMessagePrioritizer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.prioritizer = MessagePrioritizer(max_queue_size=10)
        self.monitor = PriorityMonitor(self.prioritizer)

    async def test_priority_calculation(self):
        test_msg = {
            'id': 'msg1',
            'payload': {'data': 'test'},
            'deadline': time.time() + 60,
            'metadata': {'qos': 2, 'security_level': 1}
        }
        await self.prioritizer.add_message(test_msg)
        msg = await self.prioritizer.get_next()
        self.assertGreater(msg.priority, 0.5)

    async def test_queue_eviction(self):
        for i in range(15):
            await self.prioritizer.add_message({
                'id': f'msg{i}',
                'payload': {},
                'deadline': time.time() + 3600
            })
        self.assertEqual(len(self.prioritizer.heap), 10)

if __name__ == "__main__":
    unittest.main()
