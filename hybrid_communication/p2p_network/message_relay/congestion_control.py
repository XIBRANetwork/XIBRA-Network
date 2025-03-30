"""
Advanced Congestion Control Module
Combines BBR-inspired bandwidth estimation with CUBIC window adjustment
Supports multi-path aware congestion management
"""

import math
import logging
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, Deque, Optional, Tuple
from collections import deque
import statistics

logger = logging.getLogger("xibra.p2p.congestion")

class CongestionControlError(Exception):
    """Base exception for congestion management failures"""
    pass

@dataclass(frozen=True)
class NetworkState:
    rtt: float  # milliseconds
    loss_rate: float  # 0.0-1.0
    throughput: float  # Mbps
    bandwidth_est: float  # Mbps
    inflight_packets: int

class CongestionSignal:
    def __init__(self):
        self._event = asyncio.Event()
        self._state: Optional[NetworkState] = None

    def update(self, state: NetworkState) -> None:
        self._state = state
        self._event.set()
        self._event.clear()

    async def wait(self) -> NetworkState:
        await self._event.wait()
        return self._state

class BaseCongestionController:
    def __init__(self, initial_window: int = 10):
        self.cwnd = initial_window  # Congestion window
        self.ss_thresh = float('inf')  # Slow start threshold
        self.min_rtt = float('inf')
        self.max_bandwidth = 0.0
        self._state_history: Deque[NetworkState] = deque(maxlen=100)
        self._last_update = time.monotonic()

    def update_state(self, state: NetworkState) -> None:
        """Update controller with latest network metrics"""
        self._state_history.append(state)
        self.min_rtt = min(self.min_rtt, state.rtt)
        self.max_bandwidth = max(self.max_bandwidth, state.bandwidth_est)
        self._last_update = time.monotonic()

    def calculate_window(self) -> int:
        """Return recommended congestion window size"""
        raise NotImplementedError

    def detect_timeout(self) -> bool:
        """Check for network timeout conditions"""
        elapsed = time.monotonic() - self._last_update
        return elapsed > (self.min_rtt * 4 / 1000)  # Convert ms to seconds

class BBRController(BaseCongestionController):
    """
    BBR-inspired congestion control
    Implements bandwidth-delay product estimation
    """
    def __init__(self, initial_window: int = 10):
        super().__init__(initial_window)
        self.btl_bw = 0.0  # Bottleneck bandwidth
        self.rt_prop = float('inf')  # Round-trip propagation time

    def update_state(self, state: NetworkState):
        super().update_state(state)
        self.rt_prop = min(self.rt_prop, state.rtt)
        self.btl_bw = max(self.btl_bw, state.bandwidth_est)

    def calculate_window(self) -> int:
        if self.btl_bw == 0 or self.rt_prop == float('inf'):
            return self.cwnd

        bdp = self.btl_bw * self.rt_prop / 8  # Bandwidth-delay product (packets)
        gain = 2.0 if self.cwnd < bdp else 1.0
        return min(math.ceil(bdp * gain), 2**30)

class CUBICController(BaseCongestionController):
    """
    CUBIC-inspired congestion control
    Implements window growth cubic function
    """
    def __init__(self, initial_window: int = 10):
        super().__init__(initial_window)
        self.w_max = 0
        self.k = 0.0
        self.t_epoch = 0.0

    def _cubic_function(self, t: float) -> float:
        return self.w_max * (1 - 0.7)**3 + 0.7 * (t - self.k)**3

    def calculate_window(self) -> int:
        if len(self._state_history) < 2:
            return self.cwnd

        # Detect packet loss
        loss_threshold = 0.02
        current_loss = self._state_history[-1].loss_rate

        if current_loss > loss_threshold:
            # Multiplicative decrease
            self.w_max = self.cwnd
            self.cwnd = max(math.floor(self.cwnd * 0.7), 1)
            self.ss_thresh = self.cwnd
            self.t_epoch = time.monotonic()
            return self.cwnd

        # Calculate time since last congestion event
        t = time.monotonic() - self.t_epoch
        self.k = math.cbrt(self.w_max * 0.3 / 0.7)

        if self.cwnd < self.ss_thresh:
            # Slow start
            self.cwnd += 1
        else:
            # Cubic growth
            target = self._cubic_function(t)
            if self.cwnd < target:
                self.cwnd = math.floor(target)
            else:
                self.cwnd += 0.01 * (target - self.cwnd)

        return math.floor(self.cwnd)

class HybridController(BaseCongestionController):
    """
    Adaptive hybrid controller combining BBR and loss-based approaches
    """
    def __init__(self, initial_window: int = 10):
        super().__init__(initial_window)
        self.bbr = BBRController(initial_window)
        self.cubic = CUBICController(initial_window)
        self.mode: str = 'bbr'  # 'bbr' or 'loss_based'

    def update_state(self, state: NetworkState):
        super().update_state(state)
        self.bbr.update_state(state)
        self.cubic.update_state(state)

        # Mode switching logic
        loss_rate = statistics.mean(s.loss_rate for s in self._state_history)
        if loss_rate > 0.1:
            self.mode = 'loss_based'
        elif state.bandwidth_est >= self.max_bandwidth * 0.8:
            self.mode = 'bbr'

    def calculate_window(self) -> int:
        if self.mode == 'bbr':
            return self.bbr.calculate_window()
        return self.cubic.calculate_window()

class MultiPathController:
    """
    Manages congestion control across multiple network paths
    Implements coupled congestion control for shared bottlenecks
    """
    def __init__(self, initial_window_per_path: int = 5):
        self.controllers: Dict[str, BaseCongestionController] = {}
        self.global_window = initial_window_per_path
        self._alpha = 0.9  # Coupling factor

    def add_path(self, path_id: str) -> None:
        self.controllers[path_id] = HybridController()

    def update_path_state(self, path_id: str, state: NetworkState) -> None:
        if path_id not in self.controllers:
            self.add_path(path_id)
        self.controllers[path_id].update_state(state)
        self._update_global_state()

    def _update_global_state(self) -> None:
        total_window = sum(c.cwnd for c in self.controllers.values())
        avg_rtt = statistics.mean(c.min_rtt for c in self.controllers.values())
        
        # Coupled window adjustment
        for controller in self.controllers.values():
            fair_share = total_window * (controller.cwnd / total_window)**self._alpha
            controller.cwnd = math.floor(fair_share)

    def get_path_window(self, path_id: str) -> int:
        return self.controllers[path_id].calculate_window()

class CongestionControlManager:
    """
    Central congestion management system with dynamic strategy switching
    """
    STRATEGIES = {
        'bbr': BBRController,
        'cubic': CUBICController,
        'hybrid': HybridController
    }

    def __init__(self, strategy: str = 'hybrid'):
        self.active_strategy = self.STRATEGIES[strategy]()
        self.fallback_strategy = CUBICController()
        self._strategy_lock = asyncio.Lock()

    async def dynamic_strategy_switch(self, signal: CongestionSignal) -> None:
        """Monitor network conditions and adjust control strategy"""
        while True:
            state = await signal.wait()
            async with self._strategy_lock:
                if state.loss_rate > 0.2:
                    self.active_strategy = self.fallback_strategy
                else:
                    self.active_strategy = HybridController()

    def get_window_size(self) -> int:
        return self.active_strategy.calculate_window()

    def update_network_state(self, state: NetworkState) -> None:
        self.active_strategy.update_state(state)
        self.fallback_strategy.update_state(state)

# Unit Tests
import unittest
from unittest import IsolatedAsyncioTestCase

class TestCongestionControl(IsolatedAsyncioTestCase):
    def test_bbr_window_growth(self):
        controller = BBRController()
        state = NetworkState(rtt=50, loss_rate=0, throughput=100, bandwidth_est=100, inflight_packets=10)
        controller.update_state(state)
        window = controller.calculate_window()
        self.assertGreater(window, 10)

    def test_cubic_loss_reaction(self):
        controller = CUBICController(initial_window=20)
        state = NetworkState(rtt=100, loss_rate=0.03, throughput=50, bandwidth_est=50, inflight_packets=15)
        controller.update_state(state)
        window = controller.calculate_window()
        self.assertLess(window, 20)

    async def test_multipath_coupling(self):
        mp = MultiPathController()
        mp.add_path('p1')
        mp.add_path('p2')
        
        state1 = NetworkState(rtt=50, loss_rate=0, throughput=100, bandwidth_est=100, inflight_packets=10)
        state2 = NetworkState(rtt=60, loss_rate=0, throughput=80, bandwidth_est=80, inflight_packets=8)
        
        mp.update_path_state('p1', state1)
        mp.update_path_state('p2', state2)
        
        self.assertAlmostEqual(mp.get_path_window('p1'), mp.get_path_window('p2'), delta=2)

if __name__ == "__main__":
    unittest.main()
