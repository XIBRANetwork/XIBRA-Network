"""
Enterprise-grade Latency Monitoring System
Supports ICMP/TCP/UDP latency tracking with EWMA forecasting
and adaptive thresholding for anomaly detection
"""

import asyncio
import time
import statistics
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import numpy as np
from scipy import signal
import asyncping
import psutil

logger = logging.getLogger("xibra.monitor.latency")

class LatencyMonitorError(Exception):
    """Base exception for monitoring failures"""
    pass

@dataclass(frozen=True)
class LatencyDataPoint:
    target: str
    protocol: str
    value: float  # milliseconds
    timestamp: float
    source_node: str
    sequence: int

@dataclass
class LatencyStatistics:
    mean: float
    median: float
    std_dev: float
    min: float
    max: float
    percentile_95: float
    ewma: float
    trend: float  # Slope of last 10 points

class LatencyMonitor:
    def __init__(
        self,
        update_interval: float = 5.0,
        history_size: int = 1000,
        anomaly_threshold: float = 3.0,
        protocols: List[str] = ["icmp", "tcp", "udp"]
    ):
        self._update_interval = update_interval
        self._history: Dict[Tuple[str, str], Deque[LatencyDataPoint]] = {}
        self._stats_cache: Dict[Tuple[str, str], LatencyStatistics] = {}
        self._anomaly_threshold = anomaly_threshold
        self._protocols = protocols
        self._sequence_counter = 0
        self._ewma_alpha = 0.2
        self._lock = asyncio.Lock()
        self._active = False
        self._background_tasks = set()

    async def start(self):
        """Start background monitoring tasks"""
        self._active = True
        task = asyncio.create_task(self._monitoring_loop())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def stop(self):
        """Graceful shutdown"""
        self._active = False
        await asyncio.gather(*self._background_tasks)

    async def measure_latency(
        self, 
        target: str, 
        protocol: str = "icmp",
        timeout: float = 2.0
    ) -> LatencyDataPoint:
        """Perform protocol-specific latency measurement"""
        try:
            start_time = time.monotonic()
            
            if protocol == "icmp":
                latency = await self._measure_icmp(target, timeout)
            elif protocol == "tcp":
                latency = await self._measure_tcp(target, 80, timeout)
            elif protocol == "udp":
                latency = await self._measure_udp(target, 53, timeout)
            else:
                raise LatencyMonitorError(f"Unsupported protocol: {protocol}")

            measurement = LatencyDataPoint(
                target=target,
                protocol=protocol,
                value=latency,
                timestamp=time.time(),
                source_node=self._get_node_id(),
                sequence=self._sequence_counter
            )
            
            async with self._lock:
                self._sequence_counter += 1
                key = (target, protocol)
                if key not in self._history:
                    self._history[key] = deque(maxlen=1000)
                self._history[key].append(measurement)
                self._stats_cache.pop(key, None)  # Invalidate cache

            return measurement

        except Exception as e:
            logger.error(f"Latency measurement failed: {str(e)}")
            raise LatencyMonitorError("Measurement error") from e

    async def get_statistics(
        self, 
        target: str, 
        protocol: str,
        window_size: int = 100
    ) -> LatencyStatistics:
        """Get computed statistics with caching"""
        key = (target, protocol)
        
        async with self._lock:
            if key in self._stats_cache:
                return self._stats_cache[key]
                
            data_points = list(self._history.get(key, deque()))[-window_size:]
            if not data_points:
                raise LatencyMonitorError("No data available")

            values = [dp.value for dp in data_points]
            timestamps = [dp.timestamp for dp in data_points]
            
            # Basic statistics
            mean = statistics.mean(values)
            median = statistics.median(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
            min_val = min(values)
            max_val = max(values)
            percentile_95 = np.percentile(values, 95)
            
            # EWMA calculation
            ewma = values[0]
            for value in values[1:]:
                ewma = self._ewma_alpha * value + (1 - self._ewma_alpha) * ewma
                
            # Trend analysis
            if len(timestamps) > 10:
                x = np.array(timestamps[-10:]) - timestamps[-10]
                y = np.array(values[-10:])
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0.0
                
            stats = LatencyStatistics(
                mean=mean,
                median=median,
                std_dev=std_dev,
                min=min_val,
                max=max_val,
                percentile_95=percentile_95,
                ewma=ewma,
                trend=slope
            )
            
            self._stats_cache[key] = stats
            return stats

    async def detect_anomalies(self) -> Dict[Tuple[str, str], bool]:
        """Check for latency anomalies across all targets"""
        anomalies = {}
        async with self._lock:
            for key in self._history:
                try:
                    stats = await self.get_statistics(key[0], key[1])
                    current = self._history[key][-1].value
                    
                    # Adaptive threshold based on EWMA and trend
                    threshold = stats.ewma + \
                              (stats.trend * self._update_interval) + \
                              (self._anomaly_threshold * stats.std_dev)
                              
                    anomalies[key] = current > threshold
                except Exception as e:
                    logger.warning(f"Anomaly detection failed for {key}: {str(e)}")
                    anomalies[key] = False
        return anomalies

    async def _monitoring_loop(self):
        """Periodic measurement of configured targets"""
        while self._active:
            tasks = []
            for target in self._get_monitored_targets():
                for protocol in self._protocols:
                    tasks.append(
                        self.measure_latency(target, protocol)
                    )
            
            # Execute measurements in parallel
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Perform predictive maintenance
            await self._predictive_analysis()
            
            await asyncio.sleep(self._update_interval)

    async def _predictive_analysis(self):
        """Time-series forecasting using ARIMA-like model"""
        for key in self._history:
            data = [dp.value for dp in self._history[key]]
            if len(data) < 50:
                continue
                
            # Savitzky-Golay filter for trend extraction
            trend = signal.savgol_filter(data, window_length=21, polyorder=2)
            
            # Simple linear prediction
            last_trend = trend[-10:]
            x = np.arange(len(last_trend))
            slope = np.polyfit(x, last_trend, 1)[0]
            predicted = trend[-1] + slope * self._update_interval
            
            # Update adaptive threshold
            current_std = statistics.stdev(data[-100:]) if len(data) >= 100 else 0.0
            self._anomaly_threshold = max(2.0, min(5.0, 3 * current_std))

    def _get_monitored_targets(self) -> List[str]:
        """Retrieve targets from network topology"""
        # Implementation would integrate with XIBRA's routing table
        return ["node1.xibra.net", "node2.xibra.net", "gateway.xibra.net"]

    def _get_node_id(self) -> str:
        """Get current node identifier"""
        # Implementation would use XIBRA's node registry
        return f"node_{psutil.cpu_percent()}_{psutil.net_io_counters().bytes_sent}"

    async def _measure_icmp(self, target: str, timeout: float) -> float:
        """ICMP ping measurement with asyncping"""
        try:
            result = await asyncping.ping(target, timeout=timeout)
            return result.avg_rtt
        except asyncping.exceptions.PingError as e:
            logger.warning(f"ICMP failed to {target}: {str(e)}")
            return timeout * 1000  # Return timeout value as penalty

    async def _measure_tcp(self, target: str, port: int, timeout: float) -> float:
        """TCP connect latency measurement"""
        try:
            start = time.monotonic()
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target, port),
                timeout=timeout
            )
            latency = (time.monotonic() - start) * 1000
            writer.close()
            await writer.wait_closed()
            return latency
        except Exception as e:
            logger.debug(f"TCP connect to {target}:{port} failed: {str(e)}")
            return timeout * 1000

    async def _measure_udp(self, target: str, port: int, timeout: float) -> float:
        """UDP latency measurement with empty payload"""
        try:
            start = time.monotonic()
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target, port, proto=socket.IPPROTO_UDP),
                timeout=timeout
            )
            writer.write(b"")
            await writer.drain()
            await reader.read(1)  # Wait for any response
            latency = (time.monotonic() - start) * 1000
            writer.close()
            return latency
        except Exception as e:
            logger.debug(f"UDP probe to {target}:{port} failed: {str(e)}")
            return timeout * 1000

# Integration with XIBRA's Alerting System
class LatencyAnomalyEvent:
    def __init__(self, monitor: LatencyMonitor):
        self.monitor = monitor
    
    async def trigger_actions(self):
        """Execute auto-remediation workflows"""
        anomalies = await self.monitor.detect_anomalies()
        for (target, protocol), is_anomalous in anomalies.items():
            if is_anomalous:
                logger.warning(f"Latency anomaly detected: {target} ({protocol})")
                # Integrate with XIBRA's routing table update
                # await self._update_routing_table(target)

# Unit Tests
import unittest
from unittest import IsolatedAsyncioTestCase

class TestLatencyMonitor(IsolatedAsyncioTestCase):
    async def test_basic_measurement(self):
        monitor = LatencyMonitor(protocols=["icmp"])
        await monitor.start()
        await asyncio.sleep(6)  # Wait for one monitoring cycle
        stats = await monitor.get_statistics("node1.xibra.net", "icmp")
        self.assertGreater(stats.mean, 0)
        await monitor.stop()

    async def test_anomaly_detection(self):
        monitor = LatencyMonitor(update_interval=0.1, history_size=10)
        await monitor.start()
        
        # Inject artificial latency spike
        for _ in range(10):
            await monitor.measure_latency("test", "icmp", 10.0)
        await monitor.measure_latency("test", "icmp", 500.0)
        
        anomalies = await monitor.detect_anomalies()
        self.assertTrue(anomalies.get(("test", "icmp"), False))
        await monitor.stop()

if __name__ == "__main__":
    unittest.main()
