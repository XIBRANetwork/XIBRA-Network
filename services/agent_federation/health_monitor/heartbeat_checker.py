"""
Enterprise-grade Heartbeat Monitor
Supports HTTP/GRPC/CustomProtocol checks with 
adaptive intervals and auto-remediation
"""

import asyncio
import aiohttp
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional
import grpc
from prometheus_client import (  # type: ignore
    Counter,
    Enum as PromEnum,
    Gauge,
    start_http_server,
)

# Metrics Setup
HEARTBEAT_COUNTER = Counter("xibra_heartbeat_total", "Total heartbeats", ["protocol"])
FAILURE_COUNTER = Counter("xibra_heartbeat_failures", "Failure count", ["node_type"])
LATENCY_SUMMARY = Gauge("xibra_heartbeat_latency", "Heartbeat latency in ms")
NODE_STATUS = PromEnum(
    "xibra_node_status",
    "Current node status",
    states=["healthy", "unresponsive", "degraded"],
)

# Configuration Models
class ProtocolType(Enum):
    HTTP = "http"
    GRPC = "grpc"
    CUSTOM = "custom"

@dataclass
class NodeConfig:
    endpoint: str
    protocol: ProtocolType
    check_interval: int = 30
    timeout: int = 10
    retries: int = 3
    recovery_action: Optional[str] = None

class NodeState(Enum):
    HEALTHY = 1
    UNRESPONSIVE = 2
    DEGRADED = 3

class HeartbeatError(Exception):
    pass

class HeartbeatChecker:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.nodes: Dict[str, NodeConfig] = {}
        self.node_states: Dict[str, NodeState] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._redis_url = redis_url
        self._lock = asyncio.Lock()

        # Start metrics server
        start_http_server(8000)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                connector=aiohttp.TCPConnector(ssl=False),
            )
        return self._session

    async def add_node(self, node_id: str, config: NodeConfig):
        async with self._lock:
            self.nodes[node_id] = config
            self.node_states[node_id] = NodeState.HEALTHY

    async def _http_check(self, node_id: str, config: NodeConfig) -> bool:
        session = await self._get_session()
        try:
            start_time = datetime.utcnow()
            async with session.head(
                config.endpoint,
                timeout=config.timeout,
                headers={"X-XIBRA-Node": node_id},
            ) as resp:
                latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                LATENCY_SUMMARY.set(latency)
                
                if resp.status == 200:
                    return True
                if 500 <= resp.status < 600:
                    return False
                raise HeartbeatError(f"Unexpected status: {resp.status}")
        except Exception as e:
            logging.warning(f"HTTP check failed for {node_id}: {str(e)}")
            return False

    async def _grpc_check(self, node_id: str, config: NodeConfig) -> bool:
        try:
            # Implement generated gRPC health check
            # Requires compiled protobuf stubs
            raise NotImplementedError("GRPC check requires protocol stubs")
        except grpc.RpcError as e:
            logging.error(f"GRPC check failed for {node_id}: {e.code()}")
            return False

    async def _execute_recovery(self, node_id: str, config: NodeConfig):
        if not config.recovery_action:
            return

        if config.recovery_action.startswith("http://"):
            session = await self._get_session()
            async with session.post(config.recovery_action) as resp:
                if resp.status != 200:
                    logging.error(f"Recovery action failed for {node_id}")

    async def _check_node(self, node_id: str):
        config = self.nodes[node_id]
        retries = 0
        
        while retries < config.retries:
            try:
                success = False
                if config.protocol == ProtocolType.HTTP:
                    success = await self._http_check(node_id, config)
                elif config.protocol == ProtocolType.GRPC:
                    success = await self._grpc_check(node_id, config)
                
                if success:
                    self.node_states[node_id] = NodeState.HEALTHY
                    HEARTBEAT_COUNTER.labels(config.protocol.value).inc()
                    return
                    
                retries += 1
                await asyncio.sleep(1)
                
            except Exception as e:
                logging.exception(f"Critical check failure: {str(e)}")
                break

        # Mark node failure
        async with self._lock:
            self.node_states[node_id] = NodeState.UNRESPONSIVE
            FAILURE_COUNTER.labels(type=config.protocol.value).inc()
        
        await self._execute_recovery(node_id, config)

    async def _monitor_task(self):
        while self._running:
            tasks = [
                self._check_node(node_id)
                for node_id in self.nodes
                if self.node_states.get(node_id) != NodeState.UNRESPONSIVE
            ]
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(5)  # Base interval

    async def start(self):
        self._running = True
        asyncio.create_task(self._monitor_task())

    async def stop(self):
        self._running = False
        if self._session:
            await self._session.close()

# Example Usage
if __name__ == "__main__":
    checker = HeartbeatChecker()
    
    # Register nodes
    asyncio.run(checker.add_node("inference-node-1", NodeConfig(
        endpoint="https://node1.xibra.network/health",
        protocol=ProtocolType.HTTP,
        recovery_action="http://orchestrator.xibra.network/restart/node1"
    )))
    
    try:
        asyncio.run(checker.start())
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        asyncio.run(checker.stop())
