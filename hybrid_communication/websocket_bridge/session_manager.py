"""
Enterprise Distributed Session Manager
Handles AI agent collaboration sessions with CAS operations,
automatic expiration, and cross-node synchronization
"""

import asyncio
import time
import uuid
import logging
from typing import Dict, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from contextlib import asynccontextmanager
import aioredis  # Requires redis-server >= 6.2 for ACL changes

logger = logging.getLogger("xibra.session")

class SessionException(Exception):
    """Base session management error"""
    pass

class SessionConflict(SessionException):
    """CAS operation failure"""
    pass

class SessionExpired(SessionException):
    """Session TTL exceeded"""
    pass

@dataclass(frozen=True)
class SessionConfig:
    default_ttl: int = 300  # Seconds
    max_renewals: int = 10
    lock_timeout: int = 30
    heartbeat_interval: int = 60

@dataclass
class SessionState:
    session_id: str
    participants: Dict[str, str]  # {agent_id: protocol_address}
    context: Dict[str, object]
    created_at: float
    modified_at: float
    ttl: int
    renewal_count: int = 0
    version: int = 0  # For CAS operations

class DistributedSessionManager:
    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis = aioredis.from_url(redis_url)
        self.config = SessionConfig()
        self._local_cache: Dict[str, SessionState] = {}
        self._locks = asyncio.Locks()
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}

    @asynccontextmanager
    async def session_scope(
        self, 
        initial_context: Optional[Dict] = None
    ) -> AsyncGenerator[SessionState, None]:
        """Context manager for atomic session operations"""
        session_id = str(uuid.uuid4())
        session = SessionState(
            session_id=session_id,
            participants={},
            context=initial_context or {},
            created_at=time.time(),
            modified_at=time.time(),
            ttl=self.config.default_ttl
        )
        
        try:
            await self._create_session(session)
            self._start_heartbeat(session_id)
            yield session
            await self._update_session(session, renew=True)
        finally:
            await self._cleanup_session(session_id)

    async def _create_session(self, session: SessionState):
        """Atomic session creation with CAS"""
        async with self.redis.pipeline(transaction=True) as pipe:
            await pipe.watch(session.session_id)
            if await pipe.exists(session.session_id):
                raise SessionConflict("Session ID collision")
                
            await pipe.multi()
            pipe.hset(session.session_id, mapping={
                "participants": self._serialize(session.participants),
                "context": self._serialize(session.context),
                "created_at": session.created_at,
                "version": session.version,
                "ttl": session.ttl
            })
            pipe.expire(session.session_id, session.ttl)
            await pipe.execute()

        self._local_cache[session.session_id] = session

    async def join_session(
        self, 
        session_id: str,
        agent_id: str,
        address: str,
        auth_token: str
    ) -> SessionState:
        """Join an existing session with CAS verification"""
        async with await self.redis.lock(
            f"lock:{session_id}", 
            timeout=self.config.lock_timeout
        ):
            raw_data = await self.redis.hgetall(session_id)
            if not raw_data:
                raise SessionExpired("Session not found")

            session = self._deserialize_session(session_id, raw_data)
            session.participants[agent_id] = address
            session.version += 1
            
            await self._update_session(session)
            return session

    async def _update_session(
        self, 
        session: SessionState,
        renew: bool = False
    ) -> None:
        """CAS session update with version check"""
        async with await self.redis.lock(
            f"lock:{session_id}", 
            timeout=self.config.lock_timeout
        ):
            current_version = int(
                await self.redis.hget(session.session_id, "version")
            )
            
            if session.version != current_version:
                raise SessionConflict("Version mismatch")
                
            updates = {
                "participants": self._serialize(session.participants),
                "context": self._serialize(session.context),
                "modified_at": time.time(),
                "version": session.version + 1,
                "ttl": session.ttl
            }
            
            if renew:
                updates["renewal_count"] = session.renewal_count + 1
                if updates["renewal_count"] > self.config.max_renewals:
                    raise SessionException("Max renewals exceeded")

            await self.redis.hset(session.session_id, mapping=updates)
            await self.redis.expire(session.session_id, session.ttl)

    async def get_session(self, session_id: str) -> SessionState:
        """Fetch session with local cache fallback"""
        if session := self._local_cache.get(session_id):
            if time.time() - session.modified_at < session.ttl:
                return session

        raw_data = await self.redis.hgetall(session_id)
        if not raw_data:
            raise SessionExpired("Session expired")
            
        return self._deserialize_session(session_id, raw_data)

    def _start_heartbeat(self, session_id: str):
        """Initialize periodic session renewal"""
        async def heartbeat_loop():
            while True:
                await asyncio.sleep(self.config.heartbeat_interval)
                try:
                    async with self.redis.pipeline() as pipe:
                        await pipe.expire(session_id, self.config.default_ttl)
                        await pipe.execute()
                except Exception as e:
                    logger.error(f"Heartbeat failed: {str(e)}")
                    break

        task = asyncio.create_task(heartbeat_loop())
        self._heartbeat_tasks[session_id] = task

    async def _cleanup_session(self, session_id: str):
        """Graceful session termination"""
        if task := self._heartbeat_tasks.pop(session_id, None):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await self.redis.delete(session_id)
        self._local_cache.pop(session_id, None)

    def _serialize(self, data: object) -> bytes:
        """Pluggable serialization (MessagePack/Protobuf)"""
        # Implementation would use XIBRA's core serializer
        return str(data).encode()

    def _deserialize_session(self, session_id: str, raw: dict) -> SessionState:
        """Rebuild session state from storage"""
        return SessionState(
            session_id=session_id,
            participants=self._deserialize(raw[b'participants']),
            context=self._deserialize(raw[b'context']),
            created_at=float(raw[b'created_at']),
            modified_at=float(raw[b'modified_at']),
            ttl=int(raw[b'ttl']),
            renewal_count=int(raw.get(b'renewal_count', 0)),
            version=int(raw[b'version'])
        )

    def _deserialize(self, data: bytes) -> object:
        """Pluggable deserialization"""
        return eval(data.decode())  # Replace with real deserializer

# Integration with XIBRA Security
class SessionAuthenticator:
    def __init__(self, manager: DistributedSessionManager):
        self.manager = manager
        self._jwt_secret = "xibra_secure_key"  # Use KMS in production

    async def validate_session(self, session_id: str, token: str) -> bool:
        """JWT-based session authentication"""
        # Implementation would validate token signature
        return token == "valid"  # Simplified for example

# Unit Tests
import unittest
from unittest import IsolatedAsyncioTestCase

class TestSessionManager(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = DistributedSessionManager()
        await self.manager.redis.flushdb()

    async def test_session_lifecycle(self):
        async with self.manager.session_scope() as session:
            await self.manager.join_session(
                session.session_id, 
                "agent1", 
                "grpc://node1",
                "token"
            )
            updated = await self.manager.get_session(session.session_id)
            self.assertIn("agent1", updated.participants)

    async def test_session_conflict(self):
        async with self.manager.session_scope() as s1:
            with self.assertRaises(SessionConflict):
                async with self.manager.session_scope(s1.session_id):
                    pass

if __name__ == "__main__":
    unittest.main()
