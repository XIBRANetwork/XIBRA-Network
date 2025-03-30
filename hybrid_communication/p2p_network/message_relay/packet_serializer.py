"""
Advanced Packet Serialization Module
Supports JSON, MessagePack, Protobuf with ZLIB/LZ4 compression
and ED25519 cryptographic signatures
"""

from __future__ import annotations
import json
import logging
import zlib
import lz4.frame
import msgpack
from typing import Any, Dict, Optional, Tuple, Protocol
from dataclasses import dataclass
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature
from google.protobuf import message as pb_message
import asyncio

logger = logging.getLogger("xibra.p2p.serializer")

class SerializationError(Exception):
    """Base exception for serialization failures"""
    pass

class InvalidSignatureError(SerializationError):
    """Raised when packet signature verification fails"""
    pass

class ISerializationStrategy(Protocol):
    def serialize(self, data: Any) -> bytes:
        ...
    
    def deserialize(self, raw: bytes) -> Any:
        ...

@dataclass(frozen=True)
class SerializedPacket:
    payload: bytes
    compression: Optional[str] = None
    signature: Optional[bytes] = None
    version: str = "1.0"

class PacketSerializer:
    def __init__(
        self,
        strategy: ISerializationStrategy,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
        public_key: Optional[ed25519.Ed25519PublicKey] = None
    ):
        self.strategy = strategy
        self.private_key = private_key
        self.public_key = public_key
        self._lock = asyncio.Lock()

    async def serialize(
        self,
        data: Any,
        compress: Optional[str] = None,
        sign: bool = False
    ) -> SerializedPacket:
        """Serialize data with optional compression and signing"""
        try:
            # Core serialization
            payload = self.strategy.serialize(data)
            
            # Compression
            if compress:
                payload = await self._compress(payload, compress)
            
            # Cryptographic signing
            signature = await self._sign(payload) if sign else None
            
            return SerializedPacket(
                payload=payload,
                compression=compress,
                signature=signature,
                version="1.0"
            )
        except Exception as e:
            logger.error(f"Serialization failed: {str(e)}")
            raise SerializationError(f"Serialization error: {str(e)}") from e

    async def deserialize(
        self,
        packet: SerializedPacket,
        verify: bool = False
    ) -> Any:
        """Deserialize packet with optional decompression and verification"""
        try:
            # Verify signature
            if verify:
                if not self.public_key:
                    raise InvalidSignatureError("No public key for verification")
                if not packet.signature:
                    raise InvalidSignatureError("Missing packet signature")
                
                self.public_key.verify(
                    packet.signature,
                    packet.payload
                )

            # Decompression
            payload = packet.payload
            if packet.compression:
                payload = await self._decompress(payload, packet.compression)

            # Core deserialization
            return self.strategy.deserialize(payload)
        except InvalidSignature as e:
            logger.warning("Signature verification failed")
            raise InvalidSignatureError("Invalid packet signature") from e
        except Exception as e:
            logger.error(f"Deserialization failed: {str(e)}")
            raise SerializationError(f"Deserialization error: {str(e)}") from e

    async def _compress(self, data: bytes, method: str) -> bytes:
        """Apply compression algorithm"""
        if method == "zlib":
            return zlib.compress(data, level=9)
        elif method == "lz4":
            return lz4.frame.compress(data)
        else:
            raise SerializationError(f"Unsupported compression: {method}")

    async def _decompress(self, data: bytes, method: str) -> bytes:
        """Reverse compression"""
        if method == "zlib":
            return zlib.decompress(data)
        elif method == "lz4":
            return lz4.frame.decompress(data)
        else:
            raise SerializationError(f"Unsupported decompression: {method}")

    async def _sign(self, data: bytes) -> bytes:
        """Generate ED25519 signature"""
        if not self.private_key:
            raise SerializationError("No private key available for signing")
        
        async with self._lock:
            return self.private_key.sign(data)

class JSONStrategy:
    def serialize(self, data: Any) -> bytes:
        return json.dumps(data).encode('utf-8')
    
    def deserialize(self, raw: bytes) -> Any:
        return json.loads(raw.decode('utf-8'))

class MessagePackStrategy:
    def serialize(self, data: Any) -> bytes:
        return msgpack.packb(data)
    
    def deserialize(self, raw: bytes) -> Any:
        return msgpack.unpackb(raw)

class ProtobufStrategy:
    def __init__(self, message_class: pb_message.Message):
        self.message_class = message_class
    
    def serialize(self, data: pb_message.Message) -> bytes:
        if not isinstance(data, self.message_class):
            raise SerializationError("Invalid protobuf message type")
        return data.SerializeToString()
    
    def deserialize(self, raw: bytes) -> pb_message.Message:
        message = self.message_class()
        message.ParseFromString(raw)
        return message

class SerializerFactory:
    @staticmethod
    def create_serializer(
        format: str = "msgpack",
        private_key: Optional[bytes] = None,
        public_key: Optional[bytes] = None,
        **kwargs
    ) -> PacketSerializer:
        strategies = {
            "json": JSONStrategy(),
            "msgpack": MessagePackStrategy(),
            "protobuf": ProtobufStrategy(kwargs.get('message_class'))
        }
        
        # Load keys if provided
        priv_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key) if private_key else None
        pub_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key) if public_key else None
        
        return PacketSerializer(
            strategy=strategies[format],
            private_key=priv_key,
            public_key=pub_key
        )

# Unit Tests
import unittest
from unittest import IsolatedAsyncioTestCase

class TestPacketSerializer(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Generate test keys
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        self.serializer = SerializerFactory.create_serializer(
            "msgpack",
            private_key=self.private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=None
            ),
            public_key=self.public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

    async def test_roundtrip(self):
        data = {"test": [1, 2.5, "value"]}
        packet = await self.serializer.serialize(data, compress="lz4", sign=True)
        result = await self.serializer.deserialize(packet, verify=True)
        self.assertEqual(data, result)

    async def test_signature_verification(self):
        data = {"secure": True}
        packet = await self.serializer.serialize(data, sign=True)
        
        # Tamper with payload
        tampered = SerializedPacket(
            payload=packet.payload + b"x",
            signature=packet.signature,
            version=packet.version
        )
        
        with self.assertRaises(InvalidSignatureError):
            await self.serializer.deserialize(tampered, verify=True)

    async def test_compression(self):
        data = {"large": [0] * 1000}
        packet = await self.serializer.serialize(data, compress="zlib")
        self.assertLess(len(packet.payload), len(json.dumps(data).encode()))

if __name__ == "__main__":
    unittest.main()
