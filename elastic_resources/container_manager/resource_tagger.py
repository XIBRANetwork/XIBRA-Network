"""
Enterprise Resource Tagging Engine
Implements hierarchical resource classification with 
conflict resolution, audit trails, and cross-system sync
"""

from typing import Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import hashlib
import logging
import datetime
from dataclasses import dataclass, field
import uuid
import json
from abc import ABC, abstractmethod
import threading

# Configure enterprise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resource_tagger")

class TagConflictStrategy(Enum):
    PRIORITY_OVERRIDE = auto()
    MERGE_VALUES = auto()
    BLOCK_UPDATE = auto()
    CREATE_VERSION = auto()

class ResourceType(Enum):
    COMPUTE = auto()
    STORAGE = auto()
    NETWORK = auto()
    ML_MODEL = auto()
    DATASTREAM = auto()

@dataclass(frozen=True)
class TagScope:
    system: bool = False
    tenant: bool = False
    project: bool = False
    user: bool = False

@dataclass
class ResourceTag:
    key: str
    value: str
    priority: int = 100
    scope: TagScope = field(default_factory=lambda: TagScope(system=True))
    inherited_from: Optional[uuid.UUID] = None
    expires_at: Optional[datetime.datetime] = None
    version: int = 1
    metadata: Dict[str, str] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Generate cryptographic fingerprint for validation"""
        payload = f"{self.key}:{self.value}:{self.priority}:{self.version}"
        return hashlib.sha3_256(payload.encode()).hexdigest()

class TaggingPolicy(ABC):
    @abstractmethod
    def validate(self, resource_id: uuid.UUID, tag: ResourceTag) -> bool:
        pass

class ResourceTagger:
    def __init__(self, conflict_strategy: TagConflictStrategy = TagConflictStrategy.PRIORITY_OVERRIDE):
        self._tags: Dict[uuid.UUID, Dict[str, ResourceTag]] = {}
        self._inheritance_graph: Dict[uuid.UUID, Set[uuid.UUID]] = {}
        self._audit_log: List[Tuple[datetime.datetime, str, dict]] = []
        self._lock = threading.RLock()
        self.conflict_strategy = conflict_strategy
        self._policies: List[TaggingPolicy] = []

    def register_policy(self, policy: TaggingPolicy) -> None:
        """Add enterprise tagging policy validators"""
        with self._lock:
            self._policies.append(policy)

    def _check_policies(self, resource_id: uuid.UUID, tag: ResourceTag) -> bool:
        """Enforce enterprise governance policies"""
        with self._lock:
            return all(policy.validate(resource_id, tag) for policy in self._policies)

    def apply_tag(
        self,
        resource_id: uuid.UUID,
        tag: ResourceTag,
        source: Optional[uuid.UUID] = None
    ) -> bool:
        """Atomic tag application with conflict resolution"""
        with self._lock:
            # Policy enforcement
            if not self._check_policies(resource_id, tag):
                logger.error(f"Policy violation on resource {resource_id}")
                return False

            # Conflict resolution
            current_tags = self._tags.get(resource_id, {})
            if tag.key in current_tags:
                return self._resolve_conflict(resource_id, tag, current_tags[tag.key])
            
            # New tag application
            self._tags.setdefault(resource_id, {})[tag.key] = tag
            self._log_operation("CREATE", resource_id, tag, source)
            return True

    def _resolve_conflict(self, resource_id: uuid.UUID, new_tag: ResourceTag, existing_tag: ResourceTag) -> bool:
        """Enterprise-grade conflict resolution engine"""
        strategy = self.conflict_strategy
        logger.info(f"Resolving tag conflict on {resource_id} using {strategy.name}")

        with self._lock:
            if strategy == TagConflictStrategy.PRIORITY_OVERRIDE:
                if new_tag.priority > existing_tag.priority:
                    self._tags[resource_id][new_tag.key] = new_tag
                    self._log_operation("UPDATE", resource_id, new_tag)
                    return True
                return False

            elif strategy == TagConflictStrategy.MERGE_VALUES:
                merged_value = f"{existing_tag.value},{new_tag.value}"
                merged_tag = ResourceTag(
                    key=new_tag.key,
                    value=merged_value,
                    priority=max(new_tag.priority, existing_tag.priority),
                    scope=new_tag.scope,
                    version=existing_tag.version + 1
                )
                self._tags[resource_id][new_tag.key] = merged_tag
                self._log_operation("MERGE", resource_id, merged_tag)
                return True

            elif strategy == TagConflictStrategy.CREATE_VERSION:
                versioned_key = f"{new_tag.key}_v{existing_tag.version + 1}"
                versioned_tag = ResourceTag(
                    key=versioned_key,
                    value=new_tag.value,
                    priority=new_tag.priority,
                    scope=new_tag.scope,
                    version=existing_tag.version + 1
                )
                self._tags[resource_id][versioned_key] = versioned_tag
                self._log_operation("VERSION", resource_id, versioned_tag)
                return True

            return False

    def get_tags(self, resource_id: uuid.UUID, include_inherited: bool = True) -> Dict[str, ResourceTag]:
        """Retrieve tags with inheritance resolution"""
        with self._lock:
            direct_tags = self._tags.get(resource_id, {}).copy()
            
            if include_inherited:
                for parent in self._inheritance_graph.get(resource_id, set()):
                    parent_tags = self.get_tags(parent)
                    for key, tag in parent_tags.items():
                        if key not in direct_tags or tag.priority > direct_tags[key].priority:
                            inherited_tag = ResourceTag(
                                key=tag.key,
                                value=tag.value,
                                priority=tag.priority,
                                scope=tag.scope,
                                inherited_from=parent,
                                version=tag.version
                            )
                            direct_tags[key] = inherited_tag
            
            return direct_tags

    def export_tags(self, resource_id: uuid.UUID) -> str:
        """Generate standardized tag package for cross-system sync"""
        tags = self.get_tags(resource_id)
        return json.dumps({
            "resource_id": str(resource_id),
            "tags": {
                key: {
                    "value": tag.value,
                    "scope": {f.name: f for f in tag.scope},
                    "version": tag.version,
                    "fingerprint": tag.fingerprint()
                }
                for key, tag in tags.items()
            }
        }, indent=2)

    def _log_operation(self, action: str, resource_id: uuid.UUID, tag: ResourceTag, source: Optional[uuid.UUID] = None) -> None:
        """Immutable audit logging with cryptographic hashes"""
        log_entry = (
            datetime.datetime.utcnow(),
            action,
            {
                "resource_id": str(resource_id),
                "tag_key": tag.key,
                "tag_value": tag.value,
                "source": str(source) if source else "SYSTEM",
                "fingerprint": tag.fingerprint()
            }
        )
        self._audit_log.append(log_entry)
        logger.info(f"Tag {action} logged: {log_entry}")

# Example Enterprise Policies
class CostCenterPolicy(TaggingPolicy):
    def validate(self, resource_id: uuid.UUID, tag: ResourceTag) -> bool:
        if tag.key == "cost_center":
            return tag.value.startswith("CC-") and len(tag.value) == 10
        return True

class DataClassificationPolicy(TaggingPolicy):
    def validate(self, resource_id: uuid.UUID, tag: ResourceTag) -> bool:
        if tag.key == "data_class":
            allowed = ["PUBLIC", "CONFIDENTIAL", "SECRET"]
            return tag.value in allowed
        return True

# Enterprise Usage Example
if __name__ == "__main__":
    tagger = ResourceTagger(conflict_strategy=TagConflictStrategy.PRIORITY_OVERRIDE)
    tagger.register_policy(CostCenterPolicy())
    tagger.register_policy(DataClassificationPolicy())

    # Create sample resources
    vm_id = uuid.uuid4()
    storage_id = uuid.uuid4()
    
    # Apply tags with inheritance
    tagger.apply_tag(vm_id, ResourceTag(
        key="environment",
        value="production",
        priority=200,
        scope=TagScope(system=True)
    ))
    
    tagger.apply_tag(storage_id, ResourceTag(
        key="environment",
        value="development",
        priority=100,
        scope=TagScope(tenant=True)
    ))
    
    # Establish inheritance
    tagger._inheritance_graph[storage_id] = {vm_id}
    
    # Attempt policy violation
    tagger.apply_tag(storage_id, ResourceTag(
        key="cost_center",
        value="INVALID_CODE",
        priority=300
    ))  # This will be blocked
    
    # Export tags
    print(tagger.export_tags(storage_id))
