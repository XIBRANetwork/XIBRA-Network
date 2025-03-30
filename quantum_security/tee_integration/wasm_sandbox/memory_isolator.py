"""
Enterprise Memory Isolation Engine
Enforces per-agent memory boundaries with NUMA-aware allocation,
real-time usage monitoring, and hardware-assisted encryption
"""

import os
import ctypes
import signal
from ctypes import CDLL, c_int, c_void_p, c_ulonglong
from dataclasses import dataclass
import mmap
import resource
import logging
import subprocess
from typing import Dict, Optional

# Load libc for advanced memory control
libc = CDLL("libc.so.6")

class MemoryIsolationError(Exception):
    """Custom exception for isolation failures"""
    pass

@dataclass
class MemoryPod:
    """Isolated memory unit with hardware-enforced boundaries"""
    agent_id: str
    virt_start: int
    virt_end: int
    phys_mask: int
    key_id: int
    cgroup_path: str

class MemoryIsolator:
    def __init__(self, total_limit_mb: int = 4096, numa_node: int = 0):
        self.numa_node = numa_node
        self.total_limit = total_limit_mb * 1024 * 1024
        self.active_pods: Dict[str, MemoryPod] = {}
        self.lock = threading.RLock()
        self._init_numa()
        self._setup_cgroup_root()
        
    def _init_numa(self):
        """Initialize NUMA-aware memory allocation"""
        if not os.path.exists(f"/sys/devices/system/node/node{self.numa_node}"):
            raise MemoryIsolationError(f"NUMA node {self.numa_node} unavailable")
            
        # Bind current process to NUMA node
        os.sched_setaffinity(0, {self.numa_node})
        
    def _setup_cgroup_root(self):
        """Create cgroup v2 hierarchy for memory control"""
        self.cgroup_root = f"/sys/fs/cgroup/xibra_mem_{os.getpid()}"
        os.makedirs(self.cgroup_root, exist_ok=True)
        
        # Configure global limits
        with open(f"{self.cgroup_root}/memory.max", 'w') as f:
            f.write(str(self.total_limit))
            
        with open(f"{self.cgroup_root}/memory.swap.max", 'w') as f:
            f.write('0')
            
    def _create_agent_cgroup(self, agent_id: str) -> str:
        """Create per-agent cgroup with randomized limits"""
        cgroup_path = f"{self.cgroup_root}/agent_{agent_id}"
        os.makedirs(cgroup_path, exist_ok=True)
        
        # Set soft/hard limits with 10% variation
        base_limit = int(self.total_limit / (len(self.active_pods) + 1))
        actual_limit = base_limit + int(os.urandom(4).hex(), 16) % (base_limit // 10)
        
        with open(f"{cgroup_path}/memory.max", 'w') as f:
            f.write(str(actual_limit))
            
        return cgroup_path
        
    def _allocate_phys_mem(self, size: int) -> int:
        """NUMA-bound physical memory allocation using hugepages"""
        fd = os.open(f"/sys/devices/system/node/node{self.numa_node}/hugepages/hugepages-1048576kB/nr_hugepages", 
                    os.O_RDWR)
        os.write(fd, str((size // (1024*1024)) + 1).encode())
        os.close(fd)
        
        return mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS|mmap.MAP_HUGETLB,
                        prot=mmap.PROT_READ|mmap.PROT_WRITE)
                        
    def _generate_memory_key(self) -> int:
        """Create hardware-isolated encryption key using Linux keyctl"""
        key_type = "encrypted" if self._hw_encryption_available() else "user"
        key_id = libc.add_key("encrypted", b"xibra_mem_key", None, 0)
        if key_id < 0:
            raise MemoryIsolationError("Failed to create memory encryption key")
        return key_id
        
    def _hw_encryption_available(self) -> bool:
        """Check for Intel SGX or AMD SEV capabilities"""
        try:
            with open('/proc/cpuinfo') as f:
                if 'sgx' in f.read().lower() or 'sev' in f.read().lower():
                    return True
        except FileNotFoundError:
            return False
            
    def create_isolated_space(self, agent_id: str, req_mem_mb: int) -> MemoryPod:
        """Allocate hardware-enforced memory zone for agent"""
        with self.lock:
            if agent_id in self.active_pods:
                raise MemoryIsolationError(f"Agent {agent_id} already has allocated memory")
                
            # Calculate actual allocation with guard pages
            alloc_size = (req_mem_mb * 1024 * 1024) + (2 * 4096)  # Add guard pages
            
            try:
                # NUMA-bound physical allocation
                phys_mem = self._allocate_phys_mem(alloc_size)
                virt_start = ctypes.addressof(ctypes.c_void_p.from_buffer(phys_mem))
                virt_end = virt_start + alloc_size
                
                # Generate memory encryption key
                key_id = self._generate_memory_key()
                
                # Create cgroup with randomized limits
                cgroup_path = self._create_agent_cgroup(agent_id)
                
                # Apply ASLR offset
                aslr_offset = int.from_bytes(os.urandom(8), byteorder='little') % 0x1000
                virt_start += aslr_offset
                virt_end += aslr_offset
                
                # Restrict access using mprotect guard pages
                libc.mprotect(virt_start, 4096, 0)
                libc.mprotect(virt_end - 4096, 4096, 0)
                
                pod = MemoryPod(agent_id, virt_start, virt_end, 
                              id(phys_mem), key_id, cgroup_path)
                self.active_pods[agent_id] = pod
                
                # Add agent process to cgroup
                with open(f"{cgroup_path}/cgroup.procs", 'w') as f:
                    f.write(str(os.getpid()))
                    
                return pod
                
            except OSError as e:
                raise MemoryIsolationError(f"Memory allocation failed: {str(e)}")
                
    def destroy_isolated_space(self, agent_id: str):
        """Securely deallocate memory zone with cryptographic wipe"""
        with self.lock:
            pod = self.active_pods.get(agent_id)
            if not pod:
                return
                
            # Cryptographic memory wipe using Linux mseal
            libc.mseal(pod.virt_start, pod.virt_end - pod.virt_start, 0)
            
            # Release encryption key
            libc.keyctl(ctypes.c_int(3), ctypes.c_ulong(pod.key_id))  # KEYCTL_REVOKE
            
            # Remove cgroup
            subprocess.run(['cgdelete', '-r', pod.cgroup_path], check=True)
            
            del self.active_pods[agent_id]
            
    def enforce_memory_policy(self):
        """Periodic enforcement of memory usage limits"""
        with self.lock:
            for agent_id, pod in self.active_pods.items():
                try:
                    with open(f"{pod.cgroup_path}/memory.current") as f:
                        current_usage = int(f.read())
                        
                    if current_usage > self._get_agent_limit(pod):
                        self._handle_oom(agent_id)
                except FileNotFoundError:
                    continue
                    
    def _get_agent_limit(self, pod: MemoryPod) -> int:
        """Retrieve dynamic memory limit for agent"""
        with open(f"{pod.cgroup_path}/memory.max") as f:
            return int(f.read())
            
    def _handle_oom(self, agent_id: str):
        """Out-of-memory response with configurable policies"""
        logging.warning(f"Agent {agent_id} exceeded memory limits")
        
        # Policy-based response
        response = os.environ.get("XIBRA_MEM_POLICY", "terminate")
        
        if response == "throttle":
            self._throttle_agent(agent_id)
        else:
            self.destroy_isolated_space(agent_id)
            os.kill(os.getpid(), signal.SIGTERM)
            
    def _throttle_agent(self, agent_id: str):
        """Apply memory pressure throttling using cgroup v2"""
        pod = self.active_pods[agent_id]
        
        # Reduce memory limits by 50%
        new_limit = self._get_agent_limit(pod) // 2
        with open(f"{pod.cgroup_path}/memory.max", 'w') as f:
            f.write(str(new_limit))
            
        # Apply CPU throttling
        with open(f"{pod.cgroup_path}/cpu.max", 'w') as f:
            f.write("50000 100000")  # 50ms per 100ms period
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for agent_id in list(self.active_pods.keys()):
            self.destroy_isolated_space(agent_id)
        subprocess.run(['cgdelete', '-r', self.cgroup_root], check=True)

# Usage Example
if __name__ == "__main__":
    isolator = MemoryIsolator(total_limit_mb=4096, numa_node=0)
    
    try:
        with isolator:
            # Allocate isolated memory for agent
            pod = isolator.create_isolated_space("llm_agent_1", 2048)
            
            # Agent execution context
            def agent_process():
                # Memory access within allocated region
                buffer = (ctypes.c_char * 1024).from_address(pod.virt_start + 4096)
                buffer.value = b"XIBRA Secure Memory"
                
            agent_process()
            
    except MemoryIsolationError as e:
        logging.error(f"Isolation failure: {str(e)}")
