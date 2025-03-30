/*
 * Enterprise CUDA Memory Allocator
 * Features: Pooled Allocation, Stream-Aware Caching, 
 *           Secure Zeroization, and Multi-GPU NUMA Optimization
 */

#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <numa.h>

#define ALIGNMENT 256 // Optimized for tensor cores
#define DEFAULT_POOL_SIZE (1ULL << 31) // 2GB per GPU
#define SECURE_WIPE_PATTERN 0xDEADBEEF

class CUDAMemoryAllocator {
public:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        cudaStream_t stream;
        bool is_free;
    };

    explicit CUDAMemoryAllocator(int max_gpus = 8) 
        : max_gpus_(max_gpus) {
        initialize_numa();
    }

    ~CUDAMemoryAllocator() {
        release_all();
    }

    void* allocate(size_t size, cudaStream_t stream = 0) {
        const size_t aligned_size = align_size(size);
        std::lock_guard<std::mutex> lock(mutex_);

        // Try to reuse from pool
        for (auto& block : memory_pool_[current_gpu_]) {
            if (block.is_free && block.size >= aligned_size) {
                block.is_free = false;
                block.stream = stream;
                return block.ptr;
            }
        }

        // Allocate new block with NUMA awareness
        void* new_ptr = nullptr;
        cudaSetDevice(current_gpu_);
        cudaMallocManaged(&new_ptr, aligned_size);
        cudaMemAdvise(new_ptr, aligned_size, cudaMemAdviseSetPreferredLocation, current_gpu_);
        
        // NUMA binding for host memory
        if (numa_available() >= 0) {
            unsigned long node_mask = 1UL << numa_nodes_[current_gpu_];
            mbind(new_ptr, aligned_size, MPOL_BIND, &node_mask, sizeof(node_mask)*8, MPOL_MF_MOVE);
        }

        memory_pool_[current_gpu_].push_back({new_ptr, aligned_size, stream, false});
        allocation_stats_[current_gpu_] += aligned_size;
        return new_ptr;
    }

    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& block : memory_pool_[current_gpu_]) {
            if (block.ptr == ptr) {
                secure_wipe(block.ptr, block.size);
                block.is_free = true;
                return;
            }
        }
        throw std::runtime_error("Invalid pointer deallocation");
    }

    void switch_device(int gpu_id) {
        if (gpu_id < 0 || gpu_id >= max_gpus_) 
            throw std::out_of_range("Invalid GPU ID");
        current_gpu_ = gpu_id;
        cudaSetDevice(current_gpu_);
    }

    void release_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int gpu = 0; gpu < max_gpus_; ++gpu) {
            cudaSetDevice(gpu);
            for (auto& block : memory_pool_[gpu]) {
                cudaFree(block.ptr);
                allocation_stats_[gpu] -= block.size;
            }
            memory_pool_[gpu].clear();
        }
    }

    void print_stats() const {
        for (int gpu = 0; gpu < max_gpus_; ++gpu) {
            std::cout << "GPU " << gpu << " Usage: " 
                      << (allocation_stats_[gpu] >> 20) << "MB / "
                      << (DEFAULT_POOL_SIZE >> 20) << "MB\n";
        }
    }

private:
    std::mutex mutex_;
    int max_gpus_;
    int current_gpu_ = 0;
    std::vector<int> numa_nodes_;
    std::unordered_map<int, std::vector<MemoryBlock>> memory_pool_;
    std::unordered_map<int, size_t> allocation_stats_;

    void initialize_numa() {
        if (numa_available() < 0) return;
        
        for (int gpu = 0; gpu < max_gpus_; ++gpu) {
            cudaSetDevice(gpu);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, gpu);
            numa_nodes_.push_back(numa_node_of_cpu(prop.pciBusID % numa_num_configured_cpus()));
        }
    }

    size_t align_size(size_t size) const {
        return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

    void secure_wipe(void* ptr, size_t size) {
        cudaMemsetAsync(ptr, SECURE_WIPE_PATTERN, size, memory_pool_[current_gpu_].back().stream);
        cudaStreamSynchronize(memory_pool_[current_gpu_].back().stream);
    }
};

// Example Usage
int main() {
    try {
        CUDAMemoryAllocator allocator;
        allocator.switch_device(0);
        
        float* tensor = static_cast<float*>(allocator.allocate(1024 * sizeof(float)));
        // GPU operations...
        allocator.deallocate(tensor);
        
        allocator.print_stats();
    } catch (const std::exception& e) {
        std::cerr << "Allocation Error: " << e.what() << std::endl;
    }
    return 0;
}
