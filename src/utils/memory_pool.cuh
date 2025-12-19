#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <cstddef>
#include "check_error.cuh"

/**
 * CUDA Memory Pool Allocator
 * Reduces allocation overhead and fragmentation by reusing memory blocks
 */
class CudaMemoryPool {
public:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
        
        Block(void* p, size_t s) : ptr(p), size(s), in_use(false) {}
    };
    
    static CudaMemoryPool& getInstance() {
        static CudaMemoryPool instance;
        return instance;
    }
    
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Round up to alignment boundary (128 bytes for better coalescing)
        size = alignSize(size, 128);
        
        // Try to find a free block of appropriate size
        auto it = free_blocks_.find(size);
        if (it != free_blocks_.end() && !it->second.empty()) {
            Block* block = it->second.back();
            it->second.pop_back();
            block->in_use = true;
            return block->ptr;
        }
        
        // Allocate new block
        void* ptr = nullptr;
        CUDA_OK(cudaMalloc(&ptr, size));
        
        Block* block = new Block(ptr, size);
        blocks_[ptr] = block;
        block->in_use = true;
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = blocks_.find(ptr);
        if (it != blocks_.end()) {
            Block* block = it->second;
            block->in_use = false;
            
            // Add to free list for reuse
            free_blocks_[block->size].push_back(block);
        }
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& pair : blocks_) {
            CUDA_OK(cudaFree(pair.second->ptr));
            delete pair.second;
        }
        
        blocks_.clear();
        free_blocks_.clear();
    }
    
    ~CudaMemoryPool() {
        clear();
    }
    
private:
    CudaMemoryPool() = default;
    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;
    
    size_t alignSize(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
    std::mutex mutex_;
    std::unordered_map<void*, Block*> blocks_;
    std::unordered_map<size_t, std::vector<Block*>> free_blocks_;
};

// RAII wrapper for pool-allocated memory
class PooledMemory {
public:
    PooledMemory(size_t size) : ptr_(CudaMemoryPool::getInstance().allocate(size)), size_(size) {}
    
    ~PooledMemory() {
        if (ptr_) {
            CudaMemoryPool::getInstance().deallocate(ptr_);
        }
    }
    
    void* get() { return ptr_; }
    size_t size() const { return size_; }
    
    // Non-copyable, movable
    PooledMemory(const PooledMemory&) = delete;
    PooledMemory& operator=(const PooledMemory&) = delete;
    PooledMemory(PooledMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
    }
    
private:
    void* ptr_;
    size_t size_;
};

