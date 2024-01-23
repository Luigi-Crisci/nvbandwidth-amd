/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <hip/hip_runtime.h>
#include "memcpy.h"
#include "kernels.cuh"
#include "hip/hip_vector_types.h"

#define WARMUP_COUNT 4

MemcpyNode::MemcpyNode(size_t bufferSize): bufferSize(bufferSize), buffer(nullptr) {}

hipDeviceptr_t MemcpyNode::getBuffer() const {
    return (hipDeviceptr_t)buffer;
}

size_t MemcpyNode::getBufferSize() const {
    return bufferSize;
}
void MemcpyNode::memsetPattern(hipDeviceptr_t buffer, unsigned long long size, unsigned int seed) {
    unsigned int* pattern;
    unsigned int n = 0;
    void * _buffer = (void*) buffer;
    unsigned long long _2MBchunkCount = size / (1024 * 1024 * 2);
    unsigned long long remaining = size - (_2MBchunkCount * 1024 * 1024 * 2);

    // Allocate 2MB of pattern
    CU_ASSERT(hipHostAlloc((void**)&pattern, sizeof(char) * 1024 * 1024 * 2, hipHostMallocPortable));
    xorshift2MBPattern(pattern, seed);

    for (n = 0; n < _2MBchunkCount; n++) {
        CU_ASSERT(cuMemcpy((hipDeviceptr_t)_buffer, (hipDeviceptr_t)pattern, 1024 * 1024 * 2));
        _buffer = (char*)_buffer + (1024 * 1024 * 2);
    }
    if (remaining) {
        CU_ASSERT(cuMemcpy((hipDeviceptr_t)_buffer, (hipDeviceptr_t)pattern, (size_t)remaining));
    }

    CU_ASSERT(hipCtxSynchronize());
    CU_ASSERT(hipHostFree((void*)pattern));
}

void MemcpyNode::memcmpPattern(hipDeviceptr_t buffer, unsigned long long size, unsigned int seed) {
     unsigned int* devicePattern;
    unsigned int* pattern;
    unsigned long long _2MBchunkCount = size / (1024 * 1024 * 2);
    unsigned long long remaining = size - (_2MBchunkCount * 1024 * 1024 * 2);
    unsigned int n = 0;
    unsigned int x = 0;
    void * _buffer = (void*) buffer;

    // Allocate 2MB of pattern
    CU_ASSERT(hipHostAlloc((void**)&devicePattern, sizeof(char) * 1024 * 1024 * 2, hipHostMallocPortable));
    pattern = (unsigned int*)malloc(sizeof(char) * 1024 * 1024 * 2);
    xorshift2MBPattern(pattern, seed);

    for (n = 0; n < _2MBchunkCount; n++) {
        CU_ASSERT(cuMemcpy((hipDeviceptr_t)devicePattern, (hipDeviceptr_t)_buffer, 1024 * 1024 * 2));
        CU_ASSERT(hipCtxSynchronize());
        if (memcmp(pattern, devicePattern, 1024 * 1024 * 2) != 0) {
            for (x = 0; x < (1024 * 1024 * 2) / sizeof(unsigned int); x++) {
                if (devicePattern[x] != pattern[x]) {
                    std::cout << " Invalid value when checking the pattern at <" << (void*)((char*)_buffer + n * (1024 * 1024 * 2) + x * sizeof(unsigned int)) << ">" << std::endl
                                                        << " Current offset [ " << (unsigned long long)((char*)_buffer - (char*)buffer) + (unsigned long long)(x * sizeof(unsigned int)) << "/" << (size) << "]" << std::endl;
                    std::abort();
                }
            
            }
        }

        _buffer = (char*)_buffer + (1024 * 1024 * 2);
    }
    if (remaining) {
        CU_ASSERT(cuMemcpy((hipDeviceptr_t)devicePattern, (hipDeviceptr_t)_buffer, (size_t)remaining));
        if (memcmp(pattern, devicePattern, (size_t)remaining) != 0) {
            for (x = 0; x < remaining / sizeof(unsigned int); x++) {
                if (devicePattern[x] != pattern[x]) {
                    std::cout << " Invalid value when checking the pattern at <" << (void*)((char*)buffer + n * (1024 * 1024 * 2) + x * sizeof(unsigned int)) << ">" << std::endl
                                                        << " Current offset [ " << (unsigned long long)((char*)_buffer - (char*)buffer) + (unsigned long long)(x * sizeof(unsigned int)) << "/" << (size) << "]" << std::endl;
                    std::abort();
                }
            }
        }
    }

    CU_ASSERT(hipCtxSynchronize());
    CU_ASSERT(hipHostFree((void*)devicePattern));
    free(pattern);
}
void MemcpyNode::xorshift2MBPattern(unsigned int* buffer, unsigned int seed)
{
    unsigned int oldValue = seed;
    unsigned int n = 0;
    for (n = 0; n < (1024 * 1024 * 2) / sizeof(unsigned int); n++) {
        unsigned int value = oldValue;
        value = value ^ (value << 13);
        value = value ^ (value >> 17);
        value = value ^ (value << 5);
        oldValue = value;
        buffer[n] = oldValue;
    }
}
HostNode::HostNode(size_t bufferSize, int targetDeviceId): MemcpyNode(bufferSize) {
    hipCtx_t targetCtx;

    // Before allocating host memory, set correct NUMA affinity
    setOptimalCpuAffinity(targetDeviceId);
    CU_ASSERT(hipDevicePrimaryCtxRetain(&targetCtx, targetDeviceId));
    CU_ASSERT(hipCtxSetCurrent(targetCtx));

    CU_ASSERT(hipHostAlloc(&buffer, bufferSize, hipHostMallocPortable));
}

HostNode::~HostNode() {
    if (isMemoryOwnedByCUDA(buffer)) {
        CU_ASSERT(hipHostFree(buffer));
    } else {
        free(buffer);
    }
}

// Host nodes don't have a context, return null
hipCtx_t HostNode::getPrimaryCtx() const {
    return nullptr;
}

// Host Nodes always return zero as they always represent one row in the bandwidth matrix
int HostNode::getNodeIdx() const {
    return 0;
}

std::string HostNode::getNodeString() const {
    return "Host";
}

DeviceNode::DeviceNode(size_t bufferSize, int deviceIdx): deviceIdx(deviceIdx), MemcpyNode(bufferSize) {
    CU_ASSERT(hipDevicePrimaryCtxRetain(&primaryCtx, deviceIdx));
    CU_ASSERT(hipCtxSetCurrent(primaryCtx));
    CU_ASSERT(hipMalloc((hipDeviceptr_t*)&buffer, bufferSize));
}

DeviceNode::~DeviceNode() {
    CU_ASSERT(hipCtxSetCurrent(primaryCtx));
    CU_ASSERT(hipFree((hipDeviceptr_t)buffer));
    CU_ASSERT(hipDevicePrimaryCtxRelease(deviceIdx));
}

hipCtx_t DeviceNode::getPrimaryCtx() const {
    return primaryCtx;
}

int DeviceNode::getNodeIdx() const {
    return deviceIdx;
}

std::string DeviceNode::getNodeString() const {
    return "Device " + std::to_string(deviceIdx);
}

bool DeviceNode::enablePeerAcess(const DeviceNode &peerNode) {
    int canAccessPeer = 0;
    CU_ASSERT(hipDeviceCanAccessPeer(&canAccessPeer, getNodeIdx(), peerNode.getNodeIdx()));
    if (canAccessPeer) {
        hipError_t res;
        CU_ASSERT(hipCtxSetCurrent(peerNode.getPrimaryCtx()));
        res = hipCtxEnablePeerAccess(getPrimaryCtx(), 0);
        if (res != hipErrorPeerAccessAlreadyEnabled)
            CU_ASSERT(res);

        CU_ASSERT(hipCtxSetCurrent(getPrimaryCtx()));
        res = hipCtxEnablePeerAccess(peerNode.getPrimaryCtx(), 0);
        if (res != hipErrorPeerAccessAlreadyEnabled)
            CU_ASSERT(res);

        return true;
    }
    return false;
}

MemcpyOperation::MemcpyOperation(unsigned long long loopCount, ContextPreference ctxPreference, BandwidthValue bandwidthValue) : 
        loopCount(loopCount), ctxPreference(ctxPreference), bandwidthValue(bandwidthValue)
{
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    PROC_MASK_SET(procMask, getFirstEnabledCPU());
}

MemcpyOperation::~MemcpyOperation() {
    PROC_MASK_CLEAR(procMask, 0);
}

double MemcpyOperation::doMemcpy(const MemcpyNode &srcNode, const MemcpyNode &dstNode) {
    std::vector<const MemcpyNode*> srcNodes = {&srcNode};
    std::vector<const MemcpyNode*> dstNodes = {&dstNode};
    return doMemcpy(srcNodes, dstNodes);
}

double MemcpyOperation::doMemcpy(const std::vector<const MemcpyNode*> &srcNodes, const std::vector<const MemcpyNode*> &dstNodes) {
    volatile int* blockingVar;
    std::vector<hipCtx_t> contexts(srcNodes.size());
    std::vector<hipStream_t> streams(srcNodes.size());
    std::vector<hipEvent_t> startEvents(srcNodes.size());
    std::vector<hipEvent_t> endEvents(srcNodes.size());
    std::vector<PerformanceStatistic> bandwidthStats(srcNodes.size());
    std::vector<size_t> adjustedCopySizes(srcNodes.size());
    PerformanceStatistic totalBandwidth;
    hipEvent_t totalEnd;
    std::vector<size_t> finalCopySize(srcNodes.size());

    CU_ASSERT(hipHostAlloc((void **)&blockingVar, sizeof(*blockingVar), hipHostMallocPortable));

    for (int i = 0; i < srcNodes.size(); i++) {
        // prefer source context
        if (ctxPreference == MemcpyOperation::PREFER_SRC_CONTEXT && srcNodes[i]->getPrimaryCtx() != nullptr) {
            CU_ASSERT(hipCtxSetCurrent(srcNodes[i]->getPrimaryCtx()));
            contexts[i] = srcNodes[i]->getPrimaryCtx();
        } else if (dstNodes[i]->getPrimaryCtx() != nullptr) {
            CU_ASSERT(hipCtxSetCurrent(dstNodes[i]->getPrimaryCtx()));
            contexts[i] = dstNodes[i]->getPrimaryCtx();
        }

        // allocate the per simulaneous copy resources
        CU_ASSERT(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));
        CU_ASSERT(hipEventCreateWithFlags(&startEvents[i], hipEventDefault));
        CU_ASSERT(hipEventCreateWithFlags(&endEvents[i], hipEventDefault));
        // Get the final copy size that will be used.
        // CE and SM copy sizes will differ due to possible truncation
        // during SM copies.
        finalCopySize[i] = getAdjustedCopySize(srcNodes[0]->getBufferSize(), streams[i]);
    }   
    CU_ASSERT(hipCtxSetCurrent(contexts[0]));
    CU_ASSERT(hipEventCreateWithFlags(&totalEnd, hipEventDefault));

    // This loop is for sampling the testcase (which itself has a loop count)
    for (unsigned int n = 0; n < averageLoopCount; n++) {
        *blockingVar = 0;
        // Set the memory patterns correctly before spin kernel launch etc.
        for (int i = 0; i < srcNodes.size(); i++) {
            dstNodes[i]->memsetPattern(dstNodes[i]->getBuffer(), finalCopySize[i], 0xCAFEBABE);
            srcNodes[i]->memsetPattern(srcNodes[i]->getBuffer(), finalCopySize[i], 0xBAADF00D);
        }        
        // block stream, and enqueue copy
        for (int i = 0; i < srcNodes.size(); i++) {
            CU_ASSERT(hipCtxSetCurrent(contexts[i]));

            // start the spin kernel on the stream
            CU_ASSERT(spinKernel(blockingVar, streams[i]));

            // warmup
            memcpyFunc(dstNodes[i]->getBuffer(), srcNodes[i]->getBuffer(), streams[i], srcNodes[i]->getBufferSize(), WARMUP_COUNT);
        }

        CU_ASSERT(hipCtxSetCurrent(contexts[0]));
        CU_ASSERT(hipEventRecord(startEvents[0], streams[0]));
        for (int i = 1; i < srcNodes.size(); i++) {
            // ensure that all copies are launched at the same time
            CU_ASSERT(hipCtxSetCurrent(contexts[i]));
            CU_ASSERT(hipStreamWaitEvent(streams[i], startEvents[0], 0));
            CU_ASSERT(hipEventRecord(startEvents[i], streams[i]));
        }

        for (int i = 0; i < srcNodes.size(); i++) {
            CU_ASSERT(hipCtxSetCurrent(contexts[i]));
            assert(srcNodes[i]->getBufferSize() == dstNodes[i]->getBufferSize());
            adjustedCopySizes[i] = memcpyFunc(dstNodes[i]->getBuffer(), srcNodes[i]->getBuffer(), streams[i], srcNodes[i]->getBufferSize(), loopCount);
            CU_ASSERT(hipEventRecord(endEvents[i], streams[i]));
            if (bandwidthValue == BandwidthValue::TOTAL_BW && i != 0) {
                // make stream0 wait on the all the others so we can measure total completion time
                CU_ASSERT(hipStreamWaitEvent(streams[0], endEvents[i], 0));
            }
        }

        // record the total end - only valid if BandwidthValue::TOTAL_BW is used due to StreamWaitEvent above
        CU_ASSERT(hipCtxSetCurrent(contexts[0]));
        CU_ASSERT(hipEventRecord(totalEnd, streams[0]));

        // unblock the streams
        *blockingVar = 1;

        for (hipStream_t stream : streams) {
            CU_ASSERT(hipStreamSynchronize(stream));
        }

        if (!skipVerification) {
            for (int i = 0; i < srcNodes.size(); i++) {            
                dstNodes[i]->memcmpPattern(dstNodes[i]->getBuffer(), finalCopySize[i], 0xBAADF00D);
            }
        }

        for (int i = 0; i < bandwidthStats.size(); i++) {
            float timeWithEvents = 0.0f;
            CU_ASSERT(hipEventElapsedTime(&timeWithEvents, startEvents[i], endEvents[i]));
            double elapsedWithEventsInUs = ((double) timeWithEvents * 1000.0);
            unsigned long long bandwidth = (adjustedCopySizes[i] * loopCount * 1000ull * 1000ull) / (unsigned long long) elapsedWithEventsInUs;
            bandwidthStats[i]((double) bandwidth);

            if (bandwidthValue == BandwidthValue::SUM_BW || BandwidthValue::TOTAL_BW || i == 0) {
                // Verbose print only the values that are used for the final output
                VERBOSE << "\tSample " << n << ": " << srcNodes[i]->getNodeString() << " -> " << dstNodes[i]->getNodeString() << ": " <<
                    std::fixed << std::setprecision(2) << (double)bandwidth * 1e-9 << " GB/s\n";
            }
        }

        if (bandwidthValue == BandwidthValue::TOTAL_BW) {
            float totalTime = 0.0f;
            CU_ASSERT(hipEventElapsedTime(&totalTime, startEvents[0], totalEnd));
            double elapsedTotalInUs = ((double) totalTime * 1000.0);

            // get total bytes copied
            size_t totalSize = 0;
            for (size_t size : adjustedCopySizes) {
                totalSize += size;
            }

            unsigned long long bandwidth = (totalSize * loopCount * 1000ull * 1000ull) / (unsigned long long) elapsedTotalInUs;
            totalBandwidth((double) bandwidth);

            VERBOSE << "\tSample " << n << ": Total Bandwidth : " <<
                std::fixed << std::setprecision(2) << (double)bandwidth * 1e-9 << " GB/s\n";
        }
    }

    // cleanup
    CU_ASSERT(hipHostFree((void*)blockingVar));

    CU_ASSERT(hipEventDestroy(totalEnd));

    for (int i = 0; i < srcNodes.size(); i++) {
        CU_ASSERT(hipStreamDestroy(streams[i]));
        CU_ASSERT(hipEventDestroy(startEvents[i]));
        CU_ASSERT(hipEventDestroy(endEvents[i]));
    }

    if (bandwidthValue == BandwidthValue::SUM_BW) {
        double sum = 0.0;
        for (auto stat : bandwidthStats) {
            sum += stat.returnAppropriateMetric() * 1e-9;
        }
        return sum;
    } else if (bandwidthValue == BandwidthValue::TOTAL_BW) {
        return totalBandwidth.returnAppropriateMetric() * 1e-9;
    } else {
        return bandwidthStats[0].returnAppropriateMetric() * 1e-9;
    }
}
size_t MemcpyOperationCE::getAdjustedCopySize(size_t size, hipStream_t stream) {
    //CE does not change/truncate buffer size
    return size;
}

MemcpyOperationSM::MemcpyOperationSM(unsigned long long loopCount, ContextPreference ctxPreference, BandwidthValue bandwidthValue) : 
        MemcpyOperation(loopCount, ctxPreference, bandwidthValue) {}

size_t MemcpyOperationSM::memcpyFunc(hipDeviceptr_t dst, hipDeviceptr_t src, hipStream_t stream, size_t copySize, unsigned long long loopCount) {
    return copyKernel(dst, src, copySize, stream, loopCount);
}
size_t MemcpyOperationSM::getAdjustedCopySize(size_t size, hipStream_t stream) {
    hipDevice_t dev;
    hipCtx_t ctx;

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(hipCtxGetDevice(&dev));
    int numSm;
    CU_ASSERT(hipDeviceGetAttribute(&numSm, hipDeviceAttributeMultiprocessorCount, dev));
    unsigned int totalThreadCount = numSm * numThreadPerBlock;
    // We want to calculate the exact copy sizes that will be
    // used by the copy kernels.
    if (size < (defaultBufferSize * _MiB) ) {
        // copy size is rounded down to 16 bytes
        int numUint4 = size / sizeof(uint4);
        return numUint4 * sizeof(uint4);
    }
    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = size / sizeof(uint4);
    // this truncates the copy
    sizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);
    return sizeInElement * sizeof(uint4);
}

MemcpyOperationCE::MemcpyOperationCE(unsigned long long loopCount, ContextPreference ctxPreference, BandwidthValue bandwidthValue) : 
        MemcpyOperation(loopCount, ctxPreference, bandwidthValue) {}

size_t MemcpyOperationCE::memcpyFunc(hipDeviceptr_t dst, hipDeviceptr_t src, hipStream_t stream, size_t copySize, unsigned long long loopCount) {
    for (unsigned int l = 0; l < loopCount; l++) {
        CU_ASSERT(cuMemcpyAsync(dst, src, copySize, stream));
    }

    return copySize;
}
