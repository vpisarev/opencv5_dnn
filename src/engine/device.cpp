// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {

struct CPUMemoryManager : MemoryManager
{
    void* allocate(Device*, size_t bufsize) CV_OVERRIDE { return bufsize > 0 ? malloc(bufsize) : 0; }
    void deallocate(Device*, void* handle) CV_OVERRIDE { if (handle) free(handle); }
    void* map(Device*, void* handle, size_t, int) CV_OVERRIDE { return handle; }
    void unmap(Device*, void*, void*, size_t, int) CV_OVERRIDE {}
    void copyFromDevice(Device*, void* handle, size_t offset, void* dst, size_t size) CV_OVERRIDE
    {
        if (size > 0)
            memcpy(dst, (char*)handle + offset, size);
    }
    void copyToDevice(Device*, const void* src, void* handle, size_t offset, size_t size) CV_OVERRIDE
    {
        if (size > 0)
            memcpy((char*)handle + offset, src, size);
    }
    void copyWithinDevice(Device* device, const void* srchandle, size_t srcoffset,
                          void* dsthandle, size_t dstoffset, size_t size)
    {
        if (size > 0)
            memcpy((char*)dsthandle + dstoffset, (char*)srchandle + srcoffset, size);
    }
    void fill(Device* device, void* handle, size_t offset, size_t nelems, const void* value, size_t vsize)
    {
        if (nelems == 0)
            return;
        if (vsize == 1 || !value) {
            int v = value ? (int)*(const char*)value : 0;
            memset((char*)handle + offset, v, nelems*vsize);
        } else if (vsize == 2) {
            int16_t v = *(const int16_t*)value;
            int16_t* ptr = (int16_t*)((char*)handle + offset);
            for (size_t i = 0; i < nelems; i++)
                ptr[i] = v;
        } else if (vsize == 3) {
            const uint8_t* v = (const uint8_t*)value;
            uint8_t v0=v[0], v1=v[1], v2=v[2];
            uint8_t* ptr = (uint8_t*)((char*)handle + offset);
            for (size_t i = 0; i < nelems; i++, ptr += 3) {
                ptr[0] = v0;
                ptr[1] = v1;
                ptr[2] = v2;
            }
        } else if (vsize == 4) {
            int32_t v = *(const int32_t*)value;
            int32_t* ptr = (int32_t*)((char*)handle + offset);
            for (size_t i = 0; i < nelems; i++)
                ptr[i] = v;
        } else if (vsize == 6) {
            const uint16_t* v = (const uint16_t*)value;
            uint16_t v0 = v[0], v1 = v[1], v2 = v[2];
            uint16_t* ptr = (uint16_t*)((char*)handle + offset);
            for (size_t i = 0; i < nelems; i++, ptr += 3) {
                ptr[0] = v0;
                ptr[1] = v1;
                ptr[2] = v2;
            }
        } else if (vsize == 8) {
            int64_t v = *(const int16_t*)value;
            int64_t* ptr = (int64_t*)((char*)handle + offset);
            for (size_t i = 0; i < nelems; i++)
                ptr[i] = v;
        } else if (vsize == 12) {
            const uint32_t* v = (const uint32_t*)value;
            uint32_t v0 = v[0], v1 = v[1], v2 = v[2];
            uint32_t* ptr = (uint32_t*)((char*)handle + offset);
            for (size_t i = 0; i < nelems; i++, ptr += 3) {
                ptr[0] = v0;
                ptr[1] = v1;
                ptr[2] = v2;
            }
        } else if (vsize == 16) {
            const uint64_t* v = (const uint64_t*)value;
            uint64_t v0 = v[0], v1 = v[1];
            uint64_t* ptr = (uint64_t*)((char*)handle + offset);
            for (size_t i = 0; i < nelems; i++, ptr += 2) {
                ptr[0] = v0;
                ptr[1] = v1;
            }
        } else if (vsize % 4 == 0) {
            const uint32_t* v = (const uint32_t*)value;
            uint32_t* ptr = (uint32_t*)((char*)handle + offset);
            vsize /= 4;
            for (size_t i = 0; i < nelems; i++, ptr += vsize) {
                for (size_t j = 0; j < vsize; j++)
                    ptr[j] = v[j];
            }
        } else {
            const uint8_t* v = (const uint8_t*)value;
            uint8_t* ptr = (uint8_t*)((char*)handle + offset);
            for (size_t i = 0; i < nelems; i++, ptr += vsize) {
                for (size_t j = 0; j < vsize; j++)
                    ptr[j] = v[j];
            }
        }
    }
};

static CPUMemoryManager g_CPUMemoryManager;

MemoryManager* getCPUMemoryManager() { return &g_CPUMemoryManager; }

Device::~Device() {}
bool Device::isCPU() const { return type() == Device_CPU; }

struct CPUDevice : Device
{
    virtual std::string_view name() const CV_OVERRIDE
    { return
#if (defined __SSE2__) || (defined _M_X86) || (defined _M_X64)
        "x86 CPU";
#elif (defined __ARM_NEON)
        "ARM CPU";
#elif (defined CV_SIMD)
        "Generic CPU with SIMD";
#endif
    }
    virtual DeviceType type() const CV_OVERRIDE { return Device_CPU; }
    virtual bool supportZeroCopy() const CV_OVERRIDE { return true; }
    virtual bool supportType(int typ) const CV_OVERRIDE { return true; }
    virtual int ndevices() const CV_OVERRIDE { return 1; }
    virtual int index() const CV_OVERRIDE { return 0; }
    virtual MemoryManager* defaultMemoryManager() CV_OVERRIDE { return &g_CPUMemoryManager; }
    virtual Device* getSameKindDevice(int index) const CV_OVERRIDE;
    virtual bool isSameDevice(Device* device) const CV_OVERRIDE;
};

static CPUDevice g_CPU;

Device* CPUDevice::getSameKindDevice(int index) const
{
    CV_Assert(index == 0);
    return &g_CPU;
}

bool CPUDevice::isSameDevice(Device* device) const
{
    return device == nullptr || device->type() == Device_CPU;
}

Device* getCPUDevice() { return &g_CPU; }

}}
