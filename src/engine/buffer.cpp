// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {

BufferData::BufferData()
{
    device_ = Device::CPU();
    mm_ = MemoryManager::forCPU();
    handle_ = 0;
    host_ptr_ = 0;
    size_ = 0;
    mapcount_ = 0;
}

BufferData::~BufferData()
{
    if (handle_) {
        CV_Assert(mapcount_ == 0);
        mm_->deallocate(device_, handle_);
    }
    handle_ = 0;
    host_ptr_ = 0;
    size_ = 0;
    mapcount_ = 0;
}

void BufferData::release()
{
    if (handle_) {
        CV_Assert(mapcount_ == 0);
        mm_->deallocate(device_, handle_);
        handle_ = 0;
        size_ = 0;
    }
}

Buffer BufferData::allocate(size_t size, Device* device, MemoryManager* mm)
{
    Buffer buf = std::make_shared<BufferData>();
    buf->device_ = device ? device : buf->device_;
    buf->mm_ = mm ? mm : buf->device_->defaultMemoryManager();
    buf->fit(size);

    return buf;
}

Buffer BufferData::allocateOnSameDevice(size_t size) const
{
    return allocate(size, device_, mm_);
}

void BufferData::fit(size_t size)
{
    if (size > size_) {
        release();
        handle_ = mm_->allocate(device_, size);
        if (handle_)
            size_ = size;
    }
}

Device* BufferData::device() const { return device_; }
MemoryManager* BufferData::memoryManager() const { return mm_; }
void* BufferData::handle() const { return handle_; }
size_t BufferData::size() const { return size_; }

void* BufferData::hostPtr() const
{
    CV_Assert(device_->isCPU() || mapcount_ > 0);
    return host_ptr_ ? host_ptr_ : handle_;
}

void* BufferData::map(BufAccess access)
{
    CV_Assert(device_->supportZeroCopy());
    if (CV_XADD(&mapcount_, 1) == 0)
        host_ptr_ = mm_->map(device_, handle_, size_, access);
    return host_ptr_;
}

void BufferData::unmap(BufAccess access)
{
    CV_Assert(device_->supportZeroCopy());
    if (CV_XADD(&mapcount_, -1) == 1) {
        mm_->unmap(device_, handle_, host_ptr_, size_, access);
        host_ptr_ = 0;
    }
}

}}
