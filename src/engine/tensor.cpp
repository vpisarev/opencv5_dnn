// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {

///////////////////////////// TensorSize /////////////////////////////

static void finalizeBlockLayout(TensorSize& size)
{
    if (size.layout == LAYOUT_NCHWc) {
        CV_Assert(size.ndims >= 4);
        int64_t C0 = size.size[size.ndims-1];
        CV_Assert(C0 > 1 && (C0 & (C0-1)) == 0);
        size.C = (int64_t)size.size[1]*size.size[size.ndims-1];
    }
}

TensorSize::TensorSize()
{
    layout = LAYOUT_UNKNOWN;
    ndims = 1;
    C = 0;
    size[0] = 0;
}

TensorSize::TensorSize(int ndims_, const int64_t* size_, TensorLayout layout_)
{
    layout = layout_;
    CV_Assert(0 <= ndims_ && ndims_ <= MAX_NDIMS);
    ndims = ndims_;
    for (int i = 0; i < ndims; i++) {
        size[i] = size_[i];
        CV_Assert(size[i] >= 0);
    }
    finalizeBlockLayout(*this);
}

TensorSize::TensorSize(std::initializer_list<int64_t> size_, TensorLayout layout_)
{
    layout = layout_;
    size_t sz = size_.size();
    CV_Assert(sz <= MAX_NDIMS);
    ndims = (int)sz;
    auto it = size_.begin();
    for (int i = 0; i < ndims; i++, ++it) {
        size[i] = *it;
        CV_Assert(size[i] >= 0);
    }
    finalizeBlockLayout(*this);
}

size_t TensorSize::total() const
{
    size_t p = 1;
    for (size_t i = 0; i < ndims; i++) p *= size[i];
    return p;
}

bool TensorSize::empty() const
{
    return total() == 0;
}

TensorSize TensorSize::fromArray(InputArray m, TensorLayout layout_)
{
    TensorSize size;
    size.layout = layout_;

    if (!m.empty()) {
        int msize[TensorSize::MAX_NDIMS];
        size.ndims = m.sizend(msize);
        for (int i = 0; i < size.ndims; i++)
            size.size[i] = msize[i];
        finalizeBlockLayout(size);
    }
    return size;
}

int TensorSize::toMatShape(int* mshape, int maxdims) const
{
    CV_Assert(maxdims >= ndims);
    for (int i = 0; i < ndims; i++) {
        int64_t sz_i = size[i];
        CV_Assert(INT_MIN <= sz_i && sz_i <= INT_MAX);
        mshape[i] = (int)sz_i;
    }
    return ndims;
}

TensorSize TensorSize::toBlock(int64_t C0) const
{
    CV_Assert(ndims >= 3);
    // C0 should be > 1 and be a power-of-2: 2, 4, 8, ...
    CV_Assert(C0 > 1 && (C0 & (C0-1)) == 0);
    CV_Assert(layout == LAYOUT_NCHW || layout == LAYOUT_NHWC);
    int c_idx = layout == LAYOUT_NCHW ? 1 : ndims-1;

    TensorSize newsize = *this;
    newsize.layout = LAYOUT_NCHWc;
    newsize.C = size[c_idx];
    newsize.size[newsize.ndims++] = C0;
    newsize.size[c_idx] = (size[c_idx] + C0 - 1)/C0;

    return newsize;
}

TensorSize TensorSize::fromBlock(TensorLayout newLayout) const
{
    CV_Assert(ndims >= 4);
    CV_Assert(layout == LAYOUT_NCHWc);
    // C0 should be > 1 and be a power-of-2: 2, 4, 8, ...
    int64_t C0 = size[ndims-1];
    CV_Assert(C0 > 1 && (C0 & (C0-1)) == 0);
    CV_Assert(size[1] == (C + C0-1)/C0);
    CV_Assert(newLayout == LAYOUT_NCHW || newLayout == LAYOUT_NHWC);
    int c_idx = newLayout == LAYOUT_NCHW ? 1 : ndims-2;

    TensorSize newsize = *this;
    newsize.layout = newLayout;
    newsize.C = 0;
    newsize.size[c_idx] = C;
    newsize.ndims--;

    return newsize;
}

bool operator == (const TensorSize& size1, const TensorSize& size2)
{
    if (size1.ndims != size2.ndims)
        return false;
    if (size1.layout == LAYOUT_NCHWc &&
        size2.layout == LAYOUT_NCHWc &&
        size1.C != size2.C)
        return false;
    for (int i = 0; i < size1.ndims; i++) {
        if (size1.size[i] != size2.size[i])
            return false;
    }
    return true;
}

bool operator != (const TensorSize& size1, const TensorSize& size2)
{
    return !(size1 == size2);
}

void TensorSize::dump(std::ostream& strm) const
{
    if (empty()) {
        strm << "<empty>";
    } else {
        if (layout == LAYOUT_ND)
            strm << "ND ";
        else if (layout == LAYOUT_NCHW)
            strm << "NCHW ";
        else if (layout == LAYOUT_NHWC)
            strm << "NHWC ";
        else if (layout == LAYOUT_NCHWc)
            strm << "NCHWc ";
        strm << "<";
        for (int i = 0; i < ndims; i++) {
            strm << size[i];
            if (i < ndims-1)
                strm << " x ";
        }
        strm << ">";
    }
}

size_t SizeType::totalBytes() const
{
    return size.total()*CV_ELEM_SIZE(type);
}

SizeType SizeType::toBlock(int64_t C0) const
{
    return {size.toBlock(C0), type};
}

SizeType SizeType::fromBlock(TensorLayout layout) const
{
    return {size.fromBlock(layout), type};
}

static const char* depthToStr(int type)
{
    int depth = CV_MAT_DEPTH(type);
    return
        depth == CV_8U ? "U8" :
        depth == CV_8S ? "I8" :
        depth == CV_16U ? "U16" :
        depth == CV_16S ? "I16" :
        depth == CV_32U ? "U32" :
        depth == CV_32S ? "I32" :
        depth == CV_64U ? "U64" :
        depth == CV_64S ? "I64" :
        depth == CV_32F ? "F32" :
        depth == CV_64F ? "F64" :
        depth == CV_16F ? "F16" :
        depth == CV_16BF ? "BF16" :
        depth == CV_Bool ? "Bool" : "???";
}

void SizeType::dump(std::ostream& strm) const
{
    strm << depthToStr(type);
    int cn = CV_MAT_CN(type);
    if (cn > 1)
        strm << "C" << cn;
    strm << " ";
    size.dump(strm);
}

///////////////////////////// Tensor /////////////////////////////

void Tensor::init()
{
    flags_ = 0;
    type_ = 0;
    ext_data_ = 0;
    slice_start_ = slice_maxsize_ = 0;
}

Tensor::Tensor()
{
    init();
}

Tensor::Tensor(const Tensor& t)
{
    flags_ = t.flags_;
    type_ = t.type_;
    size_ = t.size_;
    buf_ = t.buf_;
    ext_data_ = t.ext_data_;
    slice_start_ = t.slice_start_;
    slice_maxsize_ = t.slice_maxsize_;
}

Tensor& Tensor::operator = (const Tensor& t)
{
    if (this != &t) {
        flags_ = t.flags_;
        type_ = t.type_;
        size_ = t.size_;
        buf_ = t.buf_;
        ext_data_ = t.ext_data_;
        slice_start_ = t.slice_start_;
        slice_maxsize_ = t.slice_maxsize_;
    }
    return *this;
}

void Tensor::release()
{
    type_ = 0;
    size_ = TensorSize();
    if (!usesBufferSlice()) {
        flags_ = 0;
        if (buf_)
            buf_->release();
        slice_start_ = slice_maxsize_ = 0;
    }
    ext_data_ = 0;
}

Tensor::~Tensor() {}

Tensor::Tensor(const TensorSize& size, int type, Device* device)
{
    flags_ = CONTINUOUS_FLAG;
    type_ = type;
    size_ = size;
    size_t nbytes = totalBytes();
    buf_ = BufferData::allocate(nbytes, device);
    ext_data_ = 0;
    slice_start_ = slice_maxsize_ = 0;
}

Tensor::Tensor(const TensorSize& size, int type, void* data, bool copy, Device* device)
{
    init();
    setData(size, type, data, copy, device);
}

Tensor::Tensor(const Buffer& buffer)
{
    init();
    buf_ = buffer;
}

Tensor::Tensor(const Buffer& buffer, size_t start, size_t maxsize)
{
    init();
    setBufferSlice(buffer, start, maxsize);
}

Tensor::Tensor(InputArray arr, TensorLayout layout, bool copy, Device* device)
{
    init();
    setData(arr, layout, copy, device);
}

void Tensor::multiFit(Buffer& buffer,
                      std::initializer_list<SizeType> st_list,
                      std::initializer_list<Tensor*> tensors,
                      size_t alignment)
{
    alignment = alignment > 0 ? alignment : 32;
    CV_Assert(buffer);
    CV_Assert((alignment & (alignment - 1)) == 0);
    CV_Assert(st_list.size() == tensors.size());
    size_t offset = 0, size = 0;
    for (auto st: st_list) {
        size_t nbytes = st.totalBytes();
        nbytes = (nbytes + alignment-1) & (~alignment + 1);
        size += nbytes;
    }
    buffer->fit(size);
    auto st_it = st_list.begin();
    auto t_it = tensors.begin();
    size_t i, ntensors = tensors.size();
    for (i = 0; i < ntensors; i++, ++st_it, ++t_it) {
        SizeType st = *st_it;
        Tensor* t = *t_it;
        CV_Assert(t != nullptr);
        size_t nbytes = st.totalBytes();
        nbytes = (nbytes + alignment-1) & (~alignment + 1);
        t->setBufferSlice(buffer, offset, offset + nbytes);
        offset += nbytes;
    }
}

Tensor Tensor::makeScalar(int type, const void* value, Device* device)
{
    return Tensor(TensorSize({1}, LAYOUT_UNKNOWN), type, (void*)value, true, device);
}

void Tensor::fit(const TensorSize& size, int type)
{
    if (size == size_ && CV_MAT_TYPE(type) == type_)
        return;
    CV_Assert(buf_);
    CV_Assert(!ext_data_);
    flags_ |= CONTINUOUS_FLAG;
    type_ = type;
    size_ = size;
    size_t nbytes = totalBytes();
    if (usesBufferSlice()) {
        CV_Assert(nbytes <= slice_maxsize_ && slice_start_ + nbytes <= buf_->size());
        return;
    }
    buf_->fit(nbytes);
}

void Tensor::fitSameDevice(const Tensor& tensor, const TensorSize& size, int type)
{
    Device* target_device = tensor.device();
    flags_ |= CONTINUOUS_FLAG;
    type_ = type;
    size_ = size;
    ext_data_ = 0;
    size_t nbytes = totalBytes();
    if (!buf_ || !buf_->device()->isSameDevice(target_device)) {
        CV_Assert(!usesBufferSlice());
        buf_ = BufferData::allocate(nbytes, target_device);
    } else {
        buf_->fit(nbytes);
    }
}

bool Tensor::isOnSameDevice(const Tensor& tensor)
{
    return device()->isSameDevice(tensor.device());
}

void Tensor::setData(const TensorSize& size, int type, void* data, bool copy, Device* device)
{
    type_ = CV_MAT_TYPE(type);
    size_ = size;
    CV_Assert(size.empty() == (data == nullptr));
    if (!copy) {
        CV_Assert(!device || device->isCPU());
        flags_ = CONTINUOUS_FLAG;
        buf_ = Buffer();
        ext_data_ = data;
        slice_start_ = slice_maxsize_ = 0;
    } else {
        flags_ |= CONTINUOUS_FLAG;
        size_t nbytes = totalBytes();
        if (!buf_ || !buf_->device()->isSameDevice(device)) {
            CV_Assert(!usesBufferSlice());
            buf_ = BufferData::allocate(nbytes, device);
        } else {
            buf_->fit(nbytes);
        }
        buf_->memoryManager()->copyToDevice(buf_->device(), data, buf_->handle(), slice_start_, nbytes);
    }
}

void Tensor::setData(InputArray arr, TensorLayout layout, bool copy, Device* device)
{
    if (!copy) {
        CV_Assert(arr.isMat() || arr.isVector() || arr.isMatx());
        CV_Assert(arr.isContinuous());
    }
    // [TODO] once Tensor support is added to InputArray,
    // we need to provide quick path for Tensor=>Tensor construction
    // (especially if the source and the constructed Tensor are on the same device)
    Mat m = arr.getMat();

    if (!m.isContinuous()) {
        m = m.clone(); // for now we don't support non-contiguous tensors
    }
    TensorSize shape_ = TensorSize::fromArray(m, layout);
    int typ_ = m.type();
    setData(shape_, typ_, m.data, copy, device);
}

void Tensor::setBuffer(const Buffer& buffer)
{
    init();
    size_ = TensorSize();
    buf_ = buffer;
}

void Tensor::setBufferSlice(const Buffer& buffer, size_t start, size_t maxsize)
{
    CV_Assert(buffer);
    size_t bufsize = buffer->size();
    CV_Assert(start + maxsize <= bufsize);
    flags_ = BUFFER_SLICE_FLAG;
    type_ = 0;
    size_ = TensorSize();
    buf_ = buffer;
    slice_start_ = start;
    slice_maxsize_ = maxsize;
}

Buffer Tensor::buffer() const {
    CV_Assert(!ext_data_);
    return buf_;
}

bool Tensor::isContinuous() const { return (flags_ & CONTINUOUS_FLAG) != 0; }
bool Tensor::usesBufferSlice() const { return (flags_ & BUFFER_SLICE_FLAG) != 0; }

Device* Tensor::device() const { return buf_ ? buf_->device() : Device::CPU(); }
DeviceType Tensor::deviceType() const { return device()->type(); }
MemoryManager* Tensor::memoryManager() const { return buf_ ? buf_->memoryManager() : MemoryManager::forCPU(); }
size_t Tensor::total() const { return size_.total(); }
size_t Tensor::totalBytes() const { return size_.total()*elementSize(); }
size_t Tensor::elementSize() const { return CV_ELEM_SIZE(type_); }
bool Tensor::empty() const { return size_.empty(); }
void* Tensor::handle() const { return buf_ ? buf_->handle() : nullptr; }
int Tensor::dims() const { return size_.ndims; }
TensorSize Tensor::size() const { return size_; }
SizeType Tensor::sizetype() const { return SizeType({size_, type_}); }
TensorLayout Tensor::layout() const { return size_.layout; }
int Tensor::type() const { return type_; }
int Tensor::depth() const { return CV_MAT_DEPTH(type_); };
int Tensor::channels() const { return CV_MAT_CN(type_); }

void* Tensor::data() const
{
    void* dataptr = buf_ ? buf_->hostPtr() : ext_data_;
    CV_Assert(dataptr != 0 || empty()); // make sure the tensor is "mapped" to memory
    return dataptr;
}

Mat Tensor::getMat() const
{
    int mshape[TensorSize::MAX_NDIMS];
    int mdims = size_.toMatShape(mshape, TensorSize::MAX_NDIMS);
    void* dataptr = data();
    return Mat(mdims, mshape, type_, dataptr);
}

void* Tensor::map(BufAccess access)
{
    if (buf_)
        return buf_->map(access);
    return ext_data_;
}

void Tensor::unmap(BufAccess access)
{
    if (buf_)
        buf_->unmap(access);
}

Tensor Tensor::download() const
{
    if (deviceType() == Device_CPU)
        return *this;
    Tensor t(size_, type_, nullptr);
    copyTo(t);
    return t;
}

Tensor Tensor::upload(Device* device_) const
{
    if (device()->isSameDevice(device_))
        return *this;
    Tensor t(size_, type_, device_);
    copyTo(t);
    return t;
}

Tensor Tensor::uploadToSameDevice(const Tensor& t) const
{
    return upload(t.device());
}

void Tensor::copyTo(Tensor& tensor) const
{
    size_t nbytes = totalBytes();
    Device* srcdev = device();
    if (!tensor.buf_) {
        tensor.fitSameDevice(*this, size_, type_);
    } else {
        tensor.fit(size_, type_);
    }
    Device* dstdev = tensor.device();
    if (nbytes == 0)
        return;
    if (srcdev->isCPU() && dstdev->isCPU()) {
        memcpy(tensor.data(), data(), nbytes);
    } else if (srcdev->isSameDevice(dstdev)) {
        memoryManager()->copyWithinDevice(srcdev, handle(), slice_start_,
                                          tensor.handle(), tensor.slice_start_, nbytes);
    } else if (srcdev->isCPU()) {
        tensor.memoryManager()->copyToDevice(dstdev, data(), tensor.handle(),
                                             tensor.slice_start_, nbytes);
    } else if (dstdev->isCPU()) {
        memoryManager()->copyFromDevice(srcdev, handle(), slice_start_, tensor.data(), nbytes);
    } else {
        // [TODO] accelerate via mapping
        Tensor t = download();
        t.copyTo(tensor);
    }
}

static void cvtScalar(int srctype, const void* src, int dsttype, void* dst, double* temp, int max_cn)
{
    int scn = CV_MAT_CN(srctype);
    int sdepth = CV_MAT_DEPTH(srctype);
    int dcn = CV_MAT_CN(dsttype);
    int ddepth = CV_MAT_DEPTH(dsttype);
    CV_Assert(scn == dcn || scn == 1);
    CV_Assert(scn <= max_cn);
    for (int i = 0; i < scn; i++) {
        temp[i] =
            sdepth == CV_8U ? (double)((const uint8_t*)src)[i] :
            sdepth == CV_8S ? (double)((const int8_t*)src)[i] :
            sdepth == CV_16U ? (double)((const uint16_t*)src)[i] :
            sdepth == CV_16S ? (double)((const int16_t*)src)[i] :
            sdepth == CV_32U ? (double)((const uint32_t*)src)[i] :
            sdepth == CV_32S ? (double)((const int32_t*)src)[i] :
            sdepth == CV_64U ? (double)((const uint64_t*)src)[i] :
            sdepth == CV_64S ? (double)((const int64_t*)src)[i] :
            sdepth == CV_32F ? (double)((const float*)src)[i] :
            sdepth == CV_64F ? (double)((const double*)src)[i] :
            sdepth == CV_16F ? (double)((const cv::float16_t*)src)[i] :
            sdepth == CV_16BF ? (double)((const cv::bfloat16_t*)src)[i] :
            sdepth == CV_Bool ? (double)((const bool*)src)[i] : 0.;
    }
    for (int i = 0; i < dcn; i++) {
        int j = scn == dcn ? i : 0;
        if (ddepth == CV_8U)
            ((uint8_t*)dst)[i] = saturate_cast<uint8_t>(temp[j]);
        else if (ddepth == CV_8S)
            ((int8_t*)dst)[i] = saturate_cast<int8_t>(temp[j]);
        else if (ddepth == CV_16U)
            ((uint16_t*)dst)[i] = saturate_cast<uint16_t>(temp[j]);
        else if (ddepth == CV_16S)
            ((int16_t*)dst)[i] = saturate_cast<int16_t>(temp[j]);
        else if (ddepth == CV_32U)
            ((uint32_t*)dst)[i] = saturate_cast<uint32_t>(temp[j]);
        else if (ddepth == CV_32S)
            ((int32_t*)dst)[i] = saturate_cast<int32_t>(temp[j]);
        else if (ddepth == CV_64U)
            ((uint64_t*)dst)[i] = saturate_cast<uint64_t>(temp[j]);
        else if (ddepth == CV_64S)
            ((int64_t*)dst)[i] = saturate_cast<int64_t>(temp[j]);
        else if (ddepth == CV_32F)
            ((float*)dst)[i] = saturate_cast<float>(temp[j]);
        else if (ddepth == CV_64F)
            ((double*)dst)[i] = saturate_cast<double>(temp[j]);
        else if (ddepth == CV_16F)
            ((cv::float16_t*)dst)[i] = saturate_cast<cv::float16_t>(temp[j]);
        else if (ddepth == CV_16BF)
            ((cv::bfloat16_t*)dst)[i] = saturate_cast<cv::bfloat16_t>(temp[j]);
        else if (ddepth == CV_Bool)
            ((bool*)dst)[i] = temp[j] != 0;
        else
            ((int64_t*)dst)[i] = 0;
    }
}

void Tensor::setTo(int vtype, const void* value0)
{
    constexpr int MAX_CN = 8;
    double temp[MAX_CN], value[MAX_CN];

    if (empty())
        return;

    cvtScalar(vtype, value0, type_, value, temp, MAX_CN);
    memoryManager()->fill(device(), handle(), slice_start_, total(), value, elementSize());
}

template<typename _Tp> struct Fmt
{
    typedef int temp_type;
    static const char* fmt() { return "%d"; }
};

template<> struct Fmt<uint32_t>
{
    typedef unsigned temp_type;
    static const char* fmt() { return "%u"; }
};

template<> struct Fmt<int64_t>
{
    typedef long long temp_type;
    static const char* fmt() { return "%lld"; }
};

template<> struct Fmt<uint64_t>
{
    typedef unsigned long long temp_type;
    static const char* fmt() { return "%llu"; }
};

template<> struct Fmt<float>
{
    typedef float temp_type;
    static const char* fmt() { return "%.5g"; }
};

template<> struct Fmt<double>
{
    typedef double temp_type;
    static const char* fmt() { return "%.5g"; }
};

template<> struct Fmt<cv::float16_t>
{
    typedef float temp_type;
    static const char* fmt() { return "%.5g"; }
};

template<> struct Fmt<cv::bfloat16_t>
{
    typedef float temp_type;
    static const char* fmt() { return "%.4g"; }
};

template <typename _Tp>
static void dumpRow(std::ostream& strm, const _Tp* ptr, int64_t n, size_t ofs, int64_t border)
{
    char buf[128];
    const char* fmt = Fmt<_Tp>::fmt();
    int64_t i, ndump = border > 0 ? std::min(n, border*2+1) : n;
    if (border == 0)
        border = ndump;
    for (i = 0; i < ndump; i++) {
        int64_t j = n == ndump || i < border ? i : i == border ? -1 : n-border*2-1+i;
        if (i > 0)
            strm << ", ";
        if (j >= 0) {
            snprintf(buf, sizeof(buf), fmt, (typename Fmt<_Tp>::temp_type)ptr[ofs + j]);
            strm << buf;
        } else
            strm << "... ";
    }
}

static void dumpSlice(std::ostream& strm, const Tensor& t, const size_t* step, int d, size_t ofs, int64_t border)
{
    TensorSize size = t.size();
    int ndims = size.ndims;
    int64_t n = d >= ndims ? 1 : size.size[d];
    if (d >= ndims - 1) {
        int typ = t.depth();
        void* data = t.data();
        n *= t.channels();
        if (typ == CV_8U)
            dumpRow(strm, (const uint8_t*)data, n, ofs, border);
        else if (typ == CV_8S)
            dumpRow(strm, (const int8_t*)data, n, ofs, border);
        else if (typ == CV_16U)
            dumpRow(strm, (const uint16_t*)data, n, ofs, border);
        else if (typ == CV_16S)
            dumpRow(strm, (const int16_t*)data, n, ofs, border);
        else if (typ == CV_32U)
            dumpRow(strm, (const unsigned*)data, n, ofs, border);
        else if (typ == CV_32S)
            dumpRow(strm, (const int*)data, n, ofs, border);
        else if (typ == CV_64U)
            dumpRow(strm, (const uint64_t*)data, n, ofs, border);
        else if (typ == CV_64S)
            dumpRow(strm, (const int64_t*)data, n, ofs, border);
        else if (typ == CV_32F)
            dumpRow(strm, (const float*)data, n, ofs, border);
        else if (typ == CV_64F)
            dumpRow(strm, (const double*)data, n, ofs, border);
        else if (typ == CV_16F)
            dumpRow(strm, (const cv::float16_t*)data, n, ofs, border);
        else if (typ == CV_16BF)
            dumpRow(strm, (const cv::bfloat16_t*)data, n, ofs, border);
        else if (typ == CV_Bool)
            dumpRow(strm, (const bool*)data, n, ofs, border);
        else {
            CV_Error(Error::StsNotImplemented, "unsupported type");
        }
    } else {
        int64_t i, ndump = border > 0 ? std::min(n, border*2+1) : n;
        bool dots = false;
        for (i = 0; i < ndump; i++) {
            if (i > 0 && !dots) {
                int nempty_lines = ndims - 2 - d;
                for (int k = 0; k < nempty_lines; k++)
                    strm << "\n";
            }
            if (i > 0)
                strm << "\n";
            int64_t j = n == ndump || i < border ? i :
                        i == border ? -1 :
                        n - border*2 - 1 + i;
            bool dots = j < 0;
            if (!dots)
                dumpSlice(strm, t, step, d+1, ofs + j*step[d], border);
            else
                strm << "...";
        }
    }
}

void Tensor::dump(std::ostream& strm, int indent, int border0_,
                  size_t maxsz_all_, bool braces) const
{
    int border0 = border0_ > 0 ? border0_ : 3;
    size_t maxsz_all = maxsz_all_ > 0 ? maxsz_all_ : 100;
    Tensor temp;
    const Tensor* t = this;
    if (deviceType() != Device_CPU) {
        temp = download();
        t = &temp;
    }
    size_t sz_all = t->total();
    if (braces)
        strm << "[";
    if (sz_all == 0) {
        if (!braces)
            strm << "no data";
    } else {
        int ndims = size_.ndims;
        int64_t border = sz_all < maxsz_all ? 0 : border0;
        int cn = channels();
        size_t step[TensorSize::MAX_NDIMS];
        step[ndims-1] = 1;
        for (int i = ndims-2; i >= 0; i--) {
            step[i] = step[i+1]*size_.size[i+1]*cn;
            cn = 1;
        }

        dumpSlice(strm, *t, step, 0, 0, border);
    }
    if (braces)
        strm << "]";
}

}}
