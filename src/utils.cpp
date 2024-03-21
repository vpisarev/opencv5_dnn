// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "engine/engine.hpp"

namespace cv { namespace dnn {

bool isIntType(int type)
{
    int depth = CV_MAT_DEPTH(type);
    return depth < CV_32F || depth == CV_32U || depth == CV_64U || depth == CV_64S;
}

bool isSignedIntType(int type)
{
    int depth = CV_MAT_DEPTH(type);
    return depth == CV_8S || depth == CV_16S || depth == CV_32S || depth == CV_64S;
}

bool isFPType(int type)
{
    int depth = CV_MAT_DEPTH(type);
    return depth == CV_32F || depth == CV_64F || depth == CV_16F || depth == CV_16BF;
}

int normalizeAxis(int axis, int ndims)
{
    if (axis < 0)
        axis += ndims;
    CV_Assert(0 <= axis && axis < ndims);
    return axis;
}

// extract array of axes (up to TensorSize::MAX_DIMS).
// Each axis should be within a range [-ndims, ndims-1]
int normalizeAxes(const Tensor& axes, int ndims, int* axisbuf, bool* axismask_)
{
    bool axismaskbuf[TensorSize::MAX_DIMS];
    bool* axismask = axismask_ ? axismask_ : axismaskbuf;

    if (axes.empty()) {
        for (int i = 0; i < ndims; i++)
            axismask[i] = false;
        return 0;
    }

    int axistype = axes.type();
    CV_Assert(axes.ndims() <= 1);
    CV_Assert(axistype == CV_32S || axistype == CV_64S);
    int naxes = (int)axes.total();
    const int32_t* axes32 = (const int32_t*)axes.data();
    const int64_t* axes64 = (const int64_t*)axes.data();

    CV_Assert(naxes <= ndims);

    for (int i = 0; i < ndims; i++)
        axismask[i] = false;

    for (int i = 0; i < (int)naxes; i++) {
        int axis = axistype == CV_32S ? (int)axes32[i] : axes64 ? (int)axes64[i] : i;
        axis = normalizeAxis(axis, ndims);
        if (axismask[axis]) {
            CV_Error(Error::StsError, "there are duplicated axes in the axes specification");
        }
        axismask[axis] = true;
        axisbuf[i] = axis;
    }

    return (int)naxes;
}

TensorSize convInferShape(const TensorSize& inpsize, const ConvParams& convparams)
{
    CV_Assert(inpsize.layout == LAYOUT_NCHWc);

    int ndims = inpsize.ndims;
    size_t nspatialdims = (size_t)(ndims - 3);
    TensorSize outsize = inpsize;

    CV_Assert(convparams.ksizes.size() == nspatialdims);
    CV_Assert(convparams.strides.empty() || convparams.strides.size() == nspatialdims);
    CV_Assert(convparams.dilations.empty() || convparams.dilations.size() == nspatialdims);
    CV_Assert(convparams.pads.empty() || convparams.pads.size() == nspatialdims*2);

    for (size_t i = 0; i < nspatialdims; i++) {
        int ksize = convparams.ksizes[i];
        int stride = convparams.strides.empty() ? 1 : convparams.strides[i];
        int dilation = convparams.dilations.empty() ? 1 : convparams.dilations[i];
        int pad_before = 0, pad_after = 0;
        if (!convparams.pads.empty()) {
            pad_before = convparams.pads[i];
            pad_after = convparams.pads[i + nspatialdims];
        }
        outsize.size[i+2] = (inpsize.size[i+2] + pad_before + pad_after - dilation * (ksize - 1) - 1) / stride + 1;
        CV_Assert(outsize.size[i+2] >= 0);
    }

    return outsize;
}

std::ostream& ConvParams::dump(std::ostream& strm)
{
    strm << "{ ksizes: {";
    bool first = true;
    for (auto ksz: ksizes) {
        if (!first) strm << ", ";
        strm << ksz;
        first = false;
    }

    strm << "}, strides: {";
    first = true;
    for (auto ksz: strides) {
        if (!first) strm << ", ";
        strm << ksz;
        first = false;
    }

    strm << "}, dilations: {";
    first = true;
    for (auto ksz: dilations) {
        if (!first) strm << ", ";
        strm << ksz;
        first = false;
    }

    strm << "}, pads: {";
    first = true;
    for (auto ksz: pads) {
        if (!first) strm << ", ";
        strm << ksz;
        first = false;
    }
    strm << "}}";
    return strm;
}

static void calcKernelOffsets2D(int64_t Wi, int64_t KH, int64_t KW,
                                int64_t DY, int64_t DX,
                                int* yxtab, int64_t* ofstab)
{
    for (int64_t y = 0; y < KH; y++)
        for (int64_t x = 0; x < KW; x++) {
            int64_t k = y*KW + x;
            int64_t dy = y*DY, dx = x*DX;
            yxtab[k*2] = (int)dy; yxtab[k*2+1] = (int)dx;
            ofstab[k] = dy*Wi + dx;
        }
}

DepthwiseConvParams initDepthwiseConv(const TensorSize& inpsize,
                                      const ConvParams& convparams,
                                      int* yxtab, int64_t* ofstab)
{
    DepthwiseConvParams dwparams;
    TensorSize outsize = convInferShape(inpsize, convparams);

    int ndims = inpsize.ndims;
    size_t nspatialdims = ndims - 3;

    CV_Assert(inpsize.layout == LAYOUT_NCHWc);
    CV_Assert(1 <= nspatialdims && nspatialdims <= 2);
    int64_t N = outsize.size[0];
    int64_t C1 = outsize.size[1];
    int64_t C0 = outsize.size[ndims-1];
    int64_t W = outsize.size[ndims-2];
    int64_t H = nspatialdims > 1 ? outsize.size[ndims-3] : 1;
    int64_t Wi = inpsize.size[ndims-2];
    int64_t Hi = nspatialdims > 1 ? inpsize.size[ndims-3] : 1;
    int64_t SY = 1, SX = 1, DY = 1, DX = 1;
    int64_t pad_y0 = 0, pad_x0 = 0, pad_y1 = 0, pad_x1 = 0;

    int64_t KW = convparams.ksizes[nspatialdims-1];
    int64_t KH = nspatialdims > 1 ? convparams.ksizes[nspatialdims-2] : 1;

    if (!convparams.strides.empty()) {
        SX = convparams.strides[nspatialdims-1];
        SY = nspatialdims > 1 ? convparams.strides[nspatialdims-2] : 1;
    }

    if (!convparams.dilations.empty()) {
        DX = convparams.dilations[nspatialdims-1];
        DY = nspatialdims > 1 ? convparams.dilations[nspatialdims-2] : 1;
    }

    if (!convparams.pads.empty()) {
        pad_x0 = convparams.pads[nspatialdims-1];
        pad_x1 = convparams.pads[nspatialdims*2-1];
        pad_y0 = nspatialdims > 1 ? convparams.pads[nspatialdims-2] : 0;
        pad_y1 = nspatialdims > 1 ? convparams.pads[nspatialdims*2-2] : 0;
    }

    int64_t inner_y0 = (pad_y0 + SY - 1)/SY;
    int64_t inner_x0 = (pad_x0 + SX - 1)/SX;

    int64_t inner_y1 = (Hi - (KH - 1)*DY + pad_y0)/SY;
    int64_t inner_x1 = (Wi - (KW - 1)*DX + pad_x0)/SX;

    inner_y1 += inner_y1*SY - pad_y0 + (KH-1)*DY < Hi;
    inner_x1 += inner_x1*SX - pad_x0 + (KW-1)*DX < Wi;

    inner_y1 = std::min(inner_y1, H);
    inner_x1 = std::min(inner_x1, W);

    if (inner_y0 >= inner_y1 || inner_x0 >= inner_x1) {
        inner_y0 = H;
        inner_x0 = W;
    }

    calcKernelOffsets2D(Wi, KH, KW, DY, DX, yxtab, ofstab);

    dwparams.KH = KH;
    dwparams.KW = KW;
    dwparams.SY = SY;
    dwparams.SX = SX;
    dwparams.DY = DY;
    dwparams.DX = DX;

    dwparams.pad_y0 = pad_y0;
    dwparams.pad_x0 = pad_x0;
    dwparams.pad_y1 = pad_y1;
    dwparams.pad_x1 = pad_x1;

    dwparams.N = N;
    dwparams.C1 = C1;
    dwparams.C0 = C0;
    dwparams.H = H;
    dwparams.W = W;
    dwparams.Hi = Hi;
    dwparams.Wi = Wi;

    dwparams.inner_y0 = inner_y0;
    dwparams.inner_x0 = inner_x0;
    dwparams.inner_y1 = inner_y1;
    dwparams.inner_x1 = inner_x1;

    dwparams.yxtab = yxtab;
    dwparams.ofstab = ofstab;

    return dwparams;
}

std::ostream& DepthwiseConvParams::dump(std::ostream& strm)
{
    strm << "{N=" << N << ", C1=" << C1 << ", C0=" << C0;
    strm << ", H=" << H << ", W=" << W << ", Hi=" << Hi << ", Wi=" << Wi;
    strm << ", KH=" << KH << ", KW=" << KW << ", SY=" << SY << ", SX=" << SX;
    strm << ", DY=" << DY << ", DX=" << DX;
    strm << ", pad_y0=" << pad_y0 << ", pad_x0=" << pad_x0;
    strm << ", pad_y1=" << pad_y1 << ", pad_x1=" << pad_x1;
    strm << "}";
    return strm;
}

}}
