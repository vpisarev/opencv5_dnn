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

TensorSize convInferShape(const TensorSize& inpsize, const ConvParams& convparams, const TensorSize& wsize)
{
    CV_Assert(inpsize.layout == LAYOUT_NCHWc);

    int ndims = inpsize.ndims;
    size_t nspatialdims = (size_t)(ndims - 3);
    TensorSize outsize = inpsize;
    int64_t ksizes[TensorSize::MAX_DIMS];

    if (!convparams.ksizes.empty()) {
        size_t ksizes_size = convparams.ksizes.size();
        CV_Assert(ksizes_size == nspatialdims || ksizes_size == nspatialdims+2);
        for (size_t i = 0; i < nspatialdims; i++)
            ksizes[i] = convparams.ksizes[ksizes_size - nspatialdims + i];
        if (convparams.ngroups == 0 || ksizes_size == nspatialdims) {
            outsize.size[1] = inpsize.size[1];
        } else {
            CV_Assert(convparams.ksizes[0] % inpsize.size[ndims-1] == 0);
            outsize.size[1] = convparams.ksizes[0]/inpsize.size[ndims-1];
        }
    } else {
        CV_Assert(!wsize.empty() && wsize.ndims == nspatialdims + 2);
        for (size_t i = 0; i < nspatialdims; i++)
            ksizes[i] = wsize.size[wsize.ndims - nspatialdims + i];
    }

    CV_Assert(convparams.strides.empty() || convparams.strides.size() == nspatialdims);
    CV_Assert(convparams.dilations.empty() || convparams.dilations.size() == nspatialdims);
    CV_Assert(convparams.pads.empty() || convparams.pads.size() == nspatialdims*2);

    for (size_t i = 0; i < nspatialdims; i++) {
        int64_t ksize = ksizes[i];
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
    outsize.C = outsize.size[1]*outsize.size[ndims-1];

    return outsize;
}

std::ostream& ConvParams::dump(std::ostream& strm)
{
    strm << "{ ngroups: " << ngroups << ", ksizes: {";
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

static void calcKernelOffsets2D(int64_t Wi, int64_t Hk, int64_t Wk,
                                int64_t DY, int64_t DX,
                                int* yxtab, int64_t* ofstab)
{
    if (!yxtab && !ofstab)
        return;
    for (int64_t y = 0; y < Hk; y++) {
        for (int64_t x = 0; x < Wk; x++) {
            int64_t k = y*Wk + x;
            int64_t dy = y*DY, dx = x*DX;
            if (yxtab) {
                yxtab[k*2] = (int)dy;
                yxtab[k*2+1] = (int)dx;
            }
            if (ofstab) {
                ofstab[k] = dy*Wi + dx;
            }
        }
    }
}

static ConvState initConvState_(const TensorSize& inpsize, const TensorSize& wsize,
                                const ConvParams& convparams, const Op& activ_,
                                int* yxtab, int64_t* ofstab)
{
    ConvState cs;
    TensorSize outsize = convInferShape(inpsize, convparams);

    int ndims = inpsize.ndims, wdims = wsize.ndims, kdims = (int)convparams.ksizes.size();
    size_t nspatialdims = ndims - 3;

    CV_Assert((size_t)std::max(wdims, kdims) >= nspatialdims);
    CV_Assert(inpsize.layout == LAYOUT_NCHWc);
    CV_Assert(1 <= nspatialdims && nspatialdims <= 2);
    int64_t N = outsize.size[0];
    int64_t C1 = inpsize.size[1], C0 = outsize.size[ndims-1];
    int64_t K1 = wsize.empty() ? C1 : wsize.size[0]/C0;
    int64_t W = outsize.size[ndims-2];
    int64_t H = nspatialdims > 1 ? outsize.size[ndims-3] : 1;
    int64_t Wi = inpsize.size[ndims-2];
    int64_t Hi = nspatialdims > 1 ? inpsize.size[ndims-3] : 1;
    int64_t SY = 1, SX = 1, DY = 1, DX = 1;
    int64_t pad_y0 = 0, pad_x0 = 0, pad_y1 = 0, pad_x1 = 0;

    int64_t Wk = !wsize.empty() ? wsize.size[wdims-1] : convparams.ksizes[kdims-1];
    int64_t Hk = nspatialdims == 1 ? 1 : !wsize.empty() ? wsize.size[wdims-2] : convparams.ksizes[kdims-2];

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

    int64_t inner_y1 = (Hi - (Hk - 1)*DY + pad_y0)/SY;
    int64_t inner_x1 = (Wi - (Wk - 1)*DX + pad_x0)/SX;

    inner_y1 += inner_y1*SY - pad_y0 + (Hk-1)*DY < Hi;
    inner_x1 += inner_x1*SX - pad_x0 + (Wk-1)*DX < Wi;

    inner_y1 = std::min(inner_y1, H);
    inner_x1 = std::min(inner_x1, W);

    if (inner_y0 >= inner_y1 || inner_x0 >= inner_x1) {
        inner_y0 = H;
        inner_x0 = W;
    }

    calcKernelOffsets2D(Wi, Hk, Wk, DY, DX, yxtab, ofstab);

    cs.Hk = Hk;
    cs.Wk = Wk;
    cs.SY = SY;
    cs.SX = SX;
    cs.DY = DY;
    cs.DX = DX;

    cs.pad_y0 = pad_y0;
    cs.pad_x0 = pad_x0;
    cs.pad_y1 = pad_y1;
    cs.pad_x1 = pad_x1;

    cs.N = N;
    cs.ngroups = convparams.ngroups > 0 ? convparams.ngroups : C1*C0;
    cs.K1 = K1;
    cs.C1 = C1;
    cs.C0 = C0;
    cs.C = inpsize.C;
    cs.H = H;
    cs.W = W;
    cs.Hi = Hi;
    cs.Wi = Wi;

    cs.inner_y0 = inner_y0;
    cs.inner_x0 = inner_x0;
    cs.inner_y1 = inner_y1;
    cs.inner_x1 = inner_x1;

    cs.yxtab = yxtab;
    cs.ofstab = ofstab;

    cs.activation = nullptr;
    cs.fastActivation = ACTIV_NONE;

    ElemwiseOp* activ;
    if (activ_ && (activ = dynamic_cast<ElemwiseOp*>(activ_.get())) != 0) {
        cs.activation = activ->getActivation(CV_32F);
        CV_Assert(cs.activation != nullptr);
        memcpy(cs.activParams, activ->params, ElemwiseOp::MAX_PARAMS*sizeof(activ->params[0]));
        if (activ->opcode == ELWISE_RELU) {
            cs.fastActivation = ACTIV_RELU;
            cs.activation = nullptr;
        } else if (activ->opcode == ELWISE_LRELU) {
            cs.fastActivation = ACTIV_LEAKY_RELU;
            cs.activation = nullptr;
        } else if (activ->opcode == ELWISE_CLIP && cs.activParams[0] == 0.f) {
            cs.fastActivation = ACTIV_CLIP;
            cs.activation = nullptr;
        }
    }

    return cs;
}

ConvState initConvState(const TensorSize& inpsize, const TensorSize& wsize,
                        const ConvParams& convparams, const Op& activ,
                        int* yxtab, int64_t* ofstab)
{
    return initConvState_(inpsize, wsize, convparams, activ, yxtab, ofstab);
}

ConvState initPoolingState(const TensorSize& inpsize, const ConvParams& convparams,
                           int* yxtab, int64_t* ofstab)
{
    return initConvState_(inpsize, TensorSize(), convparams, Op(), yxtab, ofstab);
}

std::ostream& ConvState::dump(std::ostream& strm)
{
    strm << "{N=" << N << ", C1=" << C1 << ", C0=" << C0 << ", K1=" << K1 << ", ngroups=" << ngroups;
    strm << ", H=" << H << ", W=" << W << ", Hi=" << Hi << ", Wi=" << Wi;
    strm << ", Hk=" << Hk << ", Wk=" << Wk << ", SY=" << SY << ", SX=" << SX;
    strm << ", DY=" << DY << ", DX=" << DX;
    strm << ", pad_y0=" << pad_y0 << ", pad_x0=" << pad_x0;
    strm << ", pad_y1=" << pad_y1 << ", pad_x1=" << pad_x1;
    strm << "}";
    return strm;
}

bool ConvState::sameShape(const ConvState& cs) const
{
    return N == cs.N && C1 == cs.C1 && C0 == cs.C0 && K1 == cs.K1 &&
        ngroups == cs.ngroups && H == cs.H && W == cs.W && Hi == cs.Hi && Wi == cs.Wi &&
        Hk == cs.Hk && Wk == cs.Wk && SY == cs.SY && SX == cs.SX && DY == cs.DY && DX == cs.DX &&
        pad_y0 == cs.pad_y0 && pad_x0 == cs.pad_x0 && pad_y1 == cs.pad_y1 && pad_x1 == cs.pad_x1;
}

void serial_for_(const Range& r, std::function<void (const Range&)> body, double)
{
    body(r);
}

}}
