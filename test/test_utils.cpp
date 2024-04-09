// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_utils.hpp"
#include <math.h>

namespace cv { namespace dnn {

TensorSize ref_conv_infer_shapes(const TensorSize& inpsize, const ConvParams& convparams, const TensorSize& wsize)
{
    CV_Assert(inpsize.layout == LAYOUT_NCHWc);

    int ndims = inpsize.ndims;
    size_t nspatialdims = (size_t)(ndims - 3);
    TensorSize outsize = inpsize;
    int64_t ksizes[TensorSize::MAX_DIMS];
    int64_t C0 = inpsize.size[ndims-1];

    if (!convparams.ksizes.empty()) {
        size_t kdims = convparams.ksizes.size();
        CV_Assert(kdims == nspatialdims || kdims == nspatialdims+2);
        for (size_t i = 0; i < nspatialdims; i++)
            ksizes[i] = convparams.ksizes[kdims - nspatialdims + i];
        if (kdims == nspatialdims + 2) {
            CV_Assert(convparams.ksizes[0] % C0 == 0);
            outsize.size[1] = convparams.ksizes[0]/C0;
        }
    } else {
        size_t wdims = wsize.ndims;
        CV_Assert(wdims == nspatialdims+2);
        for (size_t i = 0; i < nspatialdims; i++)
            ksizes[i] = wsize.size[wdims - nspatialdims + i];
        CV_Assert(wsize.size[0] % C0 == 0);
        outsize.size[1] = wsize.size[0]/C0;
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

}}
