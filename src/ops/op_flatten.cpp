// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"
#include <math.h>

namespace cv { namespace dnn {

FlattenOp::~FlattenOp() {}

class FlattenOpImpl : public FlattenOp
{
public:
    FlattenOpImpl(int axis_)
    {
        axis = axis_;
    }
    virtual std::string_view name() const CV_OVERRIDE { return "Flatten"; }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<FlattenOpImpl>(axis);
    }

    virtual int minNumInputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumInputs() const CV_OVERRIDE { return 1; }
    virtual int minNumOutputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumOutputs() const CV_OVERRIDE { return 1; }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual bool supportType(int, int) const CV_OVERRIDE
    {
        return true;
    }

    virtual bool alwaysSupportInplace() const CV_OVERRIDE
    {
        return true;
    }

    virtual int64_t getFLOPS(const std::vector<SizeType> &inputs,
                             const std::vector<SizeType> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        return (int64_t)inputs[0].size.total();
    }

    TensorSize inferShapes_(const TensorSize& inpsize) const
    {
        int ndims = inpsize.ndims;
        
        CV_Assert(inpsize.layout != LAYOUT_NCHWc);
        CV_Assert(ndims >= 1);

        int i, a = axis == ndims ? axis : normalizeAxis(axis, ndims);

        int64_t sz1 = 1, sz2 = 1;
        for (i = 0; i < a; i++)
            sz1 *= inpsize.size[i];
        for (; i < ndims; i++)
            sz2 *= inpsize.size[i];

        TensorSize outsize;
        outsize.ndims = 2;
        outsize.layout = LAYOUT_ND;
        outsize.size[0] = sz1;
        outsize.size[1] = sz2;

        return outsize;
    }

    virtual void inferShapes(const Net2& net, const Graph& graph,
                            const std::vector<Arg>& inpargs,
                            const std::vector<SizeType>& inpst,
                            const std::vector<Arg>& outargs,
                            std::vector<SizeType>& outst,
                            std::vector<size_t>& tempbufs) const CV_OVERRIDE
    {
        int ninputs = (int)inpargs.size(), noutputs = (int)outargs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert(noutputs == 1);
        outst.resize(1);

        const TensorSize& inpsize = inpst[0].size;
        outst[0].size = inferShapes_(inpsize);
        outst[0].type = inferType(inpst[0].type);
        tempbufs.assign(1, (size_t)0);
    }

    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        std::vector<Buffer>& tempbufs) CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        const Tensor& inp = inputs[0];
        CV_Assert(inp.isContinuous());

        int inptype = inp.type(), outtype = inferType(inptype);
        TensorSize inpsize = inp.size();
        TensorSize outsize = inferShapes_(inpsize);
        outputs.resize(1);
        inp.reshape(outsize).copyTo(outputs[0]);
    }
};

Op FlattenOp::create(int axis)
{
    return std::make_shared<FlattenOpImpl>(axis);
}

}}
