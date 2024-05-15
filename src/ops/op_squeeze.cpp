// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"
#include <math.h>

namespace cv { namespace dnn {

SqueezeOp::~SqueezeOp() {}

class SqueezeOpImpl : public SqueezeOp
{
public:
    SqueezeOpImpl()
    {
    }
    virtual std::string_view name() const CV_OVERRIDE { return "Squeeze"; }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<SqueezeOpImpl>();
    }

    virtual int minNumInputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumInputs() const CV_OVERRIDE { return 2; }
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

    TensorSize inferShapes_(const TensorSize& inpsize, const Tensor& axes) const
    {
        int ndims = inpsize.ndims;
        int axisbuf[TensorSize::MAX_DIMS];
        bool axismask[TensorSize::MAX_DIMS];

        CV_Assert(inpsize.layout != LAYOUT_NCHWc);

        int naxes = normalizeAxes(axes, ndims, axisbuf, axismask);
        if (naxes == 0) {
            for (int i = 0; i < ndims; i++)
                axismask[i] = inpsize.size[i] == 1;
        }

        TensorSize outsize;
        int j = 0;
        for (int i = 0; i < ndims; i++) {
            if (axismask[i]) {
                if (inpsize.size[i] != 1) {
                    CV_Error(Error::StsError, "dimension with size!=1 cannot be squeezed");
                }
            } else {
                outsize.size[j++] = inpsize.size[j];
            }
        }

        outsize.ndims = j;
        outsize.layout = LAYOUT_UNKNOWN;

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
        Tensor no_axes;
        const Tensor& axes = inpargs.size() > 1 ? net.argTensor(inpargs[1]) : no_axes;
        outst[0].size = inferShapes_(inpsize, axes);
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
        TensorSize outsize = inferShapes_(inpsize, inputs.size() > 1 ? inputs[1] : Tensor());
        outputs.resize(1);
        inp.reshape(outsize).copyTo(outputs[0]);
    }
};

Op SqueezeOp::create()
{
    return std::make_shared<SqueezeOpImpl>();
}

Arg squeeze(Graph& graph, std::string_view opname,
            std::string_view outname, Arg input, Arg axes)
{
    Op op = SqueezeOp::create();
    std::vector<Arg> inputs = {input};
    if (!axes.empty())
        inputs.push_back(axes);
    return graph->append(opname, op, outname, inputs);
}

}}
