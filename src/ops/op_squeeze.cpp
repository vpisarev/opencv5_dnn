// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/net2_impl.hpp"
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

    virtual int minNumInputs() const CV_OVERRIDE { return 2; }
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

    virtual void inferTypes(const Net2& net, const Graph& graph,
                            const std::vector<Arg>& inpargs,
                            const std::vector<int>& inptypes,
                            const std::vector<Arg>& outargs,
                            std::vector<int>& outtypes) const CV_OVERRIDE
    {
        int ninputs = (int)inpargs.size(), noutputs = (int)outargs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert((int)inptypes.size() == ninputs);
        CV_Assert(noutputs == 1);

        outtypes.resize(1);
        outtypes[0] = inferType(inptypes[0]);
    }

    TensorSize inferShapes_(Net2&, const TensorSize& inpsize, const Tensor& axes, bool symbolic) const
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
                if (inpsize.size[i] != 1 && !symbolic) {
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

    virtual void inferShapes(Net2& net, const Graph& graph,
                             const std::vector<Arg>& inpargs,
                             const std::vector<TensorSize>& inpshapes,
                             const std::vector<Arg>& outargs,
                             std::vector<TensorSize>& outshapes,
                             bool symbolic) const CV_OVERRIDE
    {
        int ninputs = (int)inpargs.size(), noutputs = (int)outargs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert(noutputs == 1);
        outshapes.resize(1);

        if (symbolic) {
            CV_Assert(net.isConstArg(inpargs[1]));
        }

        const TensorSize& inpsize = inpshapes[0];
        Tensor no_axes;
        const Tensor& axes = inpargs.size() > 1 ? net.argTensor(inpargs[1]) : no_axes;
        outshapes[0] = inferShapes_(net, inpsize, axes, symbolic);
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
        TensorSize outsize = inferShapes_(net, inpsize, inputs[1], false);
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
