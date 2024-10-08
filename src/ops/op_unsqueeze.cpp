// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/net2_impl.hpp"
#include <math.h>

namespace cv { namespace dnn {

UnsqueezeOp::~UnsqueezeOp() {}

class UnsqueezeOpImpl : public UnsqueezeOp
{
public:
    UnsqueezeOpImpl()
    {
    }
    virtual std::string_view name() const CV_OVERRIDE { return "Unsqueeze"; }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<UnsqueezeOpImpl>();
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

    void inferTypes(const Net2& net, const Graph& graph,
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

    TensorSize inferShapes_(const TensorSize& inpsize, const Tensor& axes) const
    {
        int naxes = (int)axes.total();
        int inp_ndims = inpsize.ndims, out_ndims = inp_ndims + naxes;
        int axisbuf[TensorSize::MAX_DIMS];
        bool axismask[TensorSize::MAX_DIMS];

        CV_Assert(inpsize.layout != LAYOUT_NCHWc);
        CV_Assert(out_ndims <= TensorSize::MAX_DIMS);

        naxes = normalizeAxes(axes, out_ndims, axisbuf, axismask);

        TensorSize outsize;
        int j = 0;
        for (int i = 0; i < out_ndims; i++) {
            if (axismask[i]) {
                outsize.size[i] = 1;
            } else {
                outsize.size[i] = inpsize.size[j++];
            }
        }

        outsize.ndims = out_ndims;
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
        CV_Assert((int)inpshapes.size() == ninputs);
        CV_Assert(noutputs == 1);
        outshapes.resize(1);

        if (symbolic) {
            CV_Assert(net.isConstArg(inpargs[1]));
        }

        const TensorSize& inpsize = inpshapes[0];
        const Tensor& axes = net.argTensor(inpargs[1]);
        outshapes[0] = inferShapes_(inpsize, axes);
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
        TensorSize outsize = inferShapes_(inpsize, inputs[1]);
        outputs.resize(1);
        inp.reshape(outsize).copyTo(outputs[0]);
    }
};

Op UnsqueezeOp::create()
{
    return std::make_shared<UnsqueezeOpImpl>();
}

Arg unsqueeze(Graph& graph, std::string_view opname,
              std::string_view outname, Arg input, Arg axes)
{
    Op op = UnsqueezeOp::create();
    return graph->append(opname, op, outname, {input, axes});
}

}}
