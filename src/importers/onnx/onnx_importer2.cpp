// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#include <opencv2/dnn/layer_reg.private.hpp>
#include <opencv2/core/utils/fp_control_utils.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include <opencv2/core/utils/logger.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <string>

#include "../../engine/engine.hpp"
#include "opencv-onnx-pb-c.h"

namespace cv {
namespace dnn {

using std::vector;
using std::string;
using std::string_view;

static void onnxParseError(string_view ctx, string_view msg)
{
    throw std::runtime_error((!ctx.empty() ? " " + string(ctx) : "") + ": " + string(msg));
}

#define OnnxAssert(ctx, expr) if (!!(expr)) ; else onnxParseError(ctx, "assertion '" #expr "' is invalid")

static string onnxConcatCtx(string_view ctx, string_view subctx)
{
    return ctx.empty() ? string(subctx) : string(ctx) + ", " + string(subctx);
}

class OnnxImporter2
{
public:
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;
    OnnxImporter2(Net2& net, string_view fileName);
    OnnxImporter2(Net2& net, const char* buffer, size_t bufsize);

protected:
    void init(Net2& net);
    bool parse(string_view fileName);
    bool parse(const char* buffer, size_t bufsize);
    Graph parseGraph(const OpenCVOnnx__GraphProto* proto, bool subgraph);
    typedef void (OnnxImporter2::*NodeParser)
        (string_view ctx, const OpenCVOnnx__NodeProto*,
         Graph& graph, const vector<Arg>&, const vector<Arg>&);

    Net2* net;
    typedef std::unordered_map<string, NodeParser> DispatchMap;
    typedef std::unordered_map<string, DispatchMap> DomainDispatchMap;
    DomainDispatchMap alldispatch;
    string filename;
    string defaultOnnxDomain = "ai.onnx";
    string frameworkName;
    std::set<string> unsupportedOps;

    void parseIf(string_view, const OpenCVOnnx__NodeProto*,
                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseLoop(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseScan(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);

    void parseBatchNormalization(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseCast(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseClip(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseConcat(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseConstant(string_view, const OpenCVOnnx__NodeProto*,
                       Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseConstantOfShape(string_view, const OpenCVOnnx__NodeProto*,
                              Graph&, const vector<Arg>&, const vector<Arg>&);
    ConvParams parseConvParams(string_view ctx, const OpenCVOnnx__NodeProto* node_proto);
    void parseConv(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseConvTranspose(string_view, const OpenCVOnnx__NodeProto*,
                            Graph&, const vector<Arg>&, const vector<Arg>&);
    /*void parseCumSum(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseDepthToSpace(string_view, const OpenCVOnnx__NodeProto*,
                           Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseDetectionOutput(string_view, const OpenCVOnnx__NodeProto*,
                              Graph&, const vector<Arg>&, const vector<Arg>&);*/
    void parseDropout(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseElemwiseBinary(string_view, const OpenCVOnnx__NodeProto*,
                             Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseElemwiseNary(string_view, const OpenCVOnnx__NodeProto*,
                           Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseElemwiseUnary(string_view, const OpenCVOnnx__NodeProto*,
                            Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseExpand(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseFlatten(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseGather(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseGemm(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseGlobalPool(string_view, const OpenCVOnnx__NodeProto*,
                         Graph&, const vector<Arg>&, const vector<Arg>&);
    /*void parseGRU(string_view, const OpenCVOnnx__NodeProto*,
                  Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseImageScaler(string_view, const OpenCVOnnx__NodeProto*,
                          Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseInstanceNormalization(string_view, const OpenCVOnnx__NodeProto*,
                                    Graph&, const vector<Arg>&, const vector<Arg>&);*/
    void parseLRN(string_view, const OpenCVOnnx__NodeProto*,
                  Graph&, const vector<Arg>&, const vector<Arg>&);
    /*void parseLSTM(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);*/
    void parseMatMul(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseMaxPool(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    /*void parseMaxUnpool(string_view, const OpenCVOnnx__NodeProto*,
                        Graph&, const vector<Arg>&, const vector<Arg>&);
    void parsePad(string_view, const OpenCVOnnx__NodeProto*,
                  Graph&, const vector<Arg>&, const vector<Arg>&);*/
    void parsePooling(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseReduce(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseResize(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseReshape(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseShape(string_view, const OpenCVOnnx__NodeProto*,
                    Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseSlice(string_view, const OpenCVOnnx__NodeProto*,
                    Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseSoftmax(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseSplit(string_view, const OpenCVOnnx__NodeProto*,
                    Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseSqueeze(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseTranspose(string_view, const OpenCVOnnx__NodeProto*,
                        Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseUnsqueeze(string_view, const OpenCVOnnx__NodeProto*,
                        Graph&, const vector<Arg>&, const vector<Arg>&);
    /*void parseUpsample(string_view, const OpenCVOnnx__NodeProto*,
                       Graph&, const vector<Arg>&, const vector<Arg>&);*/

    // Domain: com.microsoft
    // URL: https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md
    /*void parseDequantizeLinear(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearAveragePool(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearConcat(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearConv(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearElemwiseBinary(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearGlobalAveragePool(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearLeakyRelu(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearMatMul(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearSigmoid(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQuantizeLinear(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);*/
};

struct OnnxTensor
{
    string name;
    Tensor t;
};

static int onnxDatatypeToDepth(string_view ctx, int datatype)
{
    int typ =
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED ? -1 :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UINT8 ? CV_8U :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__INT8 ? CV_8S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UINT16 ? CV_16U :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__INT16 ? CV_16S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UINT32 ? CV_32U :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__INT32 ? CV_32S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UINT64 ? CV_64U :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__INT64 ? CV_64S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__BOOL ? CV_Bool :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT ? CV_32F :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE ? CV_64F :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16 ? CV_16F :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16 ? CV_16BF :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64 ? CV_32FC2 :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128 ? CV_64FC2 : -2;
    if (typ < -1)
        onnxParseError(ctx, format("unsupported data_type %d", datatype));
    return typ;
}

static string onnxAttrTypeToString(OpenCVOnnx__AttributeProto__AttributeType tag)
{
    return tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__UNDEFINED ? "UNDEFINED" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT ? "FLOAT" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT ? "INT" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING ? "STRING" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR ? "TENSOR" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH ? "GRAPH" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR ? "SPARSE_TENSOR" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TYPE_PROTO ? "TYPE_PROTO" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS ? "FLOATS" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS ? "INTS" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS ? "STRINGS" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSORS ? "TENSORS" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPHS ? "GRAPHS" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSORS ? "SPARSE_TENSORS" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TYPE_PROTOS ? "TYPE_PROTOS" :
        format("??? (tag=%d)", (int)tag);
}

template<typename fromT, typename toT>
static void onnxParseArray(Net2* net, string_view ctx,
                           fromT** arr, size_t nelems, vector<toT>& result);

template<typename fromT, typename toT> struct OnnxParseElem
{
    OnnxParseElem(Net2*, string_view) {}
    toT parse(const fromT* proto) const { return static_cast<toT>(*proto); }
};

template<> struct OnnxParseElem<OpenCVOnnx__OperatorSetIdProto, OnnxOpSet>
{
    OnnxParseElem(Net2*, string_view) {}
    OnnxOpSet parse(const OpenCVOnnx__OperatorSetIdProto* proto) {
        return std::make_pair(proto->version, proto->domain);
    }
};

template<> struct OnnxParseElem<OpenCVOnnx__TensorShapeProto__Dimension, OnnxTensorDim>
{
    OnnxParseElem(Net2*, string_view ctx_) : ctx(ctx_), idx(-1) {}
    OnnxTensorDim parse(const OpenCVOnnx__TensorShapeProto__Dimension* proto) {
        idx++;
        if (proto->value_case == OPENCV_ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM)
            return OnnxTensorDim(proto->dim_param);
        if (proto->value_case != OPENCV_ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE)
            onnxParseError(ctx, format("unknown type of dimension #%d", idx));
        return OnnxTensorDim(proto->dim_value);
    }
    string ctx;
    int idx;
};

template<> struct OnnxParseElem<OpenCVOnnx__ValueInfoProto, OnnxArgInfo>
{
    OnnxParseElem(Net2* net_, string_view ctx_) : net(net_), ctx(ctx_) {}
    OnnxArgInfo parse(const OpenCVOnnx__ValueInfoProto* proto) {
        OnnxArgInfo arginfo;
        arginfo.name = proto->name;
        string subctx = onnxConcatCtx(ctx,
                format("parsing value info '%s'", proto->name));
        if (proto->type->value_case == OPENCV_ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE) {
            arginfo.type = onnxDatatypeToDepth(subctx, proto->type->tensor_type->elem_type);
            const OpenCVOnnx__TensorShapeProto* shape = proto->type->tensor_type->shape;
            if (shape)
                onnxParseArray(net, subctx, shape->dim, shape->n_dim, arginfo.size);
        } else if (proto->type->value_case == OPENCV_ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE) {
            onnxParseError(subctx, "sequences are not supported");
        } else {
            onnxParseError(subctx, format("unsupported value info tag %d",
                           (int)proto->type->value_case));
        }
        return arginfo;
    }
    Net2* net;
    string ctx;
};

static int64_t unpack_int64(uint8_t* p)
{
    uint64_t x = p[0] | ((uint64_t)p[1]<<8) | ((uint64_t)p[2]<<16) | ((uint64_t)p[3]<<24) |
        ((uint64_t)p[4]<<32) | ((uint64_t)p[5]<<40) | ((uint64_t)p[6]<<48) | ((uint64_t)p[7]<<56);
    return (int64_t)(x);
}

static Tensor onnxParseTensor(string_view ctx,
                              const OpenCVOnnx__TensorProto* tensor_proto)
{
    Tensor tensor;
    int i, n_dims = (int)tensor_proto->n_dims;
    size_t total = 1, elemsize;
    TensorSize shape;
    uchar* data;
    shape.ndims = n_dims;

    for (i = 0; i < n_dims; i++) {
        int64_t size_i = (int64_t)tensor_proto->dims[i];
        shape.size[i] = size_i;
        total *= size_i;
    }
    shape.layout = n_dims <= 2 ? LAYOUT_ND : LAYOUT_NCHW;
    int typ = onnxDatatypeToDepth(ctx, tensor_proto->data_type);
    if (typ < 0)
        onnxParseError(ctx, format("type of tensor '%s' is invalid (=%d)",
                        tensor_proto->name, tensor_proto->data_type));
    elemsize = CV_ELEM_SIZE(typ);
    tensor.fit(shape, typ);
    data = (uchar*)tensor.data();

    if (elemsize == 1 && tensor_proto->raw_data.len == total) {
        memcpy(data, tensor_proto->raw_data.data, total*elemsize);
    } else if (elemsize == 1 && tensor_proto->n_int32_data == total) {
        for(int j = 0; j < total; j++)
            data[j] = (uchar)(tensor_proto->int32_data[j]);
    } else if (elemsize == 4 && tensor_proto->n_float_data == total) {
        memcpy(data, tensor_proto->float_data, total*elemsize);
    } else if (elemsize == 4 &&
               tensor_proto->data_type == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16 &&
               tensor_proto->n_int32_data == total) {
        float* dst = (float*)data;
        for(size_t j = 0; j < total; j++) {
            ushort bits = (ushort)(tensor_proto->int32_data[j]);
            dst[j] = (float)float16_t::fromBits(bits);
        }
    } else if (elemsize == 4 && tensor_proto->raw_data.len == total*4) {
        uint32_t* dst = (uint32_t*)data;
        for(size_t j = 0; j < total; j++) {
            uint8_t* p = tensor_proto->raw_data.data + j*4;
            dst[j] = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
            ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        }
    } else if (elemsize == 4 && tensor_proto->n_int64_data == total) {
        int32_t* dst = (int32_t*)data;
        for(size_t j = 0; j < total; j++) {
            int64_t v = tensor_proto->int64_data[j];
            dst[j] = (int32_t)(v < INT_MIN ? INT_MIN : v > INT_MAX ? INT_MAX : (int)v);
        }
    } else if (elemsize == 4 && tensor_proto->raw_data.len == total*8) {
        int32_t* dst = (int32_t*)data;
        for(size_t j = 0; j < total; j++) {
            uint8_t* p = tensor_proto->raw_data.data + j*8;
            int64_t v = unpack_int64(p);
            dst[j] = (int32_t)(v < INT_MIN ? INT_MIN : v > INT_MAX ? INT_MAX : (int)v);
        }
    } else {
        onnxParseError(ctx, format("unsupported tensor data_type %d", (int)tensor_proto->data_type));
    }
    return tensor;
}

template<> struct OnnxParseElem<OpenCVOnnx__TensorProto, OnnxTensor>
{
    OnnxParseElem(Net2* net_, string_view ctx_) : net(net_), ctx(ctx_) {}
    OnnxTensor parse(const OpenCVOnnx__TensorProto* proto) {
        OnnxTensor tensor;
        tensor.name = proto->name;
        tensor.t = onnxParseTensor(ctx, proto);
        return tensor;
    }
    Net2* net;
    string ctx;
};

template<typename fromT, typename toT>
static void onnxParseArray(Net2* net, string_view ctx,
                           fromT** arr, size_t nelems, vector<toT>& result)
{
    OnnxParseElem<fromT, toT> elemParser(net, ctx);
    result.reserve(nelems);
    for (size_t i = 0; i < nelems; i++)
        result.push_back(elemParser.parse(arr[i]));
}

static bool onnxHaveAttr(string_view ctx,
                         const OpenCVOnnx__NodeProto* node_proto,
                         string_view attr_name)
{
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name)
            return true;
    }
    return false;
}

static Tensor onnxAttrTensor(string_view ctx, const OpenCVOnnx__NodeProto* node_proto,
                             string_view attr_name, const Tensor& defval=Tensor(),
                             bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR ||
                tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT ||
                tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS ||
                tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT ||
                tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS) {
                if (have_attr)
                    *have_attr = true;
                if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR)
                    return onnxParseTensor(ctx, attr_proto->t);
                if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT)
                    return Tensor::makeScalar(CV_64S, &attr_proto->i);
                if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT)
                    return Tensor::makeScalar(CV_32F, &attr_proto->f);
                if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS)
                    return Tensor::makeVector(CV_64S, attr_proto->ints, attr_proto->n_ints);
                if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS)
                    return Tensor::makeVector(CV_32F, attr_proto->floats, attr_proto->n_floats);
            }
            onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be TENSOR, INT, FLOAT, INTS or FLOATS)",
                            onnxAttrTypeToString(tag).c_str(), attr_proto->name));
        }
    }
    return defval;
}


template<typename _Tp>
static int onnxAttrInt(string_view ctx,
                       const OpenCVOnnx__NodeProto* node_proto,
                       string_view attr_name,
                       _Tp defval, bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT) {
                if (have_attr)
                    *have_attr = true;
                return (_Tp)attr_proto->i;
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be INT)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
    return defval;
}

static float onnxAttrFloat(string_view ctx,
                            const OpenCVOnnx__NodeProto* node_proto,
                            string_view attr_name,
                            float defval, bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT) {
                if (have_attr)
                    *have_attr = true;
                return attr_proto->f;
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be FLOAT)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
    return defval;
}

static string onnxAttrString(string_view ctx,
                              const OpenCVOnnx__NodeProto* node_proto,
                              string_view attr_name,
                              string_view defval, bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT) {
                if (have_attr)
                    *have_attr = true;
                return string((const char*)attr_proto->s.data, attr_proto->s.len);
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be STRING)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
    return string(defval);
}

template<typename _Tp>
static void onnxAttrInts(string_view ctx,
                          const OpenCVOnnx__NodeProto* node_proto,
                          string_view attr_name,
                          vector<_Tp>& values,
                          bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    values.clear();
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS) {
                if (have_attr)
                    *have_attr = true;
                size_t i, n = attr_proto->n_ints;
                values.resize(n);
                for (i = 0; i < n; i++)
                    values[i] = (_Tp)attr_proto->ints[i];
                return;
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be INTS)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
}

static void onnxAttrFloats(string_view ctx,
                          const OpenCVOnnx__NodeProto* node_proto,
                          string_view attr_name,
                          vector<float>& values,
                          bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    values.clear();
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS) {
                if (have_attr)
                    *have_attr = true;
                size_t i, n = attr_proto->n_floats;
                values.resize(n);
                for (i = 0; i < n; i++)
                    values[i] = attr_proto->floats[i];
                return;
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be INTS)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
}

static void onnxAttrStrings(string_view ctx,
                          const OpenCVOnnx__NodeProto* node_proto,
                          string_view attr_name,
                          vector<string>& values,
                          bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    values.clear();
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS) {
                if (have_attr)
                    *have_attr = true;
                size_t i, n = attr_proto->n_strings;
                values.resize(n);
                for (i = 0; i < n; i++)
                    values[i] = string((const char*)attr_proto->strings[i].data,
                                       attr_proto->strings[i].len);
                return;
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be INTS)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
}

static const OpenCVOnnx__GraphProto* onnxAttrGraph(string_view ctx,
                          const OpenCVOnnx__NodeProto* node_proto,
                          string_view attr_name)
{
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH) {
                return attr_proto->g;
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be GRAPH)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
    return nullptr;
}

Graph OnnxImporter2::parseGraph(const OpenCVOnnx__GraphProto* proto, bool subgraph)
{
    string ctx = subgraph ? string() : "parsing subgraph '" + string(proto->name) + "'";
    vector<OnnxArgInfo> inputs, outputs, values;
    vector<OnnxTensor> initializers;
    vector<Arg> node_inputs, node_outputs;

    onnxParseArray(net, ctx, proto->input, proto->n_input, inputs);
    onnxParseArray(net, ctx, proto->output, proto->n_output, outputs);
    onnxParseArray(net, ctx, proto->value_info, proto->n_value_info, values);
    onnxParseArray(net, ctx, proto->initializer, proto->n_initializer, initializers);

    for (const OnnxTensor& t: initializers)
        net->newConstArg(t.name, t.t);

    for (int k = 0; k < 3; k++) {
        ArgKind argkind = k == 0 ? DNN_ARG_INPUT : k == 1 ? DNN_ARG_OUTPUT : DNN_ARG_TEMP;
        if (subgraph) argkind = DNN_ARG_TEMP;
        const vector<OnnxArgInfo>* graph_args = k == 0 ? &inputs : k == 1 ? &outputs : &values;
        for (const OnnxArgInfo& arginfo: *graph_args) {
            Arg arg = net->newArg(arginfo.name, argkind);
            if (k == 0)
                node_inputs.push_back(arg);
            else if (k == 1)
                node_outputs.push_back(arg);
        }
    }

    Graph graph = net->newGraph(proto->name, node_inputs);
    graph->setOutputs(node_outputs);

    for (size_t i = 0; i < proto->n_node; i++) {
        OpenCVOnnx__NodeProto* node_proto = proto->node[i];
        //printf("(%d/%d). parsing %s (%s)\n", (int)i, (int)proto->n_node, node_proto->name, node_proto->op_type);
        string node_ctx = onnxConcatCtx(ctx, format("when parsing '%s' (%s)",
                                        node_proto->name, node_proto->op_type));

        node_inputs.clear();
        node_outputs.clear();

        for (size_t j = 0; j < node_proto->n_input; j++) {
            string_view input_name = node_proto->input[j];
            if (!net->haveArg(input_name)) {
                onnxParseError(node_ctx, format("cannot find input '%s'", input_name.data()));
            }
            node_inputs.push_back(net->getArg(input_name));
        }

        for (size_t j = 0; j < node_proto->n_output; j++) {
            Arg out = net->getArg(node_proto->output[j]);
            node_outputs.push_back(out);
        }

        auto domain_it = alldispatch.find(node_proto->domain);
        bool ok = false;
        if (domain_it != alldispatch.end()) {
            auto& dispatch = domain_it->second;
            auto it = dispatch.find(node_proto->op_type);
            if (it != dispatch.end()) {
                (this->*(it->second))(node_ctx, node_proto, graph, node_inputs, node_outputs);
                ok = true;
            }
        }
        if (!ok)
            unsupportedOps.insert(node_proto->op_type);
    }

    return graph;
}

OnnxImporter2::OnnxImporter2(Net2& net, string_view fileName)
{
    init(net);
    parse(fileName);
}

OnnxImporter2::OnnxImporter2(Net2& net, const char* buffer, size_t bufsize)
{
    init(net);
    parse(buffer, bufsize);
}

bool OnnxImporter2::parse(string_view filename_)
{
    size_t fsize, freaded;
    AutoBuffer<char> buf;
    char* bufptr;

    filename = filename_;
    FILE* f = fopen(filename_.data(), "rb");

    if (!f) {
        CV_LOG_DEBUG(NULL, "DNN/Onnx: cannot open file " << filename_);
        return false;
    }
    fseek(f, 0, SEEK_END);
    fsize = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    buf.allocate(fsize+256);
    bufptr = buf.data();
    freaded = fread(bufptr, 1, fsize, f);
    fclose(f);
    if (freaded != fsize) {
        CV_LOG_DEBUG(NULL, "DNN/Onnx: cannot read file " << filename_);
        return false;
    }
    return parse(bufptr, freaded);
}

bool OnnxImporter2::parse(const char* buffer, size_t datasize)
{
    bool ok = true;
    if (datasize == 0) {
        CV_LOG_DEBUG(NULL, "DNN/Onnx: the file/buffer is empty");
        return false;
    }

    {
        const uint8_t* data = (const uint8_t*)buffer;
        OpenCVOnnx__ModelProto* model = opencv_onnx__model_proto__unpack(0, datasize, data);
        if(model == 0) {
            CV_LOG_DEBUG(NULL, "DNN/Onnx: could not parse the model" << (filename.empty() ? " " + filename : ""));
            return false;
        }

        OnnxInfo info;
        info.IRVersion = model->ir_version;
        info.producer = model->producer_name;
        info.domain = model->domain;
        info.docString = model->doc_string;

        onnxParseArray(net, string(), model->opset_import, model->n_opset_import, info.opsets);
        net->setOnnxInfo(info);

        if (model->producer_name)
            frameworkName = model->producer_name;
        try {
            parseGraph(model->graph, false);
        } catch (const std::exception& exn) {
            string filectx = filename.empty() ? "" : " (" + filename + ") ";
            CV_LOG_WARNING(NULL, "DNN/Onnx: parse error" + filectx + exn.what());
            ok = false;
        }
        opencv_onnx__model_proto__free_unpacked(model, 0);
    }

    if (!unsupportedOps.empty()) {
        std::stringstream msg;
        bool plural = unsupportedOps.size() > 1, first = true;
        msg << string("DNN/Onnx: the operation") + (plural ? "s " : " ");
        for (string_view opname: unsupportedOps) {
            msg << (first ? "'" : ", '") + string(opname) + "'";
            first = false;
        }
        msg << (plural ? " are not supported" : " is not supported");
        CV_LOG_ERROR(NULL, msg.str());
        ok = false;
    }
    if (!ok) net->release();
    return ok;
}

void OnnxImporter2::init(Net2& net)
{
    DispatchMap dispatch, msdispatch;

    //dispatch["ArgMax"] = dispatch["ArgMin"] = &OnnxImporter2::parseArg;
    dispatch["AveragePool"] = &OnnxImporter2::parsePooling;
    dispatch["BatchNormalization"] = &OnnxImporter2::parseBatchNormalization;
    dispatch["Cast"] = &OnnxImporter2::parseCast;
    dispatch["Clip"] = &OnnxImporter2::parseClip;
    dispatch["Concat"] = &OnnxImporter2::parseConcat;
    dispatch["Constant"] = &OnnxImporter2::parseConstant;
    dispatch["ConstantFill"] = dispatch["ConstantOfShape"] = &OnnxImporter2::parseConstantOfShape;
    dispatch["Conv"] = &OnnxImporter2::parseConv;
    dispatch["ConvTranspose"] = &OnnxImporter2::parseConvTranspose;
    //dispatch["CumSum"] = &OnnxImporter2::parseCumSum;
    //dispatch["DetectionOutput"] = &OnnxImporter2::parseDetectionOutput;
    dispatch["Dropout"] = &OnnxImporter2::parseDropout;
    dispatch["Expand"] = &OnnxImporter2::parseExpand;
    dispatch["Flatten"] = &OnnxImporter2::parseFlatten;
    dispatch["Gather"] = &OnnxImporter2::parseGather;
    dispatch["Gemm"] = &OnnxImporter2::parseGemm;
    dispatch["GlobalAveragePool"] = dispatch["GlobalMaxPool"] = &OnnxImporter2::parseGlobalPool;
    //dispatch["GRU"] = &OnnxImporter2::parseGRU;
    //dispatch["ImageScaler"] = &OnnxImporter2::parseImageScaler;
    //dispatch["InstanceNormalization"] = &OnnxImporter2::parseInstanceNormalization;
    dispatch["LRN"] = &OnnxImporter2::parseLRN;
    //dispatch["LSTM"] = &OnnxImporter2::parseLSTM;
    dispatch["MatMul"] = &OnnxImporter2::parseMatMul;

    dispatch["MaxPool"] = &OnnxImporter2::parsePooling;
    //dispatch["MaxUnpool"] = &OnnxImporter2::parseMaxUnpool;
    //dispatch["Pad"] = &OnnxImporter2::parsePad;
    //dispatch["PRelu"] = &OnnxImporter2::parsePRelu;
    dispatch["Reshape"] = &OnnxImporter2::parseReshape;
    dispatch["Resize"] = &OnnxImporter2::parseResize;
    dispatch["Shape"] = &OnnxImporter2::parseShape;
    dispatch["Slice"] = &OnnxImporter2::parseSlice;
    dispatch["Softmax"] = &OnnxImporter2::parseSoftmax;
    //dispatch["SpaceToDepth"] = dispatch["DepthToSpace"] = &OnnxImporter2::parseDepthToSpace;
    dispatch["Split"] = &OnnxImporter2::parseSplit;
    dispatch["Squeeze"] = &OnnxImporter2::parseSqueeze;
    dispatch["Transpose"] = &OnnxImporter2::parseTranspose;
    //dispatch["Upsample"] = &OnnxImporter2::parseUpsample;
    dispatch["Unsqueeze"] = &OnnxImporter2::parseUnsqueeze;

    vector<string> elemwiseUnaryOps {
        "Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh",
        "Ceil", "Celu", "Cos", "Cosh", "Elu", "Erf",
        "Exp", "Floor", "HardSigmoid", "HardSwish", "Identity",
        "LeakyRelu", "Log", "Neg", "Relu", "Round", "Reciprocal", "Selu",
        "Sign", "Sigmoid", "Sin", "Sinh", "Softplus",
        "Softsign", "Shrink", "Sqrt", "Tan", "Tanh", "ThresholdedRelu" };
    for (const auto& name : elemwiseUnaryOps)
        dispatch[name] = &OnnxImporter2::parseElemwiseUnary;

    vector<string> elemwiseBinaryOps {
        "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Pow", "Add", "Sub", "Mul", "Div" };
    for (const auto& name : elemwiseBinaryOps)
        dispatch[name] = &OnnxImporter2::parseElemwiseBinary;

    vector<string> elemwiseNaryOps {
        "Max", "Min", "Sum" };
    for (const auto& name : elemwiseNaryOps)
        dispatch[name] = &OnnxImporter2::parseElemwiseNary;

    vector<string> reduceOps {
        "ReduceMax", "ReduceMin", "ReduceMean", "ReduceSum", "ReduceSumSquare",
        "ReduceProd", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp" };
    for (const auto& name : reduceOps)
        dispatch[name] = &OnnxImporter2::parseReduce;

    /*
    // ai.onnx: opset 10+
    dispatch["DequantizeLinear"] = &OnnxImporter2::parseDequantizeLinear;
    dispatch["QLinearConv"] = &OnnxImporter2::parseQLinearConv;
    dispatch["QLinearMatMul"] = &OnnxImporter2::parseQLinearMatMul;
    dispatch["QuantizeLinear"] = &OnnxImporter2::parseQuantizeLinear;*/
    alldispatch[defaultOnnxDomain] = dispatch;
    alldispatch[""] = dispatch;

    /*msdispatch["QLinearAdd"] = &OnnxImporter2::parseQLinearElemwiseBinary;
    msdispatch["QLinearAveragePool"] = &OnnxImporter2::parsePooling;
    msdispatch["QLinearGlobalAveragePool"] = &OnnxImporter2::parseQLinearGlobalAveragePool;
    msdispatch["QLinearConcat"] = &OnnxImporter2::parseQLinearConcat;
    msdispatch["QLinearLeakyRelu"] = &OnnxImporter2::parseQLinearLeakyRelu;
    msdispatch["QLinearSigmoid"] = &OnnxImporter2::parseQLinearSigmoid;
    alldispatch["com.microsoft"] = msdispatch;
     */
}

void OnnxImporter2::parseBatchNormalization(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                            const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t noutputs = outputs.size();
    OnnxAssert(ctx, noutputs == 1);
    if (inputs.size() != 5)
        onnxParseError(ctx, "Expected input, scale, bias, mean and var");

    double epsilon = onnxAttrFloat(ctx, node_proto, "epsilon", 1e-5f);

    batchNorm(graph, node_proto->name, node_proto->output[0],
              inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
              epsilon);
}

void OnnxImporter2::parseCast(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                              const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    int datatype = onnxAttrInt(ctx, node_proto, "to", -1);
    OnnxAssert(ctx, datatype >= 0);
    int typ = onnxDatatypeToDepth(ctx, datatype);

    cast(graph, node_proto->name, node_proto->output[0], inputs[0], typ);
}

void OnnxImporter2::parseClip(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                              const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 1 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);

    Arg minval, maxval;
    if (ninputs == 3) {
        minval = inputs[1];
        maxval = inputs[2];
    } else {
        std::string node_name = node_proto->name;
        minval = net->newConstScalarArg(node_name + ".min", onnxAttrFloat(ctx, node_proto, "min", 0.f));
        maxval = net->newConstScalarArg(node_name + ".max", onnxAttrFloat(ctx, node_proto, "max", 6.f));
    }
    clip(graph, node_proto->name, node_proto->output[0], inputs[0], minval, maxval);
}

void OnnxImporter2::parseConcat(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() >= 1);
    OnnxAssert(ctx, outputs.size() == 1);

    bool haveAxis = false;
    int axis = onnxAttrInt(ctx, node_proto, "axis", -1, &haveAxis);
    OnnxAssert(ctx, haveAxis);

    concat(graph, node_proto->name, node_proto->output[0], inputs, axis);
}


void OnnxImporter2::parseConstant(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                  const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    Tensor t;
    TensorSize scalar_shape;
    OnnxAssert(ctx, inputs.size() == 0);
    OnnxAssert(ctx, outputs.size() == 1);
    const char* names[] = {"value", "value_int", "value_float", "value_ints", "value_floats", nullptr};

    for (int i = 0; names[i] != nullptr; i++) {
        if (onnxHaveAttr(ctx, node_proto, names[i])) {
            t = onnxAttrTensor(ctx, node_proto, names[i]);
            break;
        }
    }

    if (t.empty()) {
        onnxParseError(ctx, "invalid/unsupported 'Constant' layer: must contain a tensor, a scalar or a vector of scalars");
    }

    net->newConstArg(string_view(node_proto->output[0]), t);
}

void OnnxImporter2::parseConstantOfShape(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                         const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    
    Tensor value = onnxAttrTensor(ctx, node_proto, "value", Tensor::makeScalar(0.f));
    return constantOfShape(graph, node_proto->name, node_proto->output[0], inputs[0], value);
}


ConvParams OnnxImporter2::parseConvParams(string_view ctx, const OpenCVOnnx__NodeProto* node_proto)
{
    ConvParams params;
    onnxAttrInts(ctx, node_proto, "kernel_shape", params.ksizes);
    onnxAttrInts(ctx, node_proto, "strides", params.strides);
    onnxAttrInts(ctx, node_proto, "dilations", params.dilations);
    params.ngroups = onnxAttrInt(ctx, node_proto, "group", 1);

    if (onnxHaveAttr(ctx, node_proto, "pads")) {
        onnxAttrInts(ctx, node_proto, "pads", params.pads);
    } else {
        string autopad = onnxAttrString(ctx, node_proto, "auto_pad", "NOTSET");
        params.autopad = autopad == "SAME_UPPER" ? AUTOPAD_SAME_UPPER :
                         autopad == "SAME_LOWER" ? AUTOPAD_SAME_LOWER :
                         autopad == "VALID" ? AUTOPAD_VALID : AUTOPAD_NOTSET;
    }
    return params;
};

void OnnxImporter2::parseConv(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                              const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);

    ConvParams params = parseConvParams(ctx, node_proto);
    conv(graph, node_proto->name, node_proto->output[0], inputs[0], inputs[1], ninputs >= 3 ? inputs[2] : Arg(), params);
}

void OnnxImporter2::parseConvTranspose(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                       const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);
    ConvParams params = parseConvParams(ctx, node_proto);

    std::vector<int> output_padding, output_shape;

    onnxAttrInts(ctx, node_proto, "output_padding", output_padding);
    onnxAttrInts(ctx, node_proto, "output_shape", output_shape);

    convTranspose(graph, node_proto->name, node_proto->output[0], inputs[0], inputs[1],
                  ninputs >= 3 ? inputs[2] : Arg(), params, output_padding, output_shape);
}

void OnnxImporter2::parseDropout(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                 const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size(), noutputs = outputs.size();
    OnnxAssert(ctx, 1 <= ninputs && ninputs <= 3);
    OnnxAssert(ctx, 1 <= noutputs && noutputs <= 2);

    Arg ratio;
    if (ninputs >= 2)
        ratio = inputs[1];
    else
        ratio = net->newConstScalarArg(node_proto->name + string(".ratio"),
                                       onnxAttrFloat(ctx, node_proto, "ratio", 0.5f));

    dropout(graph, node_proto->name, node_proto->output[0], inputs[0], ratio);
}

// Equal, Greater, Less, Pow, Add, Sub, Mul, Div, ...
void OnnxImporter2::parseElemwiseBinary(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                        const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2 && outputs.size() == 1);
    string_view op = node_proto->op_type;
    ElemwiseOpcode opcode =
        op == "Equal" ? ELWISE_EQUAL :
        op == "Greater" ? ELWISE_GREATER :
        op == "GreaterOrEqual" ? ELWISE_GREATER_EQUAL :
        op == "Less" ? ELWISE_LESS :
        op == "LessOrEqual" ? ELWISE_LESS_EQUAL :
        op == "Add" ? ELWISE_ADD :
        op == "Sub" ? ELWISE_SUB :
        op == "Mul" ? ELWISE_MUL :
        op == "Div" ? ELWISE_DIV :
        op == "Mod" ? ELWISE_MOD :
        op == "And" ? ELWISE_AND :
        op == "Or" ? ELWISE_OR :
        op == "Xor" ? ELWISE_XOR :
        ELWISE_NONE;

    if (opcode == ELWISE_NONE) {
        onnxParseError(ctx, format("invalid/unsupported binary operator '%s'", op.data()));
    }

    elemwise(graph, node_proto->name, node_proto->output[0], opcode, inputs[0], inputs[1]);
}

// "Sum" "Min" "Max"
void OnnxImporter2::parseElemwiseNary(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                      const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() >= 2 && outputs.size() == 1);
    string_view op = node_proto->op_type;
    ElemwiseOpcode opcode =
        op == "Max" ? ELWISE_MAX :
        op == "Min" ? ELWISE_MIN :
        op == "Sum" ? ELWISE_SUM :
        op == "Prod" ? ELWISE_MUL :
        ELWISE_NONE;

    if (opcode == ELWISE_NONE) {
        onnxParseError(ctx, format("invalid/unsupported binary operator '%s'", op.data()));
    }

    elemwise(graph, node_proto->name, node_proto->output[0], opcode, inputs);
}

void OnnxImporter2::parseElemwiseUnary(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                       const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    string_view op = node_proto->op_type;
    ElemwiseOpcode opcode =
    op == "Abs" ? ELWISE_ABS :
    op == "Acos" ? ELWISE_ACOS :
    op == "Acosh" ? ELWISE_ACOSH :
    op == "Asin" ? ELWISE_ASIN :
    op == "Asinh" ? ELWISE_ASINH :
    op == "Atan" ? ELWISE_ATAN :
    op == "Atanh" ? ELWISE_ATANH :
    op == "Ceil" ? ELWISE_CEIL :
    //op == "Celu" ? ELWISE_CELU :
    op == "Cos" ? ELWISE_COS :
    op == "Cosh" ? ELWISE_COSH :
    //op == "Elu" ? ELWISE_ELU :
    op == "Erf" ? ELWISE_ERF :
    op == "Exp" ? ELWISE_EXP :
    op == "Floor" ? ELWISE_FLOOR :
    //op == "HardSigmoid" ? ELWISE_HARD_SIGMOID :
    //op == "HardSwish" ? ELWISE_SWISH :
    //op == "Identity" ? ELWISE_IDENTITY :
    op == "LeakyRelu" ? ELWISE_LRELU :
    op == "Log" ? ELWISE_LOG :
    op == "Neg" ? ELWISE_NEG :
    op == "Relu" ? ELWISE_RELU :
    op == "Round" ? ELWISE_ROUND :
    op == "Reciprocal" ? ELWISE_RECIP :
    //op == "Selu" ? ELWISE_SELU :
    op == "Sign" ? ELWISE_SIGN :
    op == "Sigmoid" ? ELWISE_SIGMOID :
    op == "Sin" ? ELWISE_SIN :
    op == "Sinh" ? ELWISE_SINH :
    op == "Softplus" ? ELWISE_SOFTPLUS :
    op == "Softsign" ? ELWISE_SOFTSIGN :
    //op == "Shrink" ? ELWISE_SHRINK :
    op == "Sqrt" ? ELWISE_SQRT :
    op == "Tan" ? ELWISE_TAN :
    op == "Tanh" ? ELWISE_TANH :
    //op == "ThresholdedRelu" ? ELWISE_THRESHOLDED_RELU :
    ELWISE_NONE;

    float params[ElemwiseOp::MAX_PARAMS];
    size_t nparams = 0;

    if (opcode == ELWISE_NONE && op != "Identity") {
        onnxParseError(ctx, format("invalid/unsupported binary operator '%s'", op.data()));
    }

    if (opcode == ELWISE_LRELU) {
        params[nparams++] = onnxAttrFloat(ctx, node_proto, "alpha", 0.01f);
    }

    elemwise(graph, node_proto->name, node_proto->output[0], opcode, inputs[0], params, nparams);
}

void OnnxImporter2::parseExpand(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2 && outputs.size() == 1);

    expand(graph, node_proto->name, node_proto->output[0], inputs[0], inputs[1]);
}

void OnnxImporter2::parseFlatten(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                 const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    int axis = onnxAttrInt(ctx, node_proto, "axis", 1);

    flatten(graph, node_proto->name, node_proto->output[0], inputs[0], axis);
}

void OnnxImporter2::parseGather(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2);
    OnnxAssert(ctx, outputs.size() == 1);
    int axis = onnxAttrInt(ctx, node_proto, "axis", 0);

    gather(graph, node_proto->name, node_proto->output[0], inputs[0], inputs[1], axis);
}

// A * B + C = Y, we require that the dimension of A is [m, k], and the dimension of B is [n, k].
// And the dim of output Y is [m, n]
void OnnxImporter2::parseGemm(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                              const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);

    float alpha = onnxAttrFloat(ctx, node_proto, "alpha", 1.0f);
    float beta = onnxAttrFloat(ctx, node_proto, "beta", 1.0f);

    int transA = onnxAttrInt(ctx, node_proto, "transA", 0);
    int transB = onnxAttrInt(ctx, node_proto, "transB", 0);

    gemm(graph, node_proto->name, node_proto->output[0], inputs[0], inputs[1],
         (ninputs >= 3 ? inputs[2] : Arg()), transA != 0, transB != 0, alpha, beta);
}

void OnnxImporter2::parseGlobalPool(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                    const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    string_view op = node_proto->op_type;

    if (op == "GlobalAveragePool")
        globalAveragePool(graph, node_proto->name, node_proto->output[0], inputs[0]);
    else {
        onnxParseError(ctx, format("invalid/unsupported global pooling op '%s'", op.data()));
    }
}


void OnnxImporter2::parseLRN(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                             const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);

    float alpha = onnxAttrFloat(ctx, node_proto, "alpha", 1e-4f);
    float beta = onnxAttrFloat(ctx, node_proto, "beta", 0.75f);

    float bias = onnxAttrFloat(ctx, node_proto, "transA", 1.0f);
    int size = onnxAttrInt(ctx, node_proto, "transB", -1);

    if (size <= 0) {
        onnxParseError(ctx, "LRN: 'size' attribute is missing or is not positive");
    }

    LRN(graph, node_proto->name, node_proto->output[0], inputs[0], size, alpha, beta, bias);
}

void OnnxImporter2::parseMatMul(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2 && outputs.size() == 1);
    matMul(graph, node_proto->name, node_proto->output[0], inputs[0], inputs[1]);
}

enum PoolingType {
    POOLING_UNKNOWN = 0,
    POOLING_MAXPOOL = 1,
    POOLING_AVGPOOL = 2
};

void OnnxImporter2::parsePooling(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                 const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    string_view op = node_proto->op_type;
    PoolingType pooling = op == "AveragePool" ? POOLING_AVGPOOL :
    op == "MaxPool" ? POOLING_MAXPOOL :
    POOLING_UNKNOWN;

    size_t ninputs = inputs.size(), noutputs = outputs.size();
    OnnxAssert(ctx, ninputs == 1);

    ConvParams params = parseConvParams(ctx, node_proto);

    if (pooling == POOLING_AVGPOOL) {
        OnnxAssert(ctx, noutputs == 1);
        int count_include_pad = onnxAttrInt(ctx, node_proto, "count_include_pad", 0);
        averagePool(graph, node_proto->name, node_proto->output[0], inputs[0], params, count_include_pad != 0);
    }
    else if (pooling == POOLING_MAXPOOL) {
        OnnxAssert(ctx, noutputs == 1 || noutputs == 2);
        int storage_order = onnxAttrInt(ctx, node_proto, "storage_order", 0);
        maxPool(graph, node_proto->name, node_proto->output[0], inputs[0], params, noutputs == 2, storage_order == 0);
    } else {
        onnxParseError(ctx, format("unknown/unsupported pooling op '%s'", op.data()));
    }
}

void OnnxImporter2::parseReduce(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 1 || ninputs == 2);
    OnnxAssert(ctx, outputs.size() == 1);
    string_view op = node_proto->op_type;
    ReduceOpcode opcode =
        op == "ReduceL1" ? REDUCE_L1 :
        op == "ReduceL2" ? REDUCE_L2 :
        op == "ReduceLogSum" ? REDUCE_LOGSUM :
        op == "ReduceLogSumExp" ? REDUCE_LOGSUMEXP :
        op == "ReduceMax" ? REDUCE_MAX :
        op == "ReduceMin" ? REDUCE_MIN :
        op == "ReduceMean" ? REDUCE_MEAN :
        op == "ReduceProd" ? REDUCE_PROD :
        op == "ReduceSum" ? REDUCE_SUM :
        op == "ReduceSumSquare" ? REDUCE_SUM_SQUARE :
        REDUCE_NONE;

    if (opcode == REDUCE_NONE) {
        onnxParseError(ctx, format("unknown/unsupported pooling op '%s'", op.data()));
    }

    Arg axes;
    if (ninputs >= 2)
        axes = inputs[1];
    else
        axes = net->newConstArg(node_proto->name + string(".axes"), onnxAttrTensor(ctx, node_proto, "axes"));

    int keepdims = onnxAttrInt(ctx, node_proto, "keepdims", 1);
    int noopwithemptyaxes = onnxAttrInt(ctx, node_proto, "noop_with_empty_axes", 0);

    reduce(graph, node_proto->name, node_proto->output[0], opcode,
           inputs[0], axes, keepdims != 0, noopwithemptyaxes != 0);
}

void OnnxImporter2::parseReshape(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                 const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 1 || ninputs == 2);
    OnnxAssert(ctx, outputs.size() == 1);

    Arg shape;
    if (ninputs >= 2)
        shape = inputs[1];
    else {
        Tensor t = onnxAttrTensor(ctx, node_proto, "shape");
        if (t.empty()) {
            onnxParseError(ctx, "Reshape: required 'shape' attribute is missing");
        }
        shape = net->newConstArg(node_proto->name + string(".shape"), t);
    }
    int allowzero = onnxAttrInt(ctx, node_proto, "allowzero", 0);

    reshape(graph, node_proto->name, node_proto->output[0], inputs[0], shape, allowzero != 0);
}

void OnnxImporter2::parseResize(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, 1 <= ninputs && ninputs <= 4);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseShape(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                               const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1);
    OnnxAssert(ctx, outputs.size() == 1);

    int start = onnxAttrInt(ctx, node_proto, "start", 0);
    int end = onnxAttrInt(ctx, node_proto, "start", INT_MAX);

    shape(graph, node_proto->name, node_proto->output[0], inputs[0], start, end);
}

void OnnxImporter2::parseSlice(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                               const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 1 || (3 <= ninputs && ninputs <= 5));
    OnnxAssert(ctx, outputs.size() == 1);

    Arg starts, ends, axes, steps;
    string node_name = node_proto->name;

    if (ninputs >= 2) {
        starts = inputs[1];
    } else {
        Tensor t = onnxAttrTensor(ctx, node_proto, "starts");
        if (t.empty()) {
            onnxParseError(ctx, "Slice: required 'starts' attribute is missing");
        }
        starts = net->newConstArg(node_name + ".starts", t);
    }

    if (ninputs >= 3) {
        ends = inputs[2];
    } else {
        Tensor t = onnxAttrTensor(ctx, node_proto, "ends");
        if (t.empty()) {
            onnxParseError(ctx, "Slice: required 'ends' attribute is missing");
        }
        ends = net->newConstArg(node_name + ".ends", t);
    }

    if (ninputs >= 4) {
        axes = inputs[3];
    } else {
        Tensor t = onnxAttrTensor(ctx, node_proto, "axes");
        axes = net->newConstArg(node_name + ".axes", t);
    }

    if (ninputs >= 5) {
        steps = inputs[4];
    } else {
        Tensor t = onnxAttrTensor(ctx, node_proto, "steps");
        steps = net->newConstArg(node_name + ".steps", t);
    }

    slice(graph, node_proto->name, node_proto->output[0], inputs[0], starts, ends, axes, steps);
}

void OnnxImporter2::parseSoftmax(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                 const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1);
    OnnxAssert(ctx, outputs.size() == 1);

    int axis = onnxAttrInt(ctx, node_proto, "axis", -1);
    softmax(graph, node_proto->name, node_proto->output[0], inputs[0], axis);
}

void OnnxImporter2::parseSplit(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                               const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 1 || ninputs == 2);
    OnnxAssert(ctx, outputs.size() >= 1);

    Arg split_vals;
    int axis = onnxAttrInt(ctx, node_proto, "axis", 0);
    size_t num_outputs = onnxAttrInt(ctx, node_proto, "num_outputs", outputs.size());

    OnnxAssert(ctx, outputs.size() == num_outputs);

    if (ninputs > 1) {
        split_vals = inputs[1];
    } else {
        Tensor t = onnxAttrTensor(ctx, node_proto, "split");
        split_vals = net->newConstArg(node_proto->name + string(".split"), t);
    }

    split(graph, node_proto->name, node_proto->output[0], inputs[0], split_vals, axis, num_outputs);
}

void OnnxImporter2::parseSqueeze(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                 const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 1 || ninputs == 2);
    OnnxAssert(ctx, outputs.size() == 1);

    Arg axes;
    if (ninputs > 1)
        axes = inputs[1];
    else {
        Tensor t = onnxAttrTensor(ctx, node_proto, "axes");
        axes = net->newConstArg(node_proto->name + string(".axes"), t);
    }

    squeeze(graph, node_proto->name, node_proto->output[0], inputs[0], axes);
}

void OnnxImporter2::parseTranspose(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                   const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    vector<int> perm;

    onnxAttrInts(ctx, node_proto, "perm", perm);
    if (perm.empty()) {
        onnxParseError(ctx, "Transpose: required 'perm' attribute is missing");
    }

    transpose(graph, node_proto->name, node_proto->output[0], inputs[0], perm);
}

void OnnxImporter2::parseUnsqueeze(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                   const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 1 || ninputs == 2);
    OnnxAssert(ctx, outputs.size() == 1);

    Arg axes;
    if (ninputs > 1)
        axes = inputs[1];
    else {
        Tensor t = onnxAttrTensor(ctx, node_proto, "axes");
        if (t.empty()) {
            onnxParseError(ctx, "Unsqueeze: required 'axes' attribute is missing");
        }
        axes = net->newConstArg(node_proto->name + string(".axes"), t);
    }

    unsqueeze(graph, node_proto->name, node_proto->output[0], inputs[0], axes);
}

Net2 readNetFromONNX2(string_view onnxFile)
{
    Net2 net;
    OnnxImporter2 importer(net, onnxFile);
    if (net.empty())
        return net;
    net.initialize();
    return net;
}

}} // namespace
