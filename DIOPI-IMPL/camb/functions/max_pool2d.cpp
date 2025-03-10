/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <cstring>
#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

namespace {

std::vector<int> getDim(const DiopiTensor& tensor) {
    int shapeSize = tensor.shape().size();
    std::vector<int> dim(shapeSize);
    for (int i = 0; i < shapeSize; i++) {
        dim[i] = static_cast<int>(tensor.shape()[i]);
    }
    if (shapeSize == 3) {
        dim.insert(dim.begin(), 1);
    }
    return dim;
}

}  // namespace

extern "C" {
diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernelSize,
                                      diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceilMode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    DIOPI_CHECK(inputTensor.dim() == 3 || inputTensor.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    std::vector<DiopiTensor*> pTensors{&inputTensor};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor inputTensorTmp = *pTensors[0];
    DiopiTensor outTensorTmp = outTensor;
    DIOPI_CALL(dataTypeCast(ctx, outTensorTmp, inputTensorTmp.dtype()));

    std::vector<int> inputDim = getDim(inputTensorTmp);
    std::vector<int> outDim = getDim(outTensorTmp);
    CnnlTensorDesc inputDesc;
    CnnlTensorDesc outDesc;
    inputDesc.set(inputTensorTmp, CNNL_LAYOUT_NCHW, inputDim);
    outDesc.set(outTensorTmp, CNNL_LAYOUT_NCHW, outDim);

    const int64_t kernelH = kernelSize.data[0];
    const int64_t kernelW = kernelSize.len == 1 ? kernelH : kernelSize.data[1];
    int64_t strideH = 0;
    int64_t strideW = 0;
    if (stride.len == 0) {
        strideH = kernelH;
        strideW = kernelW;
    } else {
        strideH = stride.data[0];
        strideW = stride.len == 1 ? strideH : stride.data[1];
    }
    const int64_t padH = padding.data[0];
    const int64_t padW = padding.len == 1 ? padH : padding.data[1];
    const int64_t dilation0 = dilation.data[0];
    const int64_t dilation1 = dilation.len == 1 ? dilation0 : dilation.data[1];

    // calculate padding coefficients
    auto pl = 0, pr = 0, pu = 0, pd = 0;
    pu = pd = padH;
    pl = pr = padW;
    if (ceilMode) {
        // diff = (out - 1) * stride + kernel_size - input
        int diffHeight = (outDim[2] - 1) * strideH + kernelH - inputDim[2];
        int diffWidth = (outDim[3] - 1) * strideW + kernelW - inputDim[3];
        // If ceil_mode is set to true, the pad needs to be filled up.
        // If the offset pad is redundant, it will be removed.
        pd = diffHeight > padH ? diffHeight - padH : 0;
        pr = diffWidth > padW ? diffWidth - padW : 0;
    }

    CnnlResourceGuard<cnnlPoolingDescriptor_t, cnnlCreatePoolingDescriptor, cnnlDestroyPoolingDescriptor> cnnlPoolDesc;
    cnnlPoolingDescriptor_t poolDesc = cnnlPoolDesc.get();
    DIOPI_CALLCNNL(cnnlSetPooling2dDescriptor_v2(
        poolDesc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, kernelH, kernelW, pu, pd, pl, pr, strideH, strideW, dilation0, dilation1, ceilMode));

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetPoolingWorkspaceSize(handle, CNNL_POOLING_MAX, outTensor.shape()[3], inputTensor.shape()[2], &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    const void* alpha = nullptr;
    const void* beta = nullptr;
    DIOPI_CALLCNNL(cnnlPoolingForward(
        handle, poolDesc, alpha, inputDesc.get(), inputTensorTmp.data(), beta, outDesc.get(), outTensorTmp.data(), workspace, workspaceSize));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));

    return diopiSuccess;
}

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                                 diopiSize_t kernelSize, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceilMode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);
    DiopiTensor indicesTensor(indices);

    DIOPI_CHECK(inputTensor.dim() == 3 || inputTensor.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    std::vector<DiopiTensor*> pTensors{&inputTensor};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor inputTensorTmp = *pTensors[0];
    DiopiTensor outTensorTmp = outTensor;
    DIOPI_CALL(dataTypeCast(ctx, outTensorTmp, inputTensorTmp.dtype()));

    DiopiTensor indicesTensorTmp = indicesTensor;
    if (indicesTensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, indicesTensorTmp, diopi_dtype_int32));
    }
    if (inputTensorTmp.dtype() == diopi_dtype_float16) {
        DIOPI_CALL(dataTypeCast(ctx, indicesTensorTmp, diopi_dtype_int16));
    } else if (inputTensorTmp.dtype() == diopi_dtype_float32) {
        DIOPI_CALL(dataTypeCast(ctx, indicesTensorTmp, diopi_dtype_int32));
    } else {
        DIOPI_CHECK(false, "non-empty 3D or 4D (batch mode) tensor expected for input");
    }

    std::vector<int> inputDim = getDim(inputTensorTmp);
    std::vector<int> indicesDim = getDim(indicesTensorTmp);
    std::vector<int> outDim = getDim(outTensorTmp);
    CnnlTensorDesc inputDesc;
    CnnlTensorDesc indicesDesc;
    CnnlTensorDesc outDesc;
    inputDesc.set(inputTensorTmp, CNNL_LAYOUT_NCHW, inputDim);
    indicesDesc.set(indicesTensorTmp, CNNL_LAYOUT_NCHW, indicesDim);
    outDesc.set(outTensorTmp, CNNL_LAYOUT_NCHW, outDim);

    const int64_t kernelH = kernelSize.data[0];
    const int64_t kernelW = kernelSize.len == 1 ? kernelH : kernelSize.data[1];
    int64_t strideH = 0;
    int64_t strideW = 0;
    if (stride.len == 0) {
        strideH = kernelH;
        strideW = kernelW;
    } else {
        strideH = stride.data[0];
        strideW = stride.len == 1 ? strideH : stride.data[1];
    }
    const int64_t padH = padding.data[0];
    const int64_t padW = padding.len == 1 ? padH : padding.data[1];
    const int64_t dilation0 = dilation.data[0];
    const int64_t dilation1 = dilation.len == 1 ? dilation0 : dilation.data[1];

    // calculate padding coefficients
    auto pl = 0, pr = 0, pu = 0, pd = 0;
    pu = pd = padH;
    pl = pr = padW;
    if (ceilMode) {
        // diff = (out - 1) * stride + kernel_size - input
        int diffHeight = (outDim[2] - 1) * strideH + kernelH - inputDim[2];
        int diffWidth = (outDim[3] - 1) * strideW + kernelW - inputDim[3];
        // If ceil_mode is set to true, the pad needs to be filled up.
        // If the offset pad is redundant, it will be removed.
        pd = diffHeight > padH ? diffHeight - padH : 0;
        pr = diffWidth > padW ? diffWidth - padW : 0;
    }

    CnnlResourceGuard<cnnlPoolingDescriptor_t, cnnlCreatePoolingDescriptor, cnnlDestroyPoolingDescriptor> cnnlPoolDesc;
    cnnlPoolingDescriptor_t poolDesc = cnnlPoolDesc.get();
    int poolRank = kernelSize.len;
    if (poolRank == 3) {
        std::vector<int> window{kernelSize.data, kernelSize.data + kernelSize.len};
        std::vector<int> paddingTmp{padding.data, padding.data + padding.len};
        std::vector<int> strideTmp{stride.data, stride.data + stride.len};
        std::vector<int> dilationTmp{dilation.data, dilation.data + dilation.len};
        DIOPI_CALLCNNL(cnnlSetPoolingNdDescriptor_v2(
            poolDesc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, poolRank + 2, window.data(), paddingTmp.data(), strideTmp.data(), dilationTmp.data(), ceilMode));
    } else {
        DIOPI_CALLCNNL(cnnlSetPooling2dDescriptor_v2(
            poolDesc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, kernelH, kernelW, pu, pd, pl, pr, strideH, strideW, dilation0, dilation1, ceilMode));
    }

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetPoolingWithIndexWorkspaceSize(handle, inputDesc.get(), outDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlPoolingForwardWithIndex(handle,
                                               poolDesc,
                                               nullptr,
                                               inputDesc.get(),
                                               inputTensorTmp.data(),
                                               nullptr,
                                               outDesc.get(),
                                               outTensorTmp.data(),
                                               indicesDesc.get(),
                                               indicesTensorTmp.data(),
                                               workspace,
                                               workspaceSize));

    if (indicesTensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, indicesTensorTmp, diopi_dtype_int32));
    }
    DIOPI_CALL(dataTypeCast(ctx, indicesTensor, indicesTensorTmp));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));

    return diopiSuccess;
}

diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                              diopiConstTensorHandle_t input, diopiSize_t kernelSize, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceilMode, diopiConstTensorHandle_t indices) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor indicesTensor(indices);

    DIOPI_CHECK(inputTensor.dim() == 3 || inputTensor.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    std::vector<DiopiTensor*> pTensors{&inputTensor, &gradOutputTensor};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor inputTensorTmp = *pTensors[0];
    DiopiTensor gradOutputTensorTmp = *pTensors[1];
    DiopiTensor gradInputTensorTmp = gradInputTensor;
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensorTmp, inputTensorTmp.dtype()));

    DiopiTensor indicesTensorTmp = indicesTensor;
    if (indicesTensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, indicesTensorTmp, diopi_dtype_int32));
    }
    if (inputTensorTmp.dtype() == diopi_dtype_float16) {
        DIOPI_CALL(dataTypeCast(ctx, indicesTensorTmp, diopi_dtype_int16));
    } else if (inputTensorTmp.dtype() == diopi_dtype_float32) {
        DIOPI_CALL(dataTypeCast(ctx, indicesTensorTmp, diopi_dtype_int32));
    } else {
        DIOPI_CHECK(false, "non-empty 3D or 4D (batch mode) tensor expected for input");
    }

    diopiTensorHandle_t inputT = nullptr;
    diopiTensorHandle_t gradInputT = nullptr;
    diopiTensorHandle_t gradOutputT = nullptr;
    diopiTensorHandle_t indicesT = nullptr;

    auto permuteToNhwc = [&](auto src, auto& dst) {
        DiopiTensor srcTensor(src);
        std::vector<int64_t> srcShapeT64(srcTensor.shape().size());
        std::vector<int64_t> axis{0, 2, 3, 1};
        if (srcTensor.shape().size() == 3) {
            axis.clear();
            axis.push_back(1);
            axis.push_back(2);
            axis.push_back(0);
        }
        for (int i = 0; i < srcTensor.shape().size(); ++i) {
            srcShapeT64[i] = srcTensor.shape()[axis[i]];
        }

        diopiSize_t srcTShape(srcShapeT64.data(), srcShapeT64.size());
        DIOPI_CALL(diopiRequireTensor(ctx, &dst, &srcTShape, nullptr, srcTensor.dtype(), diopi_device));
        if (srcTensor.shape().size() == 4) {
            diopiSize_t nchw2nhwc(axis.data(), 4);
            DIOPI_CALL(diopiPermute(ctx, dst, src, nchw2nhwc));
        } else if (srcTensor.shape().size() == 3) {
            diopiSize_t chw2hwc(axis.data(), 3);
            DIOPI_CALL(diopiPermute(ctx, dst, src, chw2hwc));
        } else {
            DIOPI_CHECK(false, "non-empty 3D or 4D (batch mode) tensor expected for input");
        }
        return diopiSuccess;
    };

    DIOPI_CALL(permuteToNhwc(static_cast<diopiTensorHandle_t>(inputTensorTmp), inputT));
    DIOPI_CALL(permuteToNhwc(static_cast<diopiTensorHandle_t>(gradInputTensorTmp), gradInputT));
    DIOPI_CALL(permuteToNhwc(static_cast<diopiTensorHandle_t>(gradOutputTensorTmp), gradOutputT));
    DIOPI_CALL(permuteToNhwc(static_cast<diopiTensorHandle_t>(indicesTensorTmp), indicesT));

    DiopiTensor inputTensorT(inputT);
    DiopiTensor gradInputTensorT(gradInputT);
    DiopiTensor gradOutputTensorT(gradOutputT);
    DiopiTensor indicesTensorT(indicesT);

    std::vector<int> inputDim = getDim(inputTensorT);
    std::vector<int> gradInputDim = getDim(gradInputTensorT);
    std::vector<int> gradOutputDim = getDim(gradOutputTensorT);
    std::vector<int> indicesDim = getDim(indicesTensorT);
    CnnlTensorDesc inputDesc;
    CnnlTensorDesc gradInputDesc;
    CnnlTensorDesc gradOutputDesc;
    CnnlTensorDesc indicesDesc;
    inputDesc.set(inputTensorT, CNNL_LAYOUT_NHWC, inputDim);
    gradInputDesc.set(gradInputTensorT, CNNL_LAYOUT_NHWC, gradInputDim);
    gradOutputDesc.set(gradOutputTensorT, CNNL_LAYOUT_NHWC, gradOutputDim);
    indicesDesc.set(indicesTensorT, CNNL_LAYOUT_NHWC, indicesDim);

    const int64_t kernelH = kernelSize.data[0];
    const int64_t kernelW = kernelSize.len == 1 ? kernelH : kernelSize.data[1];
    int64_t strideH = 0;
    int64_t strideW = 0;
    if (stride.len == 0) {
        strideH = kernelH;
        strideW = kernelW;
    } else {
        strideH = stride.data[0];
        strideW = stride.len == 1 ? strideH : stride.data[1];
    }
    const int64_t padH = padding.data[0];
    const int64_t padW = padding.len == 1 ? padH : padding.data[1];
    const int64_t dilation0 = dilation.data[0];
    const int64_t dilation1 = dilation.len == 1 ? dilation0 : dilation.data[1];

    // calculate padding coefficients
    auto pl = 0, pr = 0, pu = 0, pd = 0;
    pu = pd = padH;
    pl = pr = padW;
    int height = (gradOutputDim[1] - 1) * strideH + kernelH;
    int width = (gradOutputDim[2] - 1) * strideW + kernelW;
    if (padH + inputDim[1] >= height) {
        pd = 0;
    }
    if (padW + inputDim[2] >= width) {
        pr = 0;
    }
    // if ceil_mode is set to true, the pad needs to be filled up.
    if (ceilMode) {
        pd = height - inputDim[1] - padH;
        pr = width - inputDim[2] - padW;
    }

    CnnlResourceGuard<cnnlPoolingDescriptor_t, cnnlCreatePoolingDescriptor, cnnlDestroyPoolingDescriptor> cnnlPoolDesc;
    cnnlPoolingDescriptor_t poolDesc = cnnlPoolDesc.get();
    DIOPI_CALLCNNL(cnnlSetPooling2dDescriptor_v2(
        poolDesc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, kernelH, kernelW, pu, pd, pl, pr, strideH, strideW, dilation0, dilation1, ceilMode));

    DIOPI_CALLCNNL(cnnlPoolingBackward(handle,
                                       poolDesc,
                                       nullptr,
                                       indicesDesc.get(),
                                       indicesTensorT.data(),
                                       gradOutputDesc.get(),
                                       gradOutputTensorT.data(),
                                       inputDesc.get(),
                                       inputTensorT.data(),
                                       nullptr,
                                       gradInputDesc.get(),
                                       gradInputTensorT.data()));

    if (gradInputTensorT.shape().size() == 4) {
        std::vector<int64_t> permNhwc2nchw{0, 3, 1, 2};
        diopiSize_t nhwc2nchw(permNhwc2nchw.data(), 4);
        DIOPI_CALL(diopiPermute(ctx, static_cast<diopiTensorHandle_t>(gradInputTensorTmp), gradInputT, nhwc2nchw));
    } else if (gradInputTensorT.shape().size() == 3) {
        std::vector<int64_t> permHwc2chw{2, 0, 1};
        diopiSize_t hwc2chw(permHwc2chw.data(), 3);
        DIOPI_CALL(diopiPermute(ctx, static_cast<diopiTensorHandle_t>(gradInputTensorTmp), gradInputT, hwc2chw));
    } else {
        DIOPI_CHECK(false, "non-empty 3D or 4D (batch mode) tensor expected for input");
    }
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTensorTmp));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
