#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                             diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t outputPadding, int64_t groups,
                                             diopiSize_t dilation) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor = DiopiTensor(input);
    DiopiTensor weightTensor = DiopiTensor(weight);
    DiopiTensor outputTensor = DiopiTensor(out);

    DiopiTensor inputCasted = inputTensor;
    DiopiTensor weightCasted = weightTensor;
    DiopiTensor outputCasted = outputTensor;

    std::vector<DiopiTensor *> tensors{&inputCasted, &weightCasted, &outputCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
    CnnlTensorDesc inputDesc(inputCasted, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc weightDesc(weightCasted, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc outputDesc(outputCasted, CNNL_LAYOUT_NCHW);

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> convDesc;

    std::vector<int> strideVec{stride.data, stride.data + stride.len};
    std::vector<int> paddingVec{padding.data, padding.data + padding.len};
    std::vector<int> dilationVec{dilation.data, dilation.data + dilation.len};

    int convPadding[4] = {paddingVec[0], paddingVec[0], paddingVec[1], paddingVec[1]};
    int convStride[2] = {strideVec[0], strideVec[1]};
    int convDilation[2] = {dilationVec[0], dilationVec[1]};

    cnnlDataType_t computeType;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&computeType, inputCasted.dtype()));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(convDesc.get(), 4, convPadding, convStride, convDilation, groups, computeType));

    size_t workspaceSizeInput;
    DIOPI_CALLCNNL(cnnlGetConvolutionBackwardDataWorkspaceSize(
        handle, weightDesc.get(), inputDesc.get(), convDesc.get(), outputDesc.get(), CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT, &workspaceSizeInput));
    void *workspaceInput;
    if (workspaceSizeInput != 0) {
        workspaceInput = requiresBuffer(ctx, workspaceSizeInput).data();
    }
    float alpha = 1.0;
    float beta = 1.0;
    DIOPI_CALLCNNL(cnnlConvolutionBackwardData(handle,
                                               &alpha,
                                               weightDesc.get(),
                                               weightCasted.data(),
                                               inputDesc.get(),
                                               inputCasted.data(),
                                               convDesc.get(),
                                               CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT,
                                               workspaceInput,
                                               workspaceSizeInput,
                                               &beta,
                                               outputDesc.get(),
                                               outputCasted.data()));

    DiopiTensor biasTensor = DiopiTensor(bias);
    if (biasTensor.defined()) {
        DiopiTensor biasCasted = biasTensor;

        std::vector<DiopiTensor *> tensors{&biasCasted};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
        CnnlTensorDesc biasDesc(biasCasted, CNNL_LAYOUT_ARRAY);

        DiopiTensor outputTr;
        std::vector<int64_t> axis{0, 2, 3, 1};
        std::vector<int64_t> TranspoedShape(outputCasted.shape().size());
        for (int i = 0; i < outputCasted.shape().size(); ++i) {
            TranspoedShape[i] = outputCasted.shape()[axis[i]];
        }
        diopiSize_t outputTrShape(TranspoedShape.data(), TranspoedShape.size());
        auto outputTrHandle = outputTr.tensorHandle();
        DIOPI_CALL(diopiRequireTensor(ctx, &outputTrHandle, &outputTrShape, nullptr, outputCasted.dtype(), diopi_device));
        diopiSize_t nchw2nhwc(axis.data(), 4);
        DIOPI_CALL(diopiPermute(ctx, outputTrHandle, outputCasted.tensorHandle(), nchw2nhwc));
        outputTr = DiopiTensor(outputTrHandle);
        CnnlTensorDesc outputTrDesc(outputTr, CNNL_LAYOUT_ARRAY);

        size_t workspaceSizeBias;
        DIOPI_CALLCNNL(cnnlGetBiasAddWorkspaceSize(handle, biasDesc.get(), outputTrDesc.get(), &workspaceSizeBias));

        void *workspaceBias = nullptr;
        if (workspaceSizeBias != 0) {
            workspaceBias = requiresBuffer(ctx, workspaceSizeBias).data();
        }

        DIOPI_CALLCNNL(
            cnnlBiasAdd(handle, &alpha, biasDesc.get(), biasCasted.data(), workspaceBias, workspaceSizeBias, &beta, outputTrDesc.get(), outputTr.data()));

        std::vector<int64_t> permNhwc2nchw{0, 3, 1, 2};
        diopiSize_t nhwc2nchw(permNhwc2nchw.data(), 4);
        DIOPI_CALL(diopiPermute(ctx, outputCasted.tensorHandle(), outputTr.tensorHandle(), nhwc2nchw));
    }
    DIOPI_CALL(dataTypeCast(ctx, outputTensor, outputCasted));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
