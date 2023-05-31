#include "../common/cnnl_scalar.hpp"
#include "../common/common.hpp"
#include "../common/float16.hpp"
#include "../diopi_helper.hpp"

namespace impl {
namespace camb {

using half = half_float::half;

extern "C" diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t squareAvg,
                                      diopiTensorHandle_t accDelta, float lr, float rho, float eps, float weightDecay) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor = DiopiTensor(input);
    DiopiTensor gradTensor = DiopiTensor(grad);
    DiopiTensor squareAvgTensor = DiopiTensor(squareAvg);
    DiopiTensor accDeltaTensor = DiopiTensor(accDelta);

    DiopiTensor inputCasted = inputTensor;
    DiopiTensor squareAvgCasted = squareAvgTensor;
    DiopiTensor accDeltaCasted = accDeltaTensor;

    DiopiTensor gradCasted;
    if (weightDecay != 0) {
        DIOPI_CALL(clone(ctx, gradTensor, gradCasted));
    } else {
        gradCasted = gradTensor;
    }

    std::vector<DiopiTensor*> tensors{&inputCasted, &gradCasted, &squareAvgCasted, &accDeltaCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    if (weightDecay != 0) {
        DIOPI_CALL(cnnlOpTensor(ctx, inputCasted, gradCasted, gradCasted, CNNL_OP_TENSOR_ADD, 1.0, static_cast<double>(weightDecay), 0.0));
    }

    CnnlTensorDesc inputDesc(inputCasted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc squareAvgDesc(squareAvgCasted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc accDeltaDesc(accDeltaCasted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradDesc(gradCasted, CNNL_LAYOUT_ARRAY);

    CnnlScalar mluLr, mluRho, mluEps;
    if (inputCasted.dtype() == diopi_dtype_float32) {
        mluLr.set<float>(lr);
        mluRho.set<float>(rho);
        mluEps.set<float>(eps);
    } else {
        mluLr.set<half>(lr);
        mluRho.set<half>(rho);
        mluEps.set<half>(eps);
    }
    DIOPI_CALLCNNL(cnnlApplyAdadelta(handle,
                                     inputDesc.get(),
                                     inputCasted.data(),
                                     squareAvgDesc.get(),
                                     squareAvgCasted.data(),
                                     accDeltaDesc.get(),
                                     accDeltaCasted.data(),
                                     gradDesc.get(),
                                     gradCasted.data(),
                                     mluLr.data(),
                                     mluRho.data(),
                                     mluEps.data()));

    DIOPI_CALL(dataTypeCast(ctx, inputTensor, inputCasted));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
