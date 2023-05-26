/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t num_outs, diopiConstTensorHandle_t input,
                                            const diopiSize_t splitSizes, int64_t dim) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto inputTensor = DiopiTensor(input);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);

    std::vector<CnnlTensorDesc> outputDescs(num_outs);
    cnnlTensorDescriptor_t* descPtrs = new cnnlTensorDescriptor_t[num_outs];
    void** dataPtrs = new void*[num_outs];
    for (int i = 0; i < num_outs; ++i) {
        auto tensor = DiopiTensor(outs[i]);
        DIOPI_CALL(outputDescs[i].set(tensor, CNNL_LAYOUT_ARRAY));
        descPtrs[i] = outputDescs[i].get();
        dataPtrs[i] = tensor.data();
    }

    size_t worksapceSize;
    DIOPI_CALLCNNL(cnnlGetSplitWorkspaceSize(handle, num_outs, &worksapceSize));

    void* worksapce = nullptr;
    if (worksapceSize != 0) {
        worksapce = requiresBuffer(ctx, worksapceSize).data();
    }

    DIOPI_CALLCNNL(cnnlSplit(handle, num_outs, dim, inputDesc.get(), inputTensor.data(), worksapce, worksapceSize, descPtrs, dataPtrs));

    delete[] descPtrs;
    delete[] dataPtrs;
    return diopiSuccess;
}
}  // namespace camb
}  // namespace impl
