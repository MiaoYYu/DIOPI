#ifndef DIOPI_IMPL_CAMB_COMMON_CNNL_SCALAR_HPP_
#define DIOPI_IMPL_CAMB_COMMON_CNNL_SCALAR_HPP_

#include <set>

#include "common.hpp"

namespace impl {
namespace camb {

class CnnlScalar {
public:
    CnnlScalar() { mluPtr_ = nullptr; }
    template <typename T1, typename T2>
    explicit CnnlScalar(T2 data) {
        DIOPI_CHECK_ABORT(set<T1>(data) == diopiSuccess, "%s", "failed to set data");
    }
    template <typename T1, typename T2>
    diopiError_t set(T2 data) {
        T1 value = T1(data);
        if (mluPtr_ == nullptr) {
            DIOPI_CHECK(cnrtMalloc(&mluPtr_, sizeof(T1)) == cnrtSuccess, "%s", "failed to malloc memory.");
        } else {
            DIOPI_CHECK(cnrtFree(mluPtr_) == cnrtSuccess, "%s", "failed to malloc memory.");
            DIOPI_CHECK(cnrtMalloc(&mluPtr_, sizeof(T1)) == cnrtSuccess, "%s", "failed to malloc memory.");
        }
        DIOPI_CHECK(cnrtMemcpy(mluPtr_, &value, sizeof(T1), cnrtMemcpyHostToDev) == cnrtSuccess, "%s", "failed to malloc memory.");
        return diopiSuccess;
    }

    CnnlScalar(const CnnlScalar& other) = delete;
    CnnlScalar(CnnlScalar&& other) = delete;
    CnnlScalar& operator=(const CnnlScalar& other) = delete;
    ~CnnlScalar() {
        if (mluPtr_ != nullptr) {
            cnrtFree(mluPtr_);
        }
    }
    const void* data() const { return mluPtr_; }

private:
    void* mluPtr_;
};

}  // namespace camb
}  // namespace impl

#endif  // DIOPI_IMPL_CAMB_COMMON_CNNL_SCALAR_HPP_
