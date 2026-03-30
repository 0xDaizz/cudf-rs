#pragma once
#include <cudf/types.hpp>
#include "rust/cxx.h"
#include <vector>

namespace cudf_shims {

/// Convert a flat i32 slice to a vector of cudf::order.
inline std::vector<cudf::order> to_column_order(rust::Slice<const int32_t> order) {
    std::vector<cudf::order> result;
    result.reserve(order.size());
    for (auto v : order) {
        result.push_back(v == 0 ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    }
    return result;
}

/// Convert a flat i32 slice to a vector of cudf::null_order.
inline std::vector<cudf::null_order> to_null_order(rust::Slice<const int32_t> order) {
    std::vector<cudf::null_order> result;
    result.reserve(order.size());
    for (auto v : order) {
        result.push_back(v == 0 ? cudf::null_order::AFTER : cudf::null_order::BEFORE);
    }
    return result;
}

} // namespace cudf_shims
