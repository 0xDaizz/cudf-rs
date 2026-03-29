#pragma once

#include <cudf/labeling/label_bins.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

/// Label elements based on membership in the specified bins.
/// left_inclusive/right_inclusive: true = YES, false = NO.
std::unique_ptr<OwnedColumn> label_bins(
    const OwnedColumn& input,
    const OwnedColumn& left_edges,
    bool left_inclusive,
    const OwnedColumn& right_edges,
    bool right_inclusive);

} // namespace cudf_shims
