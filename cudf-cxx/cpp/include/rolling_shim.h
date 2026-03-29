#pragma once

#include <cudf/rolling.hpp>
#include <cudf/types.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

/// Fixed-size rolling window aggregation.
std::unique_ptr<OwnedColumn> rolling_window(
    const OwnedColumn& col,
    int32_t preceding,
    int32_t following,
    int32_t min_periods,
    int32_t agg_kind);

} // namespace cudf_shims
