#pragma once

#include <cudf/rolling.hpp>
#include <cudf/types.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

/// Fixed-size rolling window aggregation.
std::unique_ptr<OwnedColumn> rolling_window(
    const OwnedColumn& col,
    int32_t preceding,
    int32_t following,
    int32_t min_periods,
    int32_t agg_kind);

/// Grouped rolling window aggregation.
/// The input must be pre-sorted by group_keys.
std::unique_ptr<OwnedColumn> grouped_rolling_window(
    const OwnedTable& group_keys,
    const OwnedColumn& input,
    int32_t preceding,
    int32_t following,
    int32_t min_periods,
    int32_t agg_kind);

/// Variable-size rolling window aggregation.
/// preceding_col and following_col specify per-row window sizes.
std::unique_ptr<OwnedColumn> rolling_window_variable(
    const OwnedColumn& col,
    const OwnedColumn& preceding_col,
    const OwnedColumn& following_col,
    int32_t min_periods,
    int32_t agg_kind);

} // namespace cudf_shims
