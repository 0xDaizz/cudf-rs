#pragma once

#include <cudf/quantiles.hpp>
#include <cudf/types.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

/// Compute quantile(s) of a column.
std::unique_ptr<OwnedColumn> quantile(
    const OwnedColumn& col,
    rust::Slice<const double> q,
    int32_t interp);

/// Compute quantiles of a table (row-wise).
std::unique_ptr<OwnedTable> quantiles_table(
    const OwnedTable& table,
    rust::Slice<const double> q,
    int32_t interp,
    bool is_input_sorted,
    rust::Slice<const int32_t> orders,
    rust::Slice<const int32_t> null_orders);

/// Compute percentile approximation using t-digest.
std::unique_ptr<OwnedColumn> percentile_approx(
    const OwnedColumn& tdigest_col,
    rust::Slice<const double> percentiles);

} // namespace cudf_shims
