#pragma once

#include <cudf/search.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

/// Find the lower bound indices for each row in `values` within a sorted `table`.
std::unique_ptr<OwnedColumn> lower_bound(
    const OwnedTable& table,
    const OwnedTable& values,
    rust::Slice<const int32_t> orders,
    rust::Slice<const int32_t> null_orders);

/// Find the upper bound indices for each row in `values` within a sorted `table`.
std::unique_ptr<OwnedColumn> upper_bound(
    const OwnedTable& table,
    const OwnedTable& values,
    rust::Slice<const int32_t> orders,
    rust::Slice<const int32_t> null_orders);

/// For each element in `needles`, check if it exists in `haystack`.
std::unique_ptr<OwnedColumn> contains_column(
    const OwnedColumn& haystack,
    const OwnedColumn& needles);

} // namespace cudf_shims
