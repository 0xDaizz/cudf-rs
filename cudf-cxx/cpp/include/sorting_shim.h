#pragma once

#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

// ── Sorting ────────────────────────────────────────────────────

/// Returns a column of row indices that would sort the table.
std::unique_ptr<OwnedColumn> sorted_order(
    const OwnedTable& table,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order);

/// Sort a table by its columns.
std::unique_ptr<OwnedTable> sort(
    const OwnedTable& table,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order);

/// Sort `values` table by the rows of `keys` table.
std::unique_ptr<OwnedTable> sort_by_key(
    const OwnedTable& values,
    const OwnedTable& keys,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order);

/// Stable sort `values` table by the rows of `keys` table.
std::unique_ptr<OwnedTable> stable_sort_by_key(
    const OwnedTable& values,
    const OwnedTable& keys,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order);

/// Compute rank of each element in a column.
std::unique_ptr<OwnedColumn> rank(
    const OwnedColumn& col,
    int32_t method,
    int32_t column_order,
    int32_t null_order,
    int32_t null_handling,
    bool percentage);

/// Check whether a table is sorted.
bool is_sorted(
    const OwnedTable& table,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order);

} // namespace cudf_shims
