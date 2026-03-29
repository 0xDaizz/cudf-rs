#pragma once

#include <cudf/copying.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

// ── Gather / Scatter ───────────────────────────────────────────

/// Gather rows from table by index column.
std::unique_ptr<OwnedTable> gather(
    const OwnedTable& table,
    const OwnedColumn& gather_map,
    int32_t bounds_policy);

/// Scatter source rows into target at scatter_map positions.
std::unique_ptr<OwnedTable> scatter(
    const OwnedTable& source,
    const OwnedColumn& scatter_map,
    const OwnedTable& target);

// ── Conditional Copy ───────────────────────────────────────────

/// Elementwise: select lhs where mask true, rhs where false.
std::unique_ptr<OwnedColumn> copy_if_else(
    const OwnedColumn& lhs,
    const OwnedColumn& rhs,
    const OwnedColumn& boolean_mask);

// ── Slice / Split ──────────────────────────────────────────────

/// Slice [begin, end) as an owned deep copy.
std::unique_ptr<OwnedTable> slice_table(
    const OwnedTable& table,
    int32_t begin,
    int32_t end);

/// Return the number of parts from splitting at the given indices.
int32_t split_table_count(rust::Slice<const int32_t> splits);

/// Get one part from a split operation (deep copy). Index 0..split_table_count.
std::unique_ptr<OwnedTable> split_table_part(
    const OwnedTable& table,
    rust::Slice<const int32_t> splits,
    int32_t part_index);

// ── Empty / Allocate ───────────────────────────────────────────

/// Create an empty column (all null) matching col's type and size.
std::unique_ptr<OwnedColumn> empty_like(const OwnedColumn& col);

/// Allocate a column matching col's type and size with given mask policy.
std::unique_ptr<OwnedColumn> allocate_like(
    const OwnedColumn& col,
    int32_t mask_policy);

// ── In-place Copy ──────────────────────────────────────────────

/// Copy a range from source into target column.
void copy_range(
    const OwnedColumn& source,
    OwnedColumn& target,
    int32_t source_begin,
    int32_t source_end,
    int32_t target_begin);

} // namespace cudf_shims
