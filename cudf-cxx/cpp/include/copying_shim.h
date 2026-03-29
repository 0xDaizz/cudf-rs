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
#include "scalar_shim.h"

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

/// Result of splitting a table into multiple parts.
struct SplitResult {
    std::vector<std::unique_ptr<OwnedTable>> parts;
};

/// Split a table at the given indices, returning all parts at once.
std::unique_ptr<SplitResult> split_table_all(
    const OwnedTable& table, rust::Slice<const int32_t> splits);

/// Return the number of parts in a split result.
int32_t split_result_count(const SplitResult& result);

/// Move one part out of a split result by index.
std::unique_ptr<OwnedTable> split_result_get(SplitResult& result, int32_t index);

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

// ── Reverse ───────────────────────────────────────────────────

/// Reverse the rows of a table.
std::unique_ptr<OwnedTable> reverse_table(const OwnedTable& table);

/// Reverse the elements of a column.
std::unique_ptr<OwnedColumn> reverse_column(const OwnedColumn& col);

// ── Shift ─────────────────────────────────────────────────────

/// Shift column elements by offset, filling with fill_value.
std::unique_ptr<OwnedColumn> shift_column(
    const OwnedColumn& col,
    int32_t offset,
    const OwnedScalar& fill_value);

// ── Get Element ───────────────────────────────────────────────

/// Get a single element from a column as a scalar.
std::unique_ptr<OwnedScalar> get_element(
    const OwnedColumn& col,
    int32_t index);

// ── Sample ────────────────────────────────────────────────────

/// Randomly sample n rows from a table.
std::unique_ptr<OwnedTable> sample(
    const OwnedTable& table,
    int32_t n,
    bool with_replacement,
    int64_t seed);

// ── Boolean Mask Scatter ──────────────────────────────────────

/// Scatter input rows into target at positions where boolean_mask is true.
std::unique_ptr<OwnedTable> boolean_mask_scatter(
    const OwnedTable& input,
    const OwnedColumn& boolean_mask,
    const OwnedTable& target);

// ── Has Nonempty Nulls ────────────────────────────────────────

/// Check if a column has non-empty null elements.
bool has_nonempty_nulls(const OwnedColumn& col);

} // namespace cudf_shims
