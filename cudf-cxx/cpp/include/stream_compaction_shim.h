#pragma once

#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

// ── Stream Compaction ─────────────────────────────────────────

/// Drop rows from a table where any of the specified key columns contain nulls.
/// `keys` specifies which column indices to check for nulls.
/// `threshold` is the minimum number of non-null key values required to keep a row.
std::unique_ptr<OwnedTable> drop_nulls_table(
    const OwnedTable& table,
    rust::Slice<const int32_t> keys,
    int32_t threshold);

/// Drop null values from a single column.
std::unique_ptr<OwnedColumn> drop_nulls_column(
    const OwnedColumn& col);

/// Drop rows from a table where any of the specified key columns contain NaN.
std::unique_ptr<OwnedTable> drop_nans(
    const OwnedTable& table,
    rust::Slice<const int32_t> keys);

/// Drop rows from a table where key columns contain NaN, with a threshold.
std::unique_ptr<OwnedTable> drop_nans_threshold(
    const OwnedTable& table,
    rust::Slice<const int32_t> keys,
    int32_t threshold);

/// Apply a boolean mask to a table, keeping only rows where mask is true.
std::unique_ptr<OwnedTable> apply_boolean_mask(
    const OwnedTable& table,
    const OwnedColumn& boolean_mask);

/// Returns a table with unique rows based on the specified key columns.
/// `keep`: 0=KEEP_FIRST, 1=KEEP_LAST, 2=KEEP_ANY, 3=KEEP_NONE.
/// `null_equality`: 0=EQUAL, 1=UNEQUAL.
std::unique_ptr<OwnedTable> unique(
    const OwnedTable& table,
    rust::Slice<const int32_t> keys,
    int32_t keep,
    int32_t null_equality);

/// Returns a table with distinct rows based on the specified key columns.
std::unique_ptr<OwnedTable> distinct(
    const OwnedTable& table,
    rust::Slice<const int32_t> keys,
    int32_t keep,
    int32_t null_equality);

/// Count the number of distinct elements in a column.
/// `null_handling`: 0=INCLUDE, 1=EXCLUDE.
/// `nan_handling`: 0=NAN_IS_VALID, 1=NAN_IS_NULL.
int32_t distinct_count_column(
    const OwnedColumn& col,
    int32_t null_handling,
    int32_t nan_handling);

/// Return indices of distinct rows in a table (all columns as keys).
/// `keep`: 0=KEEP_FIRST, 1=KEEP_LAST, 2=KEEP_ANY, 3=KEEP_NONE.
std::unique_ptr<OwnedColumn> distinct_indices(
    const OwnedTable& table,
    int32_t keep,
    int32_t null_equality);

/// Return distinct rows preserving input order.
std::unique_ptr<OwnedTable> stable_distinct(
    const OwnedTable& table,
    rust::Slice<const int32_t> keys,
    int32_t keep,
    int32_t null_equality);

/// Count consecutive groups of equivalent rows in a column.
int32_t unique_count_column(
    const OwnedColumn& col,
    int32_t null_handling,
    int32_t nan_handling);

/// Count consecutive groups of equivalent rows in a table.
int32_t unique_count_table(
    const OwnedTable& table,
    int32_t null_equality);

/// Count distinct rows in a table.
int32_t distinct_count_table(
    const OwnedTable& table,
    int32_t null_equality);

} // namespace cudf_shims
