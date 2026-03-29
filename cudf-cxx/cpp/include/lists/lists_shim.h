#pragma once

#include <cudf/lists/explode.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/lists/gather.hpp>
#include <cudf/lists/set_operations.hpp>
#include <cudf/lists/reverse.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"
#include "scalar_shim.h"

namespace cudf_shims {

// ── Explode ────────────────────────────────────────────────────

/// Explode a list column, expanding each list element into its own row.
std::unique_ptr<OwnedTable> lists_explode(
    const OwnedTable& table,
    int32_t explode_col_idx);

/// Explode a list column, retaining null entries and empty lists.
std::unique_ptr<OwnedTable> lists_explode_outer(
    const OwnedTable& table,
    int32_t explode_col_idx);

/// Explode with position column.
std::unique_ptr<OwnedTable> lists_explode_position(
    const OwnedTable& table,
    int32_t explode_col_idx);

/// Explode outer with position column.
std::unique_ptr<OwnedTable> lists_explode_outer_position(
    const OwnedTable& table,
    int32_t explode_col_idx);

// ── Sorting ───────────────────────────────────────────────────

/// Sort elements within each list row.
std::unique_ptr<OwnedColumn> lists_sort(
    const OwnedColumn& col,
    bool ascending,
    int32_t null_order);

// ── Contains ──────────────────────────────────────────────────

/// Check whether each list row contains the given scalar value.
std::unique_ptr<OwnedColumn> lists_contains(
    const OwnedColumn& col,
    const OwnedScalar& search_key);

/// Check whether each list row contains any null elements.
std::unique_ptr<OwnedColumn> lists_contains_nulls(
    const OwnedColumn& col);

// ── Extract ───────────────────────────────────────────────────

/// Extract the element at index from each list row.
std::unique_ptr<OwnedColumn> lists_extract(
    const OwnedColumn& col,
    int32_t index);

// ── Count Elements ────────────────────────────────────────────

/// Count elements in each list row.
std::unique_ptr<OwnedColumn> lists_count_elements(
    const OwnedColumn& col);

// ── Index Of (scalar) ─────────────────────────────────────────

/// Find position of scalar in each list row. Returns -1 if not found.
std::unique_ptr<OwnedColumn> lists_index_of_scalar(
    const OwnedColumn& col,
    const OwnedScalar& key);

// ── Combine ───────────────────────────────────────────────────

/// Concatenate lists across columns (row-wise).
std::unique_ptr<OwnedColumn> lists_concatenate_rows(
    const OwnedTable& table);

/// Concatenate nested list elements within each row.
std::unique_ptr<OwnedColumn> lists_concatenate_list_elements(
    const OwnedColumn& col);

// ── Filling (sequences) ──────────────────────────────────────

/// Generate list column of arithmetic sequences.
std::unique_ptr<OwnedColumn> lists_sequences(
    const OwnedColumn& starts,
    const OwnedColumn& sizes);

// ── Gather ────────────────────────────────────────────────────

/// Gather elements from lists based on per-row gather maps.
std::unique_ptr<OwnedColumn> lists_segmented_gather(
    const OwnedColumn& col,
    const OwnedColumn& gather_map);

// ── Set Operations ────────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_have_overlap(
    const OwnedColumn& lhs,
    const OwnedColumn& rhs);

std::unique_ptr<OwnedColumn> lists_intersect_distinct(
    const OwnedColumn& lhs,
    const OwnedColumn& rhs);

std::unique_ptr<OwnedColumn> lists_union_distinct(
    const OwnedColumn& lhs,
    const OwnedColumn& rhs);

std::unique_ptr<OwnedColumn> lists_difference_distinct(
    const OwnedColumn& lhs,
    const OwnedColumn& rhs);

// ── Reverse ───────────────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_reverse(
    const OwnedColumn& col);

// ── Stream Compaction ─────────────────────────────────────────

std::unique_ptr<OwnedColumn> lists_apply_boolean_mask(
    const OwnedColumn& col,
    const OwnedColumn& mask);

std::unique_ptr<OwnedColumn> lists_distinct(
    const OwnedColumn& col);

/// Stable sort elements within each list row.
std::unique_ptr<OwnedColumn> lists_stable_sort(
    const OwnedColumn& col,
    bool ascending,
    int32_t null_order);

/// Extract elements using per-row indices from a column.
std::unique_ptr<OwnedColumn> lists_extract_column_index(
    const OwnedColumn& col,
    const OwnedColumn& indices);

/// Check whether each list row contains the corresponding value from search_keys column.
std::unique_ptr<OwnedColumn> lists_contains_column(
    const OwnedColumn& col,
    const OwnedColumn& search_keys);

} // namespace cudf_shims
