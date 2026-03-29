#pragma once

#include <cudf/lists/explode.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/extract.hpp>
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

} // namespace cudf_shims
