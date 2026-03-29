#pragma once

#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "table_shim.h"
#include "column_shim.h"
#include "aggregation_shim.h"

namespace cudf_shims {

/// Builder for groupby operations. Holds a reference to key columns and
/// accumulates aggregation requests before execution.
struct GroupByBuilder {
    /// Copy of the keys table view (the OwnedTable must outlive this builder).
    cudf::table_view keys_view;

    /// Accumulated requests: (column_index, aggregation).
    std::vector<std::pair<cudf::size_type, std::unique_ptr<cudf::groupby_aggregation>>> requests;

    explicit GroupByBuilder(cudf::table_view keys)
        : keys_view(keys) {}
};

/// Create a groupby builder with the given key columns.
std::unique_ptr<GroupByBuilder> groupby_new(const OwnedTable& keys);

/// Add an aggregation request to the builder.
void groupby_add_request(
    GroupByBuilder& builder,
    int32_t col_idx,
    std::unique_ptr<OwnedAggregation> agg);

/// Execute the groupby and return a combined table (keys + values).
std::unique_ptr<OwnedTable> groupby_execute(
    GroupByBuilder& builder,
    const OwnedTable& values);

/// Execute and return only the key columns result.
std::unique_ptr<OwnedTable> groupby_execute_keys(
    GroupByBuilder& builder,
    const OwnedTable& values);

/// Execute and return only the aggregation result columns.
std::unique_ptr<OwnedTable> groupby_execute_values(
    GroupByBuilder& builder,
    const OwnedTable& values);

// ── GroupBy Scan ──────────────────────────────────────────────

/// Builder for groupby scan operations.
struct GroupByScanBuilder {
    cudf::table_view keys_view;
    std::vector<std::pair<cudf::size_type, std::unique_ptr<cudf::groupby_scan_aggregation>>> requests;

    explicit GroupByScanBuilder(cudf::table_view keys)
        : keys_view(keys) {}
};

/// Create a groupby scan builder.
std::unique_ptr<GroupByScanBuilder> groupby_scan_new(const OwnedTable& keys);

/// Add a scan aggregation request.
void groupby_scan_add_request(
    GroupByScanBuilder& builder,
    int32_t col_idx,
    int32_t agg_kind);

/// Execute the scan and return keys + scan results.
std::unique_ptr<OwnedTable> groupby_scan_execute(
    GroupByScanBuilder& builder,
    const OwnedTable& values);

// ── GroupBy Get Groups ────────────────────────────────────────

/// Result of get_groups: keys, offsets, and optionally grouped values.
struct GroupByGroupsResult {
    std::unique_ptr<OwnedTable> keys;
    std::unique_ptr<OwnedColumn> offsets;
    std::unique_ptr<OwnedTable> values;  // may be null if no values provided
};

/// Get the grouped keys and offsets (no values).
std::unique_ptr<GroupByGroupsResult> groupby_get_groups(
    const OwnedTable& keys);

/// Get the grouped keys, offsets, and values.
std::unique_ptr<GroupByGroupsResult> groupby_get_groups_with_values(
    const OwnedTable& keys,
    const OwnedTable& values);

/// Accessor: keys table from groups result.
std::unique_ptr<OwnedTable> groupby_groups_take_keys(GroupByGroupsResult& result);

/// Accessor: offsets column from groups result.
std::unique_ptr<OwnedColumn> groupby_groups_take_offsets(GroupByGroupsResult& result);

/// Accessor: values table from groups result (may be null).
std::unique_ptr<OwnedTable> groupby_groups_take_values(GroupByGroupsResult& result);

// ── GroupBy Replace Nulls ─────────────────────────────────────

/// Replace nulls within groups using forward or backward fill.
/// policy: 0=FORWARD, 1=BACKWARD (one per column in values).
std::unique_ptr<OwnedTable> groupby_replace_nulls(
    const OwnedTable& keys,
    const OwnedTable& values,
    rust::Slice<const int32_t> policies);

} // namespace cudf_shims
