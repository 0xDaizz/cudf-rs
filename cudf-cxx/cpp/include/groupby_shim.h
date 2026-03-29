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

} // namespace cudf_shims
