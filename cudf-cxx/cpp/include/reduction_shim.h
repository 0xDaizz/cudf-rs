#pragma once

#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "scalar_shim.h"

namespace cudf_shims {

/// Reduce a column to a single scalar value.
std::unique_ptr<OwnedScalar> reduce(
    const OwnedColumn& col,
    int32_t agg_kind,
    int32_t output_type_id);

/// Prefix scan (cumulative operation).
std::unique_ptr<OwnedColumn> scan(
    const OwnedColumn& col,
    int32_t agg_kind,
    bool inclusive);

/// Segmented reduce within segments defined by offsets.
std::unique_ptr<OwnedColumn> segmented_reduce(
    const OwnedColumn& col,
    const OwnedColumn& offsets,
    int32_t agg_kind,
    int32_t output_type_id,
    bool include_nulls);

/// Compute the minimum and maximum of a column simultaneously.
/// Returns a 2-element OwnedTable where column 0 is the min scalar as a column,
/// and column 1 is the max scalar as a column.
/// Actually returns two scalars packed into a pair struct.
struct MinMaxResult {
    std::unique_ptr<OwnedScalar> min_val;
    std::unique_ptr<OwnedScalar> max_val;
};

std::unique_ptr<MinMaxResult> minmax(const OwnedColumn& col);

/// Accessor for the min scalar from MinMaxResult.
const OwnedScalar& minmax_get_min(const MinMaxResult& result);

/// Accessor for the max scalar from MinMaxResult.
const OwnedScalar& minmax_get_max(const MinMaxResult& result);

/// Move min out of MinMaxResult.
std::unique_ptr<OwnedScalar> minmax_take_min(MinMaxResult& result);

/// Move max out of MinMaxResult.
std::unique_ptr<OwnedScalar> minmax_take_max(MinMaxResult& result);

} // namespace cudf_shims
