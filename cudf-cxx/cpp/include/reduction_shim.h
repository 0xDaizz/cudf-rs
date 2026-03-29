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

} // namespace cudf_shims
