#pragma once

#include <cudf/replace.hpp>
#include <cudf/types.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "scalar_shim.h"

namespace cudf_shims {

/// Replace null values with corresponding values from another column.
std::unique_ptr<OwnedColumn> replace_nulls_column(
    const OwnedColumn& col,
    const OwnedColumn& replacement);

/// Replace null values with a scalar.
std::unique_ptr<OwnedColumn> replace_nulls_scalar(
    const OwnedColumn& col,
    const OwnedScalar& replacement);

/// Replace NaN values with a scalar.
std::unique_ptr<OwnedColumn> replace_nans_scalar(
    const OwnedColumn& col,
    const OwnedScalar& replacement);

/// Replace NaN values with corresponding values from another column.
std::unique_ptr<OwnedColumn> replace_nans_column(
    const OwnedColumn& col,
    const OwnedColumn& replacement);

/// Clamp values to [lo, hi].
std::unique_ptr<OwnedColumn> clamp(
    const OwnedColumn& col,
    const OwnedScalar& lo,
    const OwnedScalar& hi);

/// Normalize -NaN to +NaN and -0.0 to +0.0.
std::unique_ptr<OwnedColumn> normalize_nans_and_zeros(
    const OwnedColumn& col);

/// Replace nulls using a policy (PRECEDING=0, FOLLOWING=1).
std::unique_ptr<OwnedColumn> replace_nulls_policy(
    const OwnedColumn& col,
    int32_t policy);

/// Find and replace all occurrences of old_values with new_values.
std::unique_ptr<OwnedColumn> find_and_replace_all(
    const OwnedColumn& col,
    const OwnedColumn& old_values,
    const OwnedColumn& new_values);

} // namespace cudf_shims
