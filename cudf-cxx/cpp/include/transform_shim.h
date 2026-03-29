#pragma once

#include <cudf/transform.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

/// Replace NaN values with nulls in a floating-point column.
std::unique_ptr<OwnedColumn> nans_to_nulls(const OwnedColumn& col);

/// Convert a boolean column to a bitmask (host bytes).
rust::Vec<uint8_t> bools_to_mask(const OwnedColumn& col);

} // namespace cudf_shims
