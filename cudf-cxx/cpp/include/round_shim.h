#pragma once

#include <cudf/round.hpp>
#include <cudf/column/column.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

/// Round a numeric column to the specified number of decimal places.
std::unique_ptr<OwnedColumn> round_column(const OwnedColumn& col, int32_t decimal_places);

} // namespace cudf_shims
