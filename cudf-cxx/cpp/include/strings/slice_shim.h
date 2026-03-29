#pragma once

#include <cudf/strings/slice.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_slice(
    const OwnedColumn& col, int32_t start, int32_t stop);

/// Slice each string using per-row start/stop columns.
std::unique_ptr<OwnedColumn> str_slice_column(
    const OwnedColumn& col, const OwnedColumn& starts, const OwnedColumn& stops);

} // namespace cudf_shims
