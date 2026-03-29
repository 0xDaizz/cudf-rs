#pragma once

#include <cudf/strings/slice.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_slice(
    const OwnedColumn& col, int32_t start, int32_t stop);

} // namespace cudf_shims
